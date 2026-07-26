/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2020, 2025
 ************************************************************/

/*
 * Description:
 *
 */

#include <dcg/dcg_fe/dcg_frontend.h>

void DcgFE::computeTranferforDataOp(SuperDsc& mySDsc, DataOpDsc& myDataOpDsc,
                                    int c /*=0*/) {
  DT_CHECK(!isInpFetchNeigh_);
  myDataOpDsc.pcfg_.clear();

  // initilaize Op
  initializeDataDscOp(myDataOpDsc, c);
  verificationCheckDataOpDSC(myDataOpDsc);

  if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx ||
      myDataOpDsc.op->name == OpFuncs::STCDPOpHBM ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNLX ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNHBM) {
    // checks validGap info..
    checkGapInfo(myDataOpDsc);
    // compute transfer function params..
    if (verbose_ > 0) {
      std::cout << "Computing transfer function metaData.." << std::endl;
    }
    if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx ||
        myDataOpDsc.op->name == OpFuncs::ResizeNNLX) {
      createSubPieces((STCDPOpLx*)myDataOpDsc.op);

      if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx) {
        eliminateRedundantTransaction((STCDPOpLx*)myDataOpDsc.op);

        populateCoreProdConsumerList((STCDPOpLx*)myDataOpDsc.op);

        computeInferredSegGroups((STCDPOpLx*)myDataOpDsc.op);

        computeSegCoreGroups((STCDPOpLx*)myDataOpDsc.op);
        findCoreRank((STCDPOpLx*)myDataOpDsc.op);

        // trial act
        mapDtEntryToSenComponent((STCDPOpLx*)myDataOpDsc.op);

        // verification check segCoreGroups
        checkSegCoreProperties((STCDPOpLx*)myDataOpDsc.op);

        // split dtTable entries
        splitDtTableEntriesForSegCoreGrps((STCDPOpLx*)myDataOpDsc.op);
      }

      // final act
      mapDtEntryToSenComponent((STCDPOpLx*)myDataOpDsc.op);

      // reorder subpieces in consumer/producer for segCoreGrps
      // tabulate interleaving subpieces
      if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx)
        reoderSubPieceSegCores((STCDPOpLx*)myDataOpDsc.op);

    } else if (myDataOpDsc.op->name == OpFuncs::STCDPOpHBM ||
               myDataOpDsc.op->name == OpFuncs::ResizeNNHBM) {
      createSubPieces((STCDPOpHBM*)myDataOpDsc.op);
      mapDtEntryToSenComponent((STCDPOpHBM*)myDataOpDsc.op);
    }

    determineInnerLoopOrder((baseSTCDPOp*)myDataOpDsc.op);

    if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx ||
        myDataOpDsc.op->name == OpFuncs::ResizeNNLX) {
      computerGTRInfo((baseSTCDPOp*)myDataOpDsc.op);

      // burst Info
      if (!(((STCDPOpLx*)myDataOpDsc.op)->optSTCDP))
        finalizeBurstInfo((STCDPOpLx*)myDataOpDsc.op);

      // check if out pieces are covered using sub-pieces..
      if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx)
        checkSubPieceCoverage((STCDPOpLx*)myDataOpDsc.op);

      // check if we can use unicast
      checkConvertToUnicast((STCDPOpLx*)myDataOpDsc.op);

      // compute collapse factor for each transfer
      computeInnerLoopCollapseFactor((baseSTCDPOp*)myDataOpDsc.op);

      // compute multicast metaData
      if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx) {
        computeMulticastOptMetadata((STCDPOpLx*)myDataOpDsc.op);

        // mode==3
        if ((*senCompToISAptr).at(SenComponents::L3LU).sysDef.coreArch >=
                IsaCoreGen::RCUDD1A_ISA &&
            dscGlobal->psumRing == "sfpring")
          promoteToMode3((STCDPOpLx*)myDataOpDsc.op);

        // if (verbose_)
        //  dumpMulticastOptMetadata(mySDsc.name_, (STCDPOpLx*)myDataOpDsc.op,
        //                           "multicastMetaData.txt");

        // find interleaving opportunities
        findInterleavingOpportunities((STCDPOpLx*)myDataOpDsc.op);
      }
    } else if (myDataOpDsc.op->name == OpFuncs::STCDPOpHBM ||
               myDataOpDsc.op->name == OpFuncs::ResizeNNHBM) {
      // compute collapse factor for each transfer
      computeInnerLoopCollapseFactor((baseSTCDPOp*)myDataOpDsc.op);

      // execute only if DSC has multicast
      if (((STCDPOpHBM*)myDataOpDsc.op)->reqMulticast)
        computerGTRInfo((baseSTCDPOp*)myDataOpDsc.op, c);
    }

    // gating LX unit codegen
    if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx &&
        ((STCDPOpLx*)myDataOpDsc.op)->gateLXCodeGen) {
      for (int idx = 0; idx < myDataOpDsc.coreIdsUsed_.size(); idx++) {
        auto coreID = myDataOpDsc.coreIdsUsed_[idx];
        if (((STCDPOpLx*)myDataOpDsc.op)->coreIDtoDtKey_LX.count(coreID))
          if (((STCDPOpLx*)myDataOpDsc.op)->coreIDtoDtKey_LX.at(coreID).size())
            ((STCDPOpLx*)myDataOpDsc.op)->coreIDtoDtKey_LX.at(coreID).clear();
      }
    }

    if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx &&
        sysDef.coreArch <= IsaCoreGen::RCUDD1A_ISA) {
      auto stcdp_op = static_cast<STCDPOpLx*>(myDataOpDsc.op);
      // check is the program is prone to SRQ-HB-BUG
      if (isSRQProne(stcdp_op)) {
        // step 0 : check if any dt-entry is shared across both within-lx and
        // L3SU
        for (auto& kv : stcdp_op->coreIDtoDtKey_LX) {
          std::map<int, int> oldToNewPidx;
          for (auto& pidx : kv.second) {
            auto& dt_entry = stcdp_op->dtTable_.at(pidx);
            if (dt_entry.cMemIDs.size() > 1) {
              // we have sharing
              stcdp_op->inpSP_.emplace_back(stcdp_op->inpSP_.at(pidx));  // copy
              auto new_pidx = stcdp_op->inpSP_.size() - 1;
              DT_CHECK(oldToNewPidx.count(pidx) == 0);
              oldToNewPidx[pidx] = new_pidx;
              DT_CHECK(stcdp_op->dtTable_.count(new_pidx) == 0);

              stcdp_op->dtTable_[new_pidx] = dt_entry;  // copy

              // edit the new dt entry
              auto& new_dt_entry = stcdp_op->dtTable_.at(new_pidx);
              new_dt_entry.pIDX = new_pidx;
              new_dt_entry.cIDXs.clear();
              new_dt_entry.cMemIDs.clear();
              new_dt_entry.minCMemID = new_dt_entry.pMemID;
              new_dt_entry.cMemIDs.push_back(kv.first);
              int idx = 0;
              for (auto it = dt_entry.cMemIDs.begin();
                   it != dt_entry.cMemIDs.end();) {
                if (*it == kv.first) {
                  it = dt_entry.cMemIDs.erase(it);
                  new_dt_entry.cIDXs.emplace_back(dt_entry.cIDXs.at(idx));
                  dt_entry.cIDXs.erase(dt_entry.cIDXs.begin() + idx);
                } else {
                  ++it;
                }
                idx++;
              }
              DT_CHECK(new_dt_entry.cIDXs.size() == 1);

              pidx = new_pidx;
            }
          }
        }

        DT_CHECK(!isInpFetchNeigh_);

        int split_factor =
            std::ceil((double)(stcdp_op->maxBeforeTransactions_ + 1) /
                      (double)SRQ_HW_BUG_THRES);
        // step 1 : find splits per dim
        // Only process the first element from coreIDtoDtKey_L3SU
        if (!stcdp_op->coreIDtoDtKey_L3SU.empty()) {
          auto& kv = *stcdp_op->coreIDtoDtKey_L3SU.begin();
          if (!kv.second.empty()) {
            auto& pidx = *kv.second.begin();
            const auto& inp_sp = stcdp_op->inpSP_.at(pidx);
            int net_div = 1;
            for (int idx = stcdp_op->inpLds->layoutDimOrder_.size() - 1;
                 idx >= 0; idx--) {
              const auto& dimname = stcdp_op->inpLds->layoutDimOrder_.at(idx);

              auto dimSize = inp_sp.dimToSize_.at(dimname) /
                             stcdp_op->inpLds->getNumElemsInStick(dimname);
              if (dimSize == 1) continue;

              auto cul_size = net_div * dimSize;
              if (cul_size == split_factor) {
                net_div *= dimSize;
                stcdp_op->split_factor_per_dim[dimname] = dimSize;
                break;
              } else if (cul_size > split_factor) {
                auto min_split = split_factor / net_div;
                bool factor_found = false;
                for (int64_t fac = min_split; fac <= dimSize; fac++) {
                  if ((int64_t)dimSize % fac == 0) {
                    factor_found = true;
                    net_div *= fac;
                    stcdp_op->split_factor_per_dim[dimname] = fac;
                    break;
                  }
                }
                break;
              } else {
                net_div *= dimSize;
                stcdp_op->split_factor_per_dim[dimname] = dimSize;
              }
            }
          }
        }

        // step 2 : need to find samne split for all dims + split
        for (auto& kv : stcdp_op->coreIDtoDtKey_L3SU) {
          for (auto& pidx : kv.second) {
            auto& inp_sp = stcdp_op->inpSP_.at(pidx);

            auto reduceSize = [&](auto& sub_piece) {
              for (const auto& kv_split : stcdp_op->split_factor_per_dim) {
                DT_CHECK((int64_t)sub_piece.dimToSize_.at(kv_split.first) %
                             kv_split.second ==
                         0);
                sub_piece.dimToSize_.at(kv_split.first) /= kv_split.second;
              }
            };
            reduceSize(inp_sp);
            std::set<int> cidx_visited;
            for (auto cidx : stcdp_op->dtTable_.at(pidx).cIDXs) {
              DT_CHECK(cidx_visited.count(cidx) == 0);
              cidx_visited.insert(cidx);
              auto& out_sp = stcdp_op->outSP_.at(cidx);
              reduceSize(out_sp);
            }
          }
        }
      }
    }
  } else if (myDataOpDsc.op->name == OpFuncs::GatherOpHBM) {
    GatherOpHBM* myOp = (GatherOpHBM*)myDataOpDsc.op;

    myOp->computeArrayBAddrConstraints();
    // checks validGap info..
    checkGapInfo(myDataOpDsc);
    // compute transfer function params..
    if (verbose_ > 0) {
      std::cout << "Computing transfer function metaData.." << std::endl;
    }

    // ArrayB, ArrayC split info..
    collectArrayBCPieceInfo(&myDataOpDsc);         // arrayB
    collectArrayBCPieceInfo(&myDataOpDsc, false);  // arrayC
  } else if (myDataOpDsc.op->name == OpFuncs::ScatterOpHBM) {
    ScatterOpHBM* myOp = (ScatterOpHBM*)myDataOpDsc.op;
    myOp->computeArrayBAddrConstraints();
    // checks validGap info..
    checkGapInfo(myDataOpDsc);
    // compute transfer function params..
    if (verbose_ > 0) {
      std::cout << "Computing transfer function metaData.." << std::endl;
    }

    // ArrayA, ArrayB split info..
    collectArrayBCPieceInfo(&myDataOpDsc);         // arrayB
    collectArrayBCPieceInfo(&myDataOpDsc, false);  // arrayC
  } else if (myDataOpDsc.op->name == OpFuncs::APEOpLX ||
             myDataOpDsc.op->name == OpFuncs::APEOpHBM) {
    APEOpLX* myOp = (APEOpLX*)myDataOpDsc.op;

    // compute transfer function params..
    if (verbose_ > 0) {
      std::cout << "Computing Intra-stick transfer function.." << std::endl;
    }
    computeAPEOpStickDimAddrOffset(&myDataOpDsc);
    conputeAPEtransferInfo(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::ReStickifyOpLx ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpHBM) {
    baseStickOp* myOp = (baseStickOp*)myDataOpDsc.op;

    // compute transfer function params..
    if (verbose_ > 0) {
      std::cout << "Computing Re-StickifyOp transfer function.." << std::endl;
    }

    // determine if resticky also need zero padding
    myOp->reqGapInsertion = reqIntraStickForZeroPadding(myOp);
    if (myOp->reqGapInsertion) {
      for (const auto& mapkv : myOp->lxAddrOpConsts)
        DT_CHECK(mapkv.second >= 0);
    }

    // check piece definition
    checkPieceInfo(myOp);

    // compute offset
    computeAPEOpStickDimAddrOffset(&myDataOpDsc);
    computeInpNonStickDimOffsets(myOp);  // addOffset for Non-stick dim

    if (sysDef.coreArch < IsaCoreGen::RCUDD1A_ISA)
      computeStickDimLoopInfo(&myDataOpDsc);

    // conputeAPEtransferInfo(&myDataOpDsc); //cannot use due to ibuff
    // violation
  } else if (myDataOpDsc.op->name == OpFuncs::XRFWriteHBM ||
             myDataOpDsc.op->name == OpFuncs::XRFWriteLX) {
    // checks validGap info..
    checkGapInfo(myDataOpDsc);

    if (verbose_ > 0)
      std::cout << "Computing transfer function metaData.." << std::endl;

    computeXRFTransferInfo(&myDataOpDsc);
    createPcfgsXRFWrite(&myDataOpDsc);

  } else if (myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM) {
    baseStickOp* myOp = (baseStickOp*)myDataOpDsc.op;

    // determine re-stickify subOp type
    determineSubOp(&myDataOpDsc);

    // compute transfer function params..
    if (verbose_ > 0) {
      std::cout << "Computing Re-StickifyOpWithPT (Special re-stickify) "
                   "transfer function.."
                << std::endl;
    }

    myOp->reqGapInsertion = false;  // no zero padding

    // check piece definition
    checkPieceInfo(myOp);

    //// compute offset
    computeAPEOpStickDimAddrOffset(&myDataOpDsc);
    computeInpNonStickDimOffsets(myOp);  // addOffset for Non-stick dim
  } else if (myDataOpDsc.op->name == OpFuncs::Nop) {
  } else if (myDataOpDsc.op->name == OpFuncs::StickifyOpHBM) {
    if (verbose_ > 0) {
      std::cout << "Creating PCFG for DataDsc.." << std::endl;
    }

    // EAR offset req.
    computeEAROffsetAdjustment(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::ConstPadOpLX ||
             myDataOpDsc.op->name == OpFuncs::ConstPadOpHBM) {
    auto* myOp = (ConstPadOpLX*)myDataOpDsc.op;
    fillPiecePerCore(myOp);
    computeCoreletWork(myOp);
  } else if (myDataOpDsc.op->name == OpFuncs::ITOF) {
    auto* myOp = (baseITOFOp*)myDataOpDsc.op;
    fillPiecePerCore(myOp);
  } else if (myDataOpDsc.op->name == OpFuncs::ITOFHBM) {
    auto* myOp = (baseITOFOp*)myDataOpDsc.op;
    fillPiecePerCore(myOp);
  } else {
    // add new ops here..
    DT_CHECK(0);
  }

  if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx &&
      !myDataOpDsc.labeledDs_.empty()) {
    const auto& dims = myDataOpDsc.labeledDs_.front().dimToLayoutSize_;
    const bool isGraniteP06 =
        dims.count("x") && dims.count("y") && dims.count("mb") &&
        dims.count("in") && dims.at("x") == 8 && dims.at("y") == 512 &&
        dims.at("mb") == 4 && dims.at("in") == 128;
    if (isGraniteP06) {
      const auto* p06Op = static_cast<STCDPOpLx*>(myDataOpDsc.op);
      for (const auto& [key, transfer] : p06Op->dtTable_) {
        std::cout << "P06_STCDP_FINAL key=" << key << " loop=";
        for (const auto& dim : transfer.loopOrder) {
          std::cout << dim << ",";
        }
        std::cout << " collapse=" << transfer.collapseFactor
                  << " burst=" << transfer.useBurst
                  << " producer=" << transfer.pMemID << " consumers=";
        for (const auto consumer : transfer.cMemIDs) {
          std::cout << consumer << ",";
        }
        std::cout << std::endl;
      }
      mySDsc.exportJson("p06_after_transfer_" + mySDsc.name_ + ".json");
    }
  }
}

void DcgFE::checkGapInfo(DataOpDsc& myDataOpDsc) {
  for (auto& myldsInfo : myDataOpDsc.labeledDs_) {
    for (auto& mydim : myldsInfo.layoutDimOrder_) {
      for (auto& myPiece : myldsInfo.pieces_) {
        // std::cout << myldsInfo.ldsName_ << " : " << myPiece.first << " : "
        // << mydim<< "\n";
        DT_CHECK(myPiece.second.validGap_.at(mydim).size());
        DT_CHECK(myPiece.second.validGap_.at(mydim).size() <= 2);
        if (myPiece.second.validGap_.at(mydim).size() == 2) {
          DT_CHECK(myPiece.second.validGap_.at(mydim)[0].first == 0);
          DT_CHECK(myPiece.second.validGap_.at(mydim)[1].first != 0);
          DT_CHECK(myPiece.second.validGap_.at(mydim)[0].second != 0);
        } else {
          DT_CHECK(myPiece.second.validGap_.at(mydim)[0].first != 0);
        }
      }

      // check lds
      DT_CHECK(myldsInfo.validGap_.at(mydim).size());
      DT_CHECK(myldsInfo.validGap_.at(mydim).size() <= 2);
      if (myldsInfo.validGap_.at(mydim).size() == 2) {
        DT_CHECK(myldsInfo.validGap_.at(mydim)[0].first == 0);
        DT_CHECK(myldsInfo.validGap_.at(mydim)[1].first != 0);
        DT_CHECK(myldsInfo.validGap_.at(mydim)[0].second != 0);
      } else {
        DT_CHECK(myldsInfo.validGap_.at(mydim)[0].first != 0);
      }

      int dimLen = 0;
      for (auto& vgPair : myldsInfo.validGap_.at(mydim)) {
        dimLen += vgPair.first + vgPair.second;
      }
      DT_CHECK(dimLen == myldsInfo.dimToLayoutSize_.at(mydim));
    }
  }
}

void DcgFE::verificationCheckDataOpDSC(DataOpDsc& myDataOpDsc) {
  DT_CHECK(myDataOpDsc.coreIdsUsed_.size());
  // check for primaryDs_ dimNames are in dimPool_
  for (auto const& myPdsInfo : myDataOpDsc.primaryDs_) {
    for (auto const& dimName : myPdsInfo.second.dimNames) {
      if (!myDataOpDsc.dimPool_.count(dimName)) {
        DCGUtils::error("dimName=" + dimName + " not present in dimPool_");
      }
    }
  }

  // legality check on supported OpFunc
  if (!is_any_of(myDataOpDsc.op->name, OpFuncs::STCDPOpLx, OpFuncs::STCDPOpHBM,
                 OpFuncs::ResizeNNHBM, OpFuncs::ResizeNNLX,
                 OpFuncs::GatherOpHBM, OpFuncs::APEOpLX, OpFuncs::APEOpHBM,
                 OpFuncs::ReStickifyOpLx, OpFuncs::ReStickifyOpHBM,
                 OpFuncs::ReStickifyOpWithPTLx, OpFuncs::ReStickifyOpWithPTHBM,
                 OpFuncs::XRFWriteHBM, OpFuncs::XRFWriteLX, OpFuncs::Nop,
                 OpFuncs::StickifyOpHBM, OpFuncs::ConstPadOpLX,
                 OpFuncs::ConstPadOpHBM, OpFuncs::ScatterOpHBM, OpFuncs::ITOF,
                 OpFuncs::ITOFHBM))
    DCGUtils::error("Unsupported Op : " +
                    EnumsConversion::opFuncsToString.at(myDataOpDsc.op->name));

  // some generic checks
  if (dscGlobal->dtVersion >= 2) {
    for (auto& lds : myDataOpDsc.labeledDs_) {
      for (auto& pkv : lds.pieces_) {
        for (auto& skv : pkv.second.dimToSize_)
          DT_CHECK(skv.second <= lds.dimToLayoutSize_.at(skv.first));
      }
    }
  }

  // OpFuncs check
  if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx ||
      myDataOpDsc.op->name == OpFuncs::STCDPOpHBM ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNLX ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNHBM) {
    baseSTCDPOp* myOp = (baseSTCDPOp*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    DT_CHECK(myOp->inpLds == &myDataOpDsc.labeledDs_[0]);
    DT_CHECK(myOp->outLds == &myDataOpDsc.labeledDs_[1]);
    // check stick def is same
    DT_CHECK(myOp->outLds->stickDimOrder_ == myOp->inpLds->stickDimOrder_);
    DT_CHECK(myOp->outLds->dimToStickSize_ == myOp->inpLds->dimToStickSize_);
    DT_CHECK(myOp->outLds->dimToStickSize_.size() ==
             myOp->outLds->stickDimOrder_.size());
  } else if (myDataOpDsc.op->name == OpFuncs::APEOpLX ||
             myDataOpDsc.op->name == OpFuncs::APEOpHBM ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpLx ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpHBM ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM) {
    baseStickOp* myOp = (baseStickOp*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    DT_CHECK(myOp->inpLds == &myDataOpDsc.labeledDs_[0]);
    DT_CHECK(myOp->outLds == &myDataOpDsc.labeledDs_[1]);

    if (myDataOpDsc.op->name == OpFuncs::APEOpLX ||
        myDataOpDsc.op->name == OpFuncs::APEOpHBM) {
      // check stick def is same
      DT_CHECK(myOp->outLds->stickDimOrder_ == myOp->inpLds->stickDimOrder_);
      DT_CHECK(myOp->outLds->dimToStickSize_ == myOp->inpLds->dimToStickSize_);
      DT_CHECK(myOp->outLds->dimToStickSize_.size() ==
               myOp->outLds->stickDimOrder_.size());

      for (const auto& mapkv : myOp->lxAddrOpConsts)
        DT_CHECK(mapkv.second >= 0);
    }

    // put DT_CHECKs here for re-stickifyWithPT
    if (myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx ||
        myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM) {
      auto& coreLetWorkDs =
          (myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx)
              ? ((ReStickifyOpWithPTLx*)myOp)->coreLetWorkDs
              : ((ReStickifyOpWithPTHBM*)myOp)->coreLetWorkDs;

      // corelet work divison checks
      if (coreLetWorkDs.numClToUse == 1) {
        if (coreLetWorkDs.defaultClId == 0) {
          DT_CHECK(coreLetWorkDs.cl0ToLxOffsetLU != -1);
          DT_CHECK(coreLetWorkDs.cl0ToLxOffsetSU != -1);
        } else {
          DT_CHECK(coreLetWorkDs.cl1ToLxOffsetLU != -1);
          DT_CHECK(coreLetWorkDs.cl1ToLxOffsetSU != -1);
        }
      } else if (coreLetWorkDs.numClToUse == 2) {
        DT_CHECK(coreLetWorkDs.cl0ToLxOffsetLU != -1);
        DT_CHECK(coreLetWorkDs.cl1ToLxOffsetLU != -1);
        DT_CHECK(coreLetWorkDs.cl0ToLxOffsetSU != -1);
        DT_CHECK(coreLetWorkDs.cl1ToLxOffsetSU != -1);
        DT_CHECK(
            myOp->inpLds->dimToLayoutSize_.count(coreLetWorkDs.workSplitDim));
      } else {
        DT_CHECK(0);
      }

      if (myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx) {
        if (((ReStickifyOpWithPTLx*)myOp)->doInPlace) {
          DT_CHECK(myOp->inpLds->stickDimOrder_.size() == 1);
          DT_CHECK(myOp->outLds->stickDimOrder_.size() == 1);
          DT_CHECK(myOp->inpLds->layoutDimOrder_.at(0) ==
                   myOp->inpLds->stickDimOrder_.at(0));
          DT_CHECK(myOp->inpLds->layoutDimOrder_.at(1) ==
                   myOp->outLds->stickDimOrder_.at(0));
          DT_CHECK(myOp->inpLds->layoutDimOrder_ ==
                   myOp->outLds->layoutDimOrder_);
          DT_CHECK(myOp->inpLds->dimToLayoutSize_.at(
                       myOp->inpLds->layoutDimOrder_.at(0)) <=
                   myOp->inpLds->dimToStickSize_.at(
                       myOp->inpLds->layoutDimOrder_.at(0)) *
                       2);

          DT_CHECK(coreLetWorkDs.numClToUse == 2);
          DT_CHECK(coreLetWorkDs.workSplitDim ==
                   myOp->inpLds->stickDimOrder_.at(0));
        }
      }
    }

  } else if (myDataOpDsc.op->name == OpFuncs::XRFWriteLX ||
             myDataOpDsc.op->name == OpFuncs::XRFWriteHBM) {
    baseXRFOp* myOp = (baseXRFOp*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    DT_CHECK(myOp->inpLds == &myDataOpDsc.labeledDs_[0]);
    DT_CHECK(myOp->outLds == &myDataOpDsc.labeledDs_[1]);
    // check stick def is same
    DT_CHECK(myOp->outLds->stickDimOrder_ == myOp->inpLds->stickDimOrder_);
    DT_CHECK(myOp->outLds->dimToStickSize_ == myOp->inpLds->dimToStickSize_);
    DT_CHECK(myOp->outLds->dimToStickSize_.size() ==
             myOp->outLds->stickDimOrder_.size());
    // coreletId check
    DT_CHECK(myOp->coreletId == 0 || myOp->coreletId == 1);

    // check outPiece
    for (auto& coreID : myDataOpDsc.coreIdsUsed_) {
      if (myDataOpDsc.op->name == OpFuncs::XRFWriteHBM)
        DT_CHECK(((XRFWriteHBM*)myOp)
                     ->coreIDtoANInfo.at(coreID)
                     .outPieceOrder.size() == ((XRFWriteHBM*)myOp)
                                                  ->coreIDtoANInfo.at(coreID)
                                                  .inpPieceOrder.size());
    }
  } else if (myDataOpDsc.op->name == OpFuncs::ConstPadOpLX) {
    auto* myOp = (ConstPadOpLX*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 1);
    DT_CHECK(myOp->Lds == &myDataOpDsc.labeledDs_[0]);
  } else if (myDataOpDsc.op->name == OpFuncs::ConstPadOpHBM) {
    auto* myOp = (ConstPadOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    DT_CHECK(myOp->Lds == &myDataOpDsc.labeledDs_[0]);
    DT_CHECK(myOp->ldsHBM == &myDataOpDsc.labeledDs_[1]);
  } else if (is_any_of(myDataOpDsc.op->name, OpFuncs::ITOF, OpFuncs::ITOFHBM)) {
    auto* myOp = (baseITOFOp*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 5);
    DT_CHECK(myOp->input_ == &myDataOpDsc.labeledDs_[0]);
    DT_CHECK(myOp->inp_mask_ == &myDataOpDsc.labeledDs_[2]);
    DT_CHECK(myOp->scale_ == &myDataOpDsc.labeledDs_[1]);
    DT_CHECK(myOp->scale_mask_ == &myDataOpDsc.labeledDs_[3]);
    DT_CHECK(myOp->output_ == &myDataOpDsc.labeledDs_[4]);
  }

  auto checkAnalyticalFillingForNBuffer = [](auto& coreIDtoANInfo) {
    // should be N-buffer use case
    for (const auto& c_am : coreIDtoANInfo) {
      if (c_am.second.inpPieceOrder.size() == c_am.second.outPieceOrder.size())
        continue;

      DT_CHECK(c_am.second.isAnalyticalMode);
      bool is_NB_L3LU = (c_am.second.inpPieceOrder.size() > 1);
      bool is_NB_L3SU = (c_am.second.outPieceOrder.size() > 1);

      DT_CHECK(is_NB_L3LU || is_NB_L3SU);
      DT_CHECK(!(is_NB_L3LU && is_NB_L3SU));

      if (is_NB_L3LU) {
        DT_CHECK(c_am.second.getAddrInfo().count(SenComponents::L3LU));
        DT_CHECK(c_am.second.getAddrInfo().count(SenComponents::LXSU0));
        DT_CHECK(c_am.second.getAddrInfo().at(SenComponents::L3LU)->getType() ==
                 AddrInfo::TYPE::STRIDE);
        DT_CHECK(
            c_am.second.getAddrInfo().at(SenComponents::LXSU0)->getType() ==
            AddrInfo::TYPE::STRIDE);
      } else {
        DT_CHECK(c_am.second.getAddrInfo().count(SenComponents::L3SU));
        DT_CHECK(c_am.second.getAddrInfo().count(SenComponents::LXLU0));
        DT_CHECK(c_am.second.getAddrInfo().at(SenComponents::L3SU)->getType() ==
                 AddrInfo::TYPE::STRIDE);
        DT_CHECK(
            c_am.second.getAddrInfo().at(SenComponents::LXLU0)->getType() ==
            AddrInfo::TYPE::STRIDE);
      }
    }
  };

  // check number of pieces
  if (myDataOpDsc.op->name == OpFuncs::STCDPOpHBM ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNHBM) {
    STCDPOpHBM* myOp = (STCDPOpHBM*)myDataOpDsc.op;
    if (myOp->outLds->pieces_.size() && myOp->inpLds->pieces_.size())
      if (myOp->outLds->pieces_.size() != myOp->inpLds->pieces_.size()) {
        checkAnalyticalFillingForNBuffer(myOp->coreIDtoANInfo);
      }
  } else if (myDataOpDsc.op->name == OpFuncs::APEOpHBM ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpHBM ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM) {
    APEOpHBM* myOp = static_cast<APEOpHBM*>(myDataOpDsc.op);
    DT_CHECK(!myOp->outLds->pieces_.empty() && !myOp->inpLds->pieces_.empty());
    if (myOp->outLds->pieces_.size() != myOp->inpLds->pieces_.size()) {
      checkAnalyticalFillingForNBuffer(myOp->coreIDtoANInfo);
    }
  } else if (myDataOpDsc.op->name == OpFuncs::XRFWriteHBM) {
    XRFWriteHBM* myOp = (XRFWriteHBM*)myDataOpDsc.op;
    DT_CHECK(myOp->outLds->pieces_.size() == myOp->inpLds->pieces_.size());
  } else if (is_any_of(myDataOpDsc.op->name, OpFuncs::ITOF, OpFuncs::ITOFHBM)) {
    auto* myOp = (baseITOFOp*)myDataOpDsc.op;
    DT_CHECK(myOp->input_->pieces_.size() == myOp->output_->pieces_.size());
    DT_CHECK(myOp->scale_->pieces_.size());
    DT_CHECK(myOp->inp_mask_->pieces_.size());
    DT_CHECK(myOp->scale_mask_->pieces_.size());
  }

  // check for labeledDs_ are in derived from primaryDs_
  std::set<std::string> inpDims;
  std::set<std::string> outDims;
  for (int idx = 0; idx < myDataOpDsc.labeledDs_.size(); idx++) {
    auto const& myldsInfo = myDataOpDsc.labeledDs_[idx];
    if (!myDataOpDsc.primaryDs_.count(myldsInfo.pdsName_)) {
      DCGUtils::error("primaryDs with name_=" + myldsInfo.pdsName_ +
                      " not found");
    }
    if (!DCGUtils::isSubSet(
            myldsInfo.layoutDimOrder_,
            myDataOpDsc.primaryDs_[myldsInfo.pdsName_].dimNames))
      DCGUtils::error("labedDs=" + myldsInfo.ldsName_ +
                      " layoutDimOrder_ used incorrect DimName");
    if (!DCGUtils::isSubSet(
            myldsInfo.stickDimOrder_,
            myDataOpDsc.primaryDs_[myldsInfo.pdsName_].dimNames))
      DCGUtils::error("labedDs=" + myldsInfo.ldsName_ +
                      " stickDimOrder_ used incorrect DimName");
    if (!DCGUtils::isSubSet(
            myldsInfo.dimToLayoutSize_,
            myDataOpDsc.primaryDs_[myldsInfo.pdsName_].dimNames))
      DCGUtils::error("labedDs=" + myldsInfo.ldsName_ +
                      " dimToLayoutSize_ used incorrect DimName");
    if (!DCGUtils::isSubSet(
            myldsInfo.dimToStickSize_,
            myDataOpDsc.primaryDs_[myldsInfo.pdsName_].dimNames))
      DCGUtils::error("labedDs=" + myldsInfo.ldsName_ +
                      " dimToStickSize_ used incorrect DimName");

    if (myldsInfo.dimToLayoutSize_.size() != myldsInfo.layoutDimOrder_.size())
      DCGUtils::error("labedDs=" + myldsInfo.ldsName_ +
                      " dimToLayoutSize_.size() != layoutDimOrder_.size()");
    if (myldsInfo.dimToStickSize_.size() != myldsInfo.stickDimOrder_.size())
      DCGUtils::error("labedDs=" + myldsInfo.ldsName_ +
                      " dimToStickSize_.size() != stickDimOrder_.size()");

    for (auto const& dimName : myldsInfo.layoutDimOrder_) {
      idx == 0 ? inpDims.insert(dimName) : outDims.insert(dimName);
      // validGap info
      if (myDataOpDsc.op->name != OpFuncs::StickifyOpHBM)
        DT_CHECK(myldsInfo.validGap_.count(dimName));
    }

    // check PlacementInfo
    for (auto const& myPiece : myldsInfo.pieces_) {
      if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx ||
          myDataOpDsc.op->name == OpFuncs::ResizeNNLX ||
          myDataOpDsc.op->name == OpFuncs::ReStickifyOpLx ||
          myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx ||
          myDataOpDsc.op->name == OpFuncs::APEOpLX ||
          myDataOpDsc.op->name == OpFuncs::ITOF) {
        DT_CHECK(myPiece.second.placement.size() == 1);
        DT_CHECK(myPiece.second.placement.count(SenComponents::LX) == 1);
      } else if (myDataOpDsc.op->name == OpFuncs::STCDPOpHBM ||
                 myDataOpDsc.op->name == OpFuncs::ResizeNNHBM ||
                 myDataOpDsc.op->name == OpFuncs::APEOpHBM ||
                 myDataOpDsc.op->name == OpFuncs::ReStickifyOpHBM ||
                 myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM ||
                 myDataOpDsc.op->name == OpFuncs::ITOFHBM) {
        DT_CHECK(myPiece.second.placement.count(SenComponents::LX) == 1);
      } else if (myDataOpDsc.op->name == OpFuncs::XRFWriteLX ||
                 myDataOpDsc.op->name == OpFuncs::XRFWriteHBM) {
        if (idx == 0) {  // inputLds
          DT_CHECK(myPiece.second.placement.count(SenComponents::LX) == 1);
          if (myDataOpDsc.op->name == OpFuncs::XRFWriteLX)
            DT_CHECK(myPiece.second.placement.size() == 1);
        } else {
          DT_CHECK(myPiece.second.placement.size() == 1);
          auto it = myPiece.second.placement.begin();
          baseXRFOp* myOp = (baseXRFOp*)myDataOpDsc.op;
          if (myOp->coreletId == 0)
            DT_CHECK(it->first >= 23 &&
                     it->first <=
                         30);  // PTROW0_0 to PTROW7_0 in SenComponents enum
          else if (myOp->coreletId == 1)
            DT_CHECK(it->first >= 31 &&
                     it->first <=
                         38);  // PTROW0_1 to PTROW7_1 in SenComponents enum
          else
            DT_CHECK(0);
        }
      }
      for (auto const& myPlacement : myPiece.second.placement) {
        DT_CHECK(myPlacement.second.MemId().getSingleData().size() ==
                 myPlacement.second.StartAddr().getSingleData().size());
        if (myPlacement.first == SenComponents::HBM) {
          DT_CHECK(myPlacement.second.MemId().getSingleData().size() == 1);
          DT_CHECK(myPlacement.second.MemId().getSingleData().at(0) == -1);
        }
      }

      // check for piece name
      DT_CHECK(myPiece.first == myPiece.second.key_);
    }

    // stickSize
    double stickDinLen = myldsInfo.wordLength;
    for (auto& len : myldsInfo.dimToStickSize_) {
      std::string dimName = len.first;
      DT_CHECK(DCGUtils::getValCount(myldsInfo.stickDimOrder_, dimName) == 1);
      stickDinLen *= len.second;
    }
    DT_CHECK(stickDinLen <=
             sysDef.bytesPerStick);  // stickSize=SenStickSize bytes
  }

  // check if the dims of output labeledDs_ is present in input labeledDs_
  // if dim is not present relation with one of the input Dim is required..
  if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx ||
      myDataOpDsc.op->name == OpFuncs::STCDPOpHBM ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNLX ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNHBM ||
      myDataOpDsc.op->name == OpFuncs::APEOpLX ||
      myDataOpDsc.op->name == OpFuncs::APEOpHBM ||
      myDataOpDsc.op->name == OpFuncs::ReStickifyOpLx ||
      myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx ||
      myDataOpDsc.op->name == OpFuncs::ReStickifyOpHBM ||
      myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM ||
      myDataOpDsc.op->name == OpFuncs::XRFWriteLX ||
      myDataOpDsc.op->name == OpFuncs::XRFWriteHBM) {
    for (auto const& outDimName : outDims) {
      if (!inpDims.count(outDimName)) {
        DCGUtils::error("dimName=" + outDimName +
                        " present in output but not in any input\n");
      }
    }
  } else if (myDataOpDsc.op->name == OpFuncs::GatherOpHBM) {
    GatherOpHBM* myOp = (GatherOpHBM*)myDataOpDsc.op;
    GatherOpHBM* myOpHBM = (GatherOpHBM*)myDataOpDsc.op;
    // std::string bufferDim_ = myOp->bufferDim_;
    DT_CHECK(myOp->arrayA->layoutDimOrder_.size() >= 2);
    DT_CHECK(myOp->arrayB->layoutDimOrder_.size() <= 2);
    DT_CHECK(myOp->arrayC->layoutDimOrder_.size() >=
             myOp->arrayA->layoutDimOrder_.size());
    DT_CHECK(myOp->arrayC->layoutDimOrder_.size() ==
             myOp->arrayA->layoutDimOrder_.size() +
                 myOp->arrayB->layoutDimOrder_.size() - 1);
    DT_CHECK(myOp->arrayB->stickDimOrder_.size() >= 1);
    DT_CHECK(myOp->arrayA->stickDimOrder_.size() >= 1);
    DT_CHECK(myOp->arrayC->stickDimOrder_.size() >= 1);
    DT_CHECK(myOp->arrayA->stickDimOrder_.size() ==
             myOp->arrayC->stickDimOrder_.size());
    // DT_CHECK(myOp->arrayA->stickDimOrder_[0] ==
    // myOp->arrayC->layoutDimOrder_[0]);
    for (auto& dimName : myOp->arrayA->stickDimOrder_)
      DT_CHECK(myOp->arrayC->dimToStickSize_.count(dimName));

    DT_CHECK(myOp->arrayA->dimToStickSize_.count(myOp->gatherScatterDim) == 0);
    DT_CHECK(myOp->arrayC->dimToStickSize_.count(myOp->gatherScatterDim) == 0);
    DT_CHECK(myOp->arrayB->dimToStickSize_.count(myOp->gatherScatterDim));
    DT_CHECK(myOp->arrayB->stickDimOrder_[0] ==
             myOp->arrayB->layoutDimOrder_[0]);  // stick should be inner most

    for (auto& dimName : myOp->arrayA->layoutDimOrder_) {
      DT_CHECK(myOp->arrayC->dimToLayoutSize_.count(dimName));
      if (myOp->arrayB->dimToLayoutSize_.count(dimName) == 0)
        DT_CHECK(myOp->arrayC->dimToLayoutSize_.at(dimName) ==
                 myOp->arrayA->dimToLayoutSize_.at(dimName));
    }

    // check validGaps
    bool is_arrayC_hbm_pinned = myOp->arrayC->isHbmPinned();
    for (int i = 0; i < 2; i++) {
      auto* lds = i == 0 ? myOp->arrayA : myOp->arrayC;
      for (auto& dimName : lds->layoutDimOrder_) {
        DT_CHECK(lds->validGap_.at(dimName).size() == 1);

        DT_CHECK(lds->dimToLayoutSize_.count(dimName));
        DT_CHECK(lds->validGap_.at(dimName)[0].first <=
                 lds->dimToLayoutSize_.at(dimName));
        DT_CHECK(lds->validGap_.at(dimName)[0].second +
                     lds->validGap_.at(dimName)[0].first ==
                 lds->dimToLayoutSize_.at(dimName));

        for (const auto& p : lds->pieces_) {
          DT_CHECK(p.second.validGap_.at(dimName).size() == 1);
          DT_CHECK(p.second.dimToSize_.at(dimName) +
                   DCGUtils::getTotalElements(p.second.validGap_.at(dimName)));

          if (is_arrayC_hbm_pinned || i != 1) {
            DT_CHECK(p.second.validGap_.at(dimName)[0].second == 0);  // no gaps
            DT_CHECK(p.second.dimToSize_.count(dimName));
            DT_CHECK(p.second.validGap_.at(dimName)[0].first ==
                     p.second.dimToSize_.at(dimName));
          }
        }
      }
    }

    auto validSize_arrayB =
        PieceInfo::getTotalValidsInVG(myOp->arrayB->validGap_);
    auto validSize_arrayC =
        PieceInfo::getTotalValidsInVG(myOp->arrayC->validGap_);
    for (auto& dimName : myOp->arrayB->layoutDimOrder_) {
      DT_CHECK(myOp->arrayC->dimToLayoutSize_.count(dimName));
      DT_CHECK(validSize_arrayC.at(dimName) == validSize_arrayB.at(dimName));
      DT_CHECK(myOp->arrayB->dimToLayoutSize_.at(dimName) >=
               validSize_arrayB.at(dimName));
    }
    DT_CHECK(myOp->gatherScatterDim != "null");

    // check each piece of arrayB
    for (const auto& pInfo : myOp->arrayB->pieces_) {
      // valid+Gap == dimSize
      DT_CHECK(pInfo.second.dimToSize_ ==
               PieceInfo::getTotalElementsInVG(pInfo.second.validGap_));

      // product of valids across dim
      int product = 1;
      for (const auto& mapkv :
           PieceInfo::getTotalValidsInVG(pInfo.second.validGap_))
        product *= mapkv.second;

      DT_CHECK(product <= sysDef.bytesPerStick / 4);
    }

    // pieces in a core are all same
    for (const auto& kv : myOp->coreIDtoANInfo) {
      // input pieces
      if (kv.second.inpPieceOrder.size() > 1) {
        auto firstPiece = kv.second.inpPieceOrder.at(0);
        for (const auto& iPiece : kv.second.inpPieceOrder) {
          DT_CHECK(myOp->arrayB->pieces_.count(iPiece));
          DT_CHECK(myOp->arrayB->pieces_.at(iPiece).dimToSize_ ==
                   myOp->arrayB->pieces_.at(firstPiece).dimToSize_);
          DT_CHECK(myOp->arrayB->pieces_.at(iPiece).validGap_ ==
                   myOp->arrayB->pieces_.at(firstPiece).validGap_);
        }
      }

      // input pieces
      if (kv.second.outPieceOrder.size() > 1) {
        auto firstPiece = kv.second.outPieceOrder.at(0);
        for (const auto& oPiece : kv.second.outPieceOrder) {
          DT_CHECK(myOp->arrayC->pieces_.count(oPiece));
          DT_CHECK(myOp->arrayC->pieces_.at(oPiece).dimToSize_ ==
                   myOp->arrayC->pieces_.at(firstPiece).dimToSize_);
          DT_CHECK(myOp->arrayC->pieces_.at(oPiece).validGap_ ==
                   myOp->arrayC->pieces_.at(firstPiece).validGap_);
        }
      }
    }
  } else if (myDataOpDsc.op->name == OpFuncs::ScatterOpHBM) {
    ScatterOpHBM* s_op = (ScatterOpHBM*)myDataOpDsc.op;

    DT_CHECK(s_op->arrayC->layoutDimOrder_.size() >= 2);
    DT_CHECK(s_op->arrayB->layoutDimOrder_.size() <= 2);
    DT_CHECK(s_op->arrayA->layoutDimOrder_.size() >=
             s_op->arrayA->layoutDimOrder_.size());
    DT_CHECK(s_op->arrayA->layoutDimOrder_.size() ==
             s_op->arrayC->layoutDimOrder_.size() +
                 s_op->arrayB->layoutDimOrder_.size() - 1);
    DT_CHECK(s_op->arrayB->stickDimOrder_.size() >= 1);
    DT_CHECK(s_op->arrayA->stickDimOrder_.size() >= 1);
    DT_CHECK(s_op->arrayC->stickDimOrder_.size() >= 1);
    DT_CHECK(s_op->arrayA->stickDimOrder_.size() ==
             s_op->arrayC->stickDimOrder_.size());
    DT_CHECK(s_op->gatherScatterDim != "null");

    for (auto& dimName : s_op->arrayC->stickDimOrder_)
      DT_CHECK(s_op->arrayA->dimToStickSize_.count(dimName));

    DT_CHECK(s_op->arrayA->dimToStickSize_.count(s_op->gatherScatterDim) == 0);
    DT_CHECK(s_op->arrayC->dimToStickSize_.count(s_op->gatherScatterDim) == 0);
    DT_CHECK(s_op->arrayB->dimToStickSize_.count(s_op->gatherScatterDim));
    DT_CHECK(s_op->arrayB->stickDimOrder_[0] ==
             s_op->arrayB->layoutDimOrder_[0]);  // stick should be inner most

    // enforce no layout change
    // first prepare non 1 layout order
    std::vector<std::string> nonone_layout_order_arrayA;
    std::vector<std::string> nonone_layout_order_arrayC;

    for (int a_idx = 0; a_idx < s_op->arrayA->layoutDimOrder_.size(); a_idx++) {
      auto dimname = s_op->arrayA->layoutDimOrder_.at(a_idx);
      if (s_op->arrayC->dimToLayoutSize_.count(dimname) == 0) continue;
      if (s_op->arrayA->dimToLayoutSize_.at(dimname) == 1) {
        continue;
      }
      nonone_layout_order_arrayA.emplace_back(dimname);
    }

    for (int c_idx = 0; c_idx < s_op->arrayC->layoutDimOrder_.size(); c_idx++) {
      auto dimname = s_op->arrayC->layoutDimOrder_.at(c_idx);
      if (s_op->arrayA->dimToLayoutSize_.count(dimname) == 0) {
        continue;
      }
      if (s_op->arrayA->dimToLayoutSize_.at(dimname) == 1) {
        continue;
      }
      nonone_layout_order_arrayC.emplace_back(dimname);
    }

    DT_CHECK_MSG(nonone_layout_order_arrayC == nonone_layout_order_arrayA,
                 "Layout Order should match");

    for (auto& dimName : s_op->arrayC->layoutDimOrder_) {
      DT_CHECK(s_op->arrayA->dimToLayoutSize_.count(dimName));
      if (s_op->arrayB->dimToLayoutSize_.count(dimName) == 0)
        DT_CHECK(PieceInfo::getTotalValid(s_op->arrayC->validGap_.at(
                     dimName)) == s_op->arrayA->dimToLayoutSize_.at(dimName));
    }

    // check validGaps
    for (int i = 0; i < 2; i++) {
      auto* lds = i == 0 ? s_op->arrayA : s_op->arrayC;
      for (auto& dimName : lds->layoutDimOrder_) {
        DT_CHECK(lds->validGap_.at(dimName).size() == 1);
        if (i == 0)
          DT_CHECK(lds->validGap_.at(dimName)[0].second == 0);  // no gaps
        DT_CHECK(lds->dimToLayoutSize_.count(dimName));
        if (i == 0)
          DT_CHECK(lds->validGap_.at(dimName)[0].first ==
                   lds->dimToLayoutSize_.at(dimName));
        else
          DT_CHECK(lds->validGap_.at(dimName)[0].first +
                       lds->validGap_.at(dimName)[0].second ==
                   lds->dimToLayoutSize_.at(dimName));

        for (const auto& p : lds->pieces_) {
          DT_CHECK(p.second.validGap_.at(dimName).size() == 1);
          if (i == 0)
            DT_CHECK(p.second.validGap_.at(dimName)[0].second == 0);  // no gaps
          DT_CHECK(p.second.dimToSize_.count(dimName));
          if (i == 0) {
            DT_CHECK(p.second.validGap_.at(dimName)[0].first ==
                     p.second.dimToSize_.at(dimName));
          } else {
            DT_CHECK(p.second.validGap_.at(dimName)[0].first +
                         p.second.validGap_.at(dimName)[0].second ==
                     p.second.dimToSize_.at(dimName));

            if (p.second.validGap_.at(dimName)[0].second > 0) {
              DT_CHECK(p.second.validGap_.at(dimName) ==
                       lds->validGap_.at(dimName));
            }
          }
        }
      }
    }

    for (auto& dimName : s_op->arrayB->layoutDimOrder_) {
      DT_CHECK(s_op->arrayA->dimToLayoutSize_.count(dimName));
      auto dimToLayoutSize_ =
          PieceInfo::getTotalValidsInVG(s_op->arrayB->validGap_);
      DT_CHECK(s_op->arrayB->dimToLayoutSize_.at(dimName) >=
               dimToLayoutSize_.at(dimName));
      DT_CHECK(s_op->arrayA->dimToLayoutSize_.at(dimName) >=
               dimToLayoutSize_.at(dimName));
    }

    // check each piece of arrayB
    for (const auto& pInfo : s_op->arrayB->pieces_) {
      // valid+Gap == dimSize
      DT_CHECK(pInfo.second.dimToSize_ ==
               PieceInfo::getTotalElementsInVG(pInfo.second.validGap_));

      // product of valids across dim

      int product = 1;
      for (const auto& mapkv :
           PieceInfo::getTotalValidsInVG(pInfo.second.validGap_))
        product *= mapkv.second;
      if (s_op->arrayB->isHbmPinned()) {
        DT_CHECK(product <=
                 sysDef.bytesPerStick / 4);  // valids should be <= than a stick
      } else {
        DT_CHECK(product % (uint64_t)(sysDef.bytesPerStick / 4) == 0);
      }
    }

    DT_CHECK(s_op->arrayB->pieces_.size());
    DT_CHECK(s_op->arrayA->pieces_.size());
    DT_CHECK(s_op->arrayC->pieces_.size());

    // check if B is lxpinned than inner most dim is ScatterDim
    if (s_op->arrayB->isLxPinned()) {
      DT_CHECK(s_op->arrayB->dimToStickSize_.size() == 1);
    }

    // pieces in a core are all same
    DT_CHECK(s_op->arrayC->isHbmPinned());
    for (auto& mapkv : s_op->coreIDtoANInfo) {
      DT_CHECK(!mapkv.second.pieceOrder_arrayC.empty());
      if (s_op->arrayB->isHbmPinned())
        DT_CHECK(!mapkv.second.pieceOrder_arrayB.empty());
      else
        DT_CHECK(mapkv.second.pieceOrder_arrayB.empty());

      if (s_op->arrayA->isHbmPinned())
        DT_CHECK(!mapkv.second.pieceOrder_arrayA.empty());
      else
        DT_CHECK(mapkv.second.pieceOrder_arrayA.empty());
    }

    // for all pieces in arrayB
    for (const auto& p : s_op->arrayB->pieces_) {
      DT_CHECK(s_op->arrayB_stride.count(p.first));
      DT_CHECK(s_op->arrayB_startIdx.count(p.first));
    }
  } else if (myDataOpDsc.op->name == OpFuncs::Nop) {
    // do nothing
  } else if (myDataOpDsc.op->name == OpFuncs::StickifyOpHBM) {
    StickifyOpHBM* myOp = (StickifyOpHBM*)myDataOpDsc.op;
    int tEleInnerDim =
        myOp->validGapInnerDim.first + myOp->validGapInnerDim.second;
    for (int idx = 0; idx < myDataOpDsc.labeledDs_.size(); idx++) {
      // can have only one dim
      auto const& myldsInfo = myDataOpDsc.labeledDs_[idx];
      DT_CHECK(myldsInfo.dimToLayoutSize_.size() == 1);
      DT_CHECK(myldsInfo.dimToStickSize_.size() == 1);
      DT_CHECK(myldsInfo.layoutDimOrder_.size() == 1);
      DT_CHECK(myldsInfo.stickDimOrder_.size() == 1);
    }
    DT_CHECK(myOp->inpLds->wordLength == myOp->outLds->wordLength);
    DT_CHECK(myOp->inpLds->validGap_.size() == 1);
    long totalValidInp =
        PieceInfo::getTotalValidsInVG(myOp->inpLds->validGap_).begin()->second;
    DT_CHECK(totalValidInp % myOp->validGapInnerDim.first == 0);

    DT_CHECK(myOp->inpLds->dimToLayoutSize_.begin()->second - totalValidInp >=
             2 * myOp->inpLds->dimToStickSize_.begin()
                     ->second);  // we need gap of 2 extra sticks

    // input always come from HBM
    int totalPices = myOp->inpLds->pieces_.size();
    DT_CHECK(totalValidInp % totalPices ==
             0); /* can be relaxed for imbalanced work, could be removed once
                graphOptimizer is filling dataDSC correctly for balanced
                work-division*/

    int sticksPerWork = (totalValidInp / totalPices) /
                            (sysDef.bytesPerStick / myOp->inpLds->wordLength) +
                        2;
    for (const auto& p : myOp->inpLds->pieces_) {
      DT_CHECK(p.second.placement.size() == 2);
      DT_CHECK(p.second.placement.count(SenComponents::HBM));
      DT_CHECK(p.second.placement.count(SenComponents::LX));
      DT_CHECK(p.second.dimToSize_.size() == 1);
      DT_CHECK(p.second.dimToSize_.begin()->second ==
               sticksPerWork * (sysDef.bytesPerStick /
                                myOp->inpLds->wordLength));  // can be relaxed
      DT_CHECK(myOp->lxStOffIPiece.count(p.first));
    }
    DT_CHECK((long)myOp->outLds->dimToLayoutSize_.begin()->second %
                 tEleInnerDim ==
             0);
    // DT_CHECK(tEleInnerDim %
    // (long)myOp->outLds->dimToStickSize_.begin()->second
    // ==
    //       0);
    DT_CHECK(myOp->validGapInnerDim.second !=
             0);                             // no gap --> stickifyOp not req.
    DT_CHECK(myOp->reqZeroCanvas == false);  // not supported

    DT_CHECK(myOp->inpLds->pieces_.size() == myOp->outLds->pieces_.size());
    if (myOp->initialGap) {
      long outSize = myOp->outLds->dimToLayoutSize_.begin()->second;
      DT_CHECK(outSize % totalPices == 0);
      int oSticksPerWorker =
          (outSize / totalPices) /
              (sysDef.bytesPerStick / myOp->outLds->wordLength) +
          1;  // need space for 1 extra stick

      for (const auto& p : myOp->outLds->pieces_) {
        DT_CHECK(p.second.placement.size());
        DT_CHECK(p.second.placement.count(SenComponents::LX));
        DT_CHECK(p.second.dimToSize_.size() == 1);
        DT_CHECK(p.second.dimToSize_.begin()->second ==
                 oSticksPerWorker *
                     (sysDef.bytesPerStick / myOp->outLds->wordLength));
      }
    }

    DT_CHECK(myOp->inpLds->df != DataFormats::INVALID);
    DT_CHECK(myOp->outLds->df != DataFormats::INVALID);
    DT_CHECK(myOp->outLds->df == myOp->inpLds->df);
  }

  // ResizeNN checks
  if (myDataOpDsc.op->name == OpFuncs::ResizeNNHBM ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNLX) {
    std::map<std::string, int> upSizeFactor;
    if (myDataOpDsc.op->name == OpFuncs::ResizeNNHBM) {
      ResizeNNHBM* myOpResize = (ResizeNNHBM*)myDataOpDsc.op;
      upSizeFactor = myOpResize->upSizeFactor;
      // check for opConst Info..
      if (myOpResize->genConstIntr) {
        DT_CHECK(myOpResize->pieceNameToOpConsts.size());
        for (auto& pNamekv : myOpResize->pieceNameToOpConsts)
          DT_CHECK(myOpResize->inpLds->pieces_.count(pNamekv.first));
      }
    } else {
      ResizeNNLX* myOpResize = (ResizeNNLX*)myDataOpDsc.op;
      upSizeFactor = myOpResize->upSizeFactor;
      // check for opConst Info..
      if (myOpResize->genConstIntr) {
        DT_CHECK(myOpResize->pieceNameToOpConsts.size());
        for (auto& pNamekv : myOpResize->pieceNameToOpConsts)
          DT_CHECK(myOpResize->inpLds->pieces_.count(pNamekv.first));
      }
    }

    baseSTCDPOp* myOp = (baseSTCDPOp*)myDataOpDsc.op;
    // check upSizeFactor
    for (auto& mapkv : upSizeFactor) {
      DT_CHECK(myOp->inpLds->dimToLayoutSize_.count(mapkv.first));
      DT_CHECK(mapkv.second >= 1);
    }

    DT_CHECK(myOp->inpLds->layoutDimOrder_.size() >= 1);
    DT_CHECK(myOp->outLds->layoutDimOrder_.size() >= 1);

    for (auto& dim1 : myOp->inpLds->layoutDimOrder_) {
      DT_CHECK(upSizeFactor.count(dim1));
      DT_CHECK(myOp->outLds->dimToLayoutSize_.at(dim1) >=
               myOp->inpLds->dimToLayoutSize_.at(dim1) * upSizeFactor.at(dim1));
    }

    DT_CHECK(myOp->inpLds->df != DataFormats::INVALID);
    DT_CHECK(myOp->outLds->df != DataFormats::INVALID);
    DT_CHECK(myOp->outLds->df == myOp->inpLds->df);
  }

  // ConstPadOpLX checks
  if (myDataOpDsc.op->name == OpFuncs::ConstPadOpLX) {
    auto* constOp = (ConstPadOpLX*)myDataOpDsc.op;
    DT_CHECK(constOp->pieceNameToOpConsts.size());
    for (auto& pNamekv : constOp->pieceNameToOpConsts)
      DT_CHECK(constOp->Lds->pieces_.count(pNamekv.first));

    // check each piece
    for (const auto& pInfo : constOp->Lds->pieces_) {
      // valid+Gap == dimSize
      DT_CHECK(pInfo.second.dimToSize_ ==
               PieceInfo::getTotalElementsInVG(pInfo.second.validGap_));
    }
  } else if (myDataOpDsc.op->name == OpFuncs::ConstPadOpHBM) {
    auto* constOp = (ConstPadOpHBM*)myDataOpDsc.op;
    DT_CHECK(constOp->pieceNameToOpConsts.size());
    std::unordered_set<int> coreIDs;
    for (auto& pNamekv : constOp->pieceNameToOpConsts) {
      DT_CHECK(constOp->Lds->pieces_.count(pNamekv.first));

      DT_CHECK(constOp->Lds->pieces_.at(pNamekv.first).dimToSize_ ==
               PieceInfo::getTotalValidsInVG(
                   constOp->Lds->pieces_.at(pNamekv.first).validGap_));

      int num_elements = 1;
      for (const auto& kvsize :
           constOp->Lds->pieces_.at(pNamekv.first).dimToSize_)
        num_elements *= kvsize.second;
      DT_CHECK(num_elements ==
               (sysDef.bytesPerStick / constOp->Lds->wordLength) *
                   sysDef.l3BurstSize);

      DT_CHECK(constOp->Lds->pieces_.at(pNamekv.first)
                   .placement.count(SenComponents::LX));
      DT_CHECK(constOp->Lds->pieces_.at(pNamekv.first)
                   .placement.at(SenComponents::LX)
                   .MemId()
                   .getSingleData()
                   .size());
      for (const auto& memId_Vec : constOp->Lds->pieces_.at(pNamekv.first)
                                       .placement.at(SenComponents::LX)
                                       .MemId()
                                       .getAllData()) {
        for (const auto& memId : memId_Vec) {
          DT_CHECK(coreIDs.count(memId) == 0);
          coreIDs.insert(memId);
        }
      }
    }

    // check each piece
    for (const auto& pInfo : constOp->ldsHBM->pieces_) {
      // valid+Gap == dimSize
      DT_CHECK(pInfo.second.dimToSize_ ==
               PieceInfo::getTotalValidsInVG(pInfo.second.validGap_));

      for (const auto& kv2 : pInfo.second.placement) {
        DT_CHECK(kv2.first == SenComponents::HBM);
        DT_CHECK(kv2.second.MemId().getSingleData().size() == 1);
        DT_CHECK(kv2.second.StartAddr().getSingleData().size() == 1);
        DT_CHECK(coreIDs.count(kv2.second.MemId().getSingleData().back()) ==
                 0);  // need to have const piece in LX
      }
    }
  }

  if (is_any_of(myDataOpDsc.op->name, OpFuncs::ITOF, OpFuncs::ITOFHBM)) {
    auto* itof_op = (baseITOFOp*)myDataOpDsc.op;
    DT_CHECK(itof_op->input_->stickDimOrder_.size() == 2);
    DT_CHECK(itof_op->output_->stickDimOrder_.size() == 1);
    DT_CHECK(itof_op->output_->stickDimOrder_.at(0) ==
             itof_op->input_->stickDimOrder_.at(1));
    auto nout_dim = itof_op->output_->stickDimOrder_.at(0);
    auto nin_dim = itof_op->input_->stickDimOrder_.at(0);

    int nin_instick =
        itof_op->output_->wordLength / itof_op->input_->wordLength;
    DT_CHECK(itof_op->input_->dimToStickSize_.at(nin_dim) == nin_instick);

    // format checks
    DT_CHECK(itof_op->output_->df == DataFormats::SEN169_FP16);
    DT_CHECK(itof_op->scale_->df == DataFormats::SEN169_FP16);
    DT_CHECK(itof_op->input_->df == itof_op->inp_mask_->df);
    DT_CHECK(itof_op->scale_->df == itof_op->scale_mask_->df);
    DT_CHECK(is_any_of(itof_op->input_->df, DataFormats::SENINT4,
                       DataFormats::SENINT8));

    // group size checks at LDS level (we will do piece-wise check later)
    int gout = itof_op->input_->dimToLayoutSize_.at(nout_dim) /
               itof_op->scale_->dimToLayoutSize_.at(nout_dim);

    DT_CHECK(gout >= 8 || gout == 1);
    if (gout >= 32)
      DT_CHECK_MSG(gout % 32 == 0, "should be a perfect multiple of 32");
    else
      DT_CHECK_MSG(32 % gout == 0, "should be a perfect divisor of 32");

    int gin = itof_op->input_->dimToLayoutSize_.at(nin_dim) /
              itof_op->scale_->dimToLayoutSize_.at(nin_dim);

    if (gin >= nin_instick)
      DT_CHECK_MSG(gin % nin_instick == 0,
                   "should be a perfect multiple of WLOut/WLin");
    else
      DT_CHECK_MSG(nin_instick % gin == 0,
                   "should be a perfect divisor of WLOut/WLin");

    DT_CHECK(itof_op->scale_mask_->isLxPinned());
    DT_CHECK(itof_op->inp_mask_->isLxPinned());
    if (myDataOpDsc.op->name == OpFuncs::ITOFHBM) {
      // checks for ITOFHBM
      auto* itof_hbmop = (ITOFOpHBM*)myDataOpDsc.op;

      for (auto& mapkv : itof_hbmop->coreIDtoANInfo) {
        bool is_inp_hbm_pinned = itof_op->input_->isHbmPinned();
        bool is_out_hbm_pinned = itof_op->output_->isHbmPinned();
        bool is_scale_hbm_pinned = itof_op->scale_->isHbmPinned();
        if (!is_inp_hbm_pinned)
          DT_CHECK(itof_hbmop->stride_input_.count(mapkv.first));
        if (!is_out_hbm_pinned)
          DT_CHECK(itof_hbmop->stride_output_.count(mapkv.first));
        if (!is_scale_hbm_pinned) {
          DT_CHECK(itof_hbmop->stride_scale_.count(mapkv.first));
        } else {
          DT_CHECK(itof_hbmop->stride_scale_l3lu_.count(mapkv.first));
        }

        for (auto& dimname : itof_op->input_->layoutDimOrder_) {
          if (is_inp_hbm_pinned) {
            DT_CHECK(mapkv.second.loopCount.count(dimname));
            DT_CHECK(mapkv.second.getAddrInfo(SenComponents::L3LU)
                         ->isPresent(dimname));
          } else {
            DT_CHECK(itof_hbmop->stride_input_.at(mapkv.first).count(dimname));
          }
          if (is_out_hbm_pinned) {
            DT_CHECK(mapkv.second.loopCountL3SU.count(dimname));
            DT_CHECK(mapkv.second.getAddrInfo(SenComponents::L3SU)
                         ->isPresent(dimname));
          } else {
            DT_CHECK(itof_hbmop->stride_output_.at(mapkv.first).count(dimname));
          }
          if (is_scale_hbm_pinned) {
            DT_CHECK(
                itof_hbmop->stride_scale_l3lu_.at(mapkv.first).count(dimname));
          } else {
            DT_CHECK(itof_hbmop->stride_scale_.at(mapkv.first).count(dimname));
          }

          DT_CHECK(mapkv.second.loopCountL3SU.at(dimname) ==
                   mapkv.second.loopCount.at(dimname));
        }
      }
    }
  }

  // APEOp (Anywhere Pad Elimination) checks
  if (myDataOpDsc.op->name == OpFuncs::APEOpLX ||
      myDataOpDsc.op->name == OpFuncs::APEOpHBM ||
      myDataOpDsc.op->name == OpFuncs::ReStickifyOpLx ||
      myDataOpDsc.op->name == OpFuncs::ReStickifyOpHBM) {
    baseStickOp* myOp = (baseStickOp*)myDataOpDsc.op;
    // don't allow permute during this Op
    // DT_CHECK(myOp->inpLds->layoutDimOrder_ == myOp->outLds->layoutDimOrder_);

    int tStickElem = 1;
    int stickValidInp = 1;
    int stickDimElem = 1;
    for (auto& mapkv : myOp->inpLds->validGap_) {
      if (DCGUtils::isValPresent(myOp->inpLds->stickDimOrder_, mapkv.first)) {
        int elem_counter = 0;
        int valid_counter = 0;
        stickDimElem *= myOp->inpLds->dimToLayoutSize_.at(mapkv.first);
        for (auto& validGapPair : mapkv.second) {
          elem_counter += validGapPair.first + validGapPair.second;
          valid_counter += validGapPair.first;
        }
        tStickElem *= elem_counter;
        stickValidInp *= valid_counter;
      }
    }
    DT_CHECK((int64_t)(tStickElem * myOp->inpLds->wordLength) %
                 sysDef.bytesPerStick ==
             0);
    DT_CHECK(tStickElem == stickDimElem);

    tStickElem = 1;
    int stickValidOut = 1;
    stickDimElem = 1;
    for (auto& mapkv : myOp->outLds->validGap_) {
      if (DCGUtils::isValPresent(myOp->outLds->stickDimOrder_, mapkv.first)) {
        int elem_counter = 0;
        int valid_counter = 0;
        stickDimElem *= myOp->outLds->dimToLayoutSize_.at(mapkv.first);
        for (auto& validGapPair : mapkv.second) {
          elem_counter += validGapPair.first + validGapPair.second;
          valid_counter += validGapPair.first;
        }
        tStickElem *= elem_counter;
        stickValidOut *= valid_counter;
      }
    }
    DT_CHECK((int64_t)(tStickElem * myOp->outLds->wordLength) %
                 sysDef.bytesPerStick ==
             0);
    DT_CHECK(tStickElem == stickDimElem);

    if (myDataOpDsc.op->name == OpFuncs::APEOpLX ||
        myDataOpDsc.op->name == OpFuncs::APEOpHBM)
      DT_CHECK(stickValidOut == stickValidInp);

    // we don't support APE for FP8, INT4 formats
    // need to uncomment DT_CHECKs and test FP8, INT4 formats
    DT_CHECK(myOp->outLds->wordLength == 2);
    DT_CHECK(myOp->inpLds->wordLength == 2);

    DT_CHECK(myOp->inpLds->df != DataFormats::INVALID);
    DT_CHECK(myOp->outLds->df != DataFormats::INVALID);
    DT_CHECK(myOp->outLds->df == myOp->inpLds->df);
  }
}

void DcgFE::initializeDataDscOp(DataOpDsc& myDataOpDsc, int c /*=0*/) {
  auto clearBaseStcdpOpState = [](baseSTCDPOp* myOp) {
    // base stcdp
    myOp->inpSP_.clear();
    myOp->outSP_.clear();
    myOp->dtTable_.clear();
    myOp->coreIDtoDtKey_L3SU.clear();
    myOp->coreIDtoDtKey_L3LU.clear();
    myOp->coreIDtoDtKey_LX.clear();
  };

  auto clearBaseStickOpState = [](baseStickOp* myOp) {
    // base stickOp clear
    myOp->stickTransfers.clear();
    myOp->iStickDimToOffset.clear();
    myOp->oStickDimToOffset.clear();
  };

  auto clearBaseGatherOpState = [](GatherOpHBM* myOp) {
    // base stickOp clear
    myOp->coreIDtoGTRInfo_ArrayB.clear();
    myOp->coreIDtoGTRInfo_ArrayC.clear();
    myOp->coreIdToPieceInfo.clear();
  };

  auto clearBaseScatterOpState = [](ScatterOpHBM* myOp) {
    // base stickOp clear
    myOp->coreIDtoGTRInfo_ArrayB.clear();
    myOp->coreIDtoGTRInfo_ArrayA.clear();
    myOp->coreIdToPieceInfo.clear();
  };

  auto clearBaseITOFOpState = [](baseITOFOp* myOp) {
    // base stickOp clear
    // myOp->lx_offset_.clear();
  };

  // clear pcfg
  myDataOpDsc.pcfg_.clear();
  // create another function for this..
  if (myDataOpDsc.opName == OpFuncs::STCDPOpLx) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new STCDPOpLx;
      myDataOpDsc.op->name = OpFuncs::STCDPOpLx;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::STCDPOpLx);
    }
    STCDPOpLx* myOp = (STCDPOpLx*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];

    // clear datadsc states
    myOp->idealStWindowToDtKey.clear();
    myOp->prodConsList.clear();
    myOp->idealStWindowToContention.clear();
    myOp->coreIDtoTrRank.clear();
    myOp->interLeaveDtTableIdx.clear();
    myOp->prodCoreGpList.clear();
    clearBaseStcdpOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::STCDPOpHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new STCDPOpHBM;
      myDataOpDsc.op->name = OpFuncs::STCDPOpHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::STCDPOpHBM);
    }
    STCDPOpHBM* myOp = (STCDPOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    for (auto& coreIDkv : myOp->coreIDtoANInfo)
      coreIDkv.second.loopCountL3SU = coreIDkv.second.loopCount;
    clearBaseStcdpOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::ResizeNNHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ResizeNNHBM;
      myDataOpDsc.op->name = OpFuncs::ResizeNNHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ResizeNNHBM);
    }
    ResizeNNHBM* myOp = (ResizeNNHBM*)myDataOpDsc.op;
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    for (auto& dim1 : myOp->inpLds->layoutDimOrder_) {
      if (!myOp->upSizeFactor.count(dim1)) myOp->upSizeFactor[dim1] = 1;
    }
    for (auto& coreIDkv : myOp->coreIDtoANInfo)
      coreIDkv.second.loopCountL3SU = coreIDkv.second.loopCount;

    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
    clearBaseStcdpOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::ResizeNNLX) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ResizeNNLX;
      myDataOpDsc.op->name = OpFuncs::ResizeNNLX;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ResizeNNLX);
    }
    ResizeNNLX* myOp = (ResizeNNLX*)myDataOpDsc.op;
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    for (auto& dim1 : myOp->inpLds->layoutDimOrder_) {
      if (!myOp->upSizeFactor.count(dim1)) myOp->upSizeFactor[dim1] = 1;
    }
    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
    clearBaseStcdpOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::GatherOpHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new GatherOpHBM;
      myDataOpDsc.op->name = OpFuncs::GatherOpHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::GatherOpHBM);
    }
    GatherOpHBM* myOp = (GatherOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 3);
    myOp->arrayA = &myDataOpDsc.labeledDs_[0];
    myOp->arrayB = &myDataOpDsc.labeledDs_[1];
    myOp->arrayC = &myDataOpDsc.labeledDs_[2];
    clearBaseGatherOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::APEOpLX) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new APEOpLX;
      myDataOpDsc.op->name = OpFuncs::APEOpLX;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::APEOpLX);
    }
    APEOpLX* myOp = (APEOpLX*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
    clearBaseStickOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::APEOpHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new APEOpHBM;
      myDataOpDsc.op->name = OpFuncs::APEOpHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::APEOpHBM);
    }
    APEOpHBM* myOp = (APEOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    for (auto& coreIDkv : myOp->coreIDtoANInfo)
      coreIDkv.second.loopCountL3SU = coreIDkv.second.loopCount;
    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
    clearBaseStickOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::ReStickifyOpLx) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ReStickifyOpLx;
      myDataOpDsc.op->name = OpFuncs::ReStickifyOpLx;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ReStickifyOpLx);
    }
    ReStickifyOpLx* myOp = (ReStickifyOpLx*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
    clearBaseStickOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::ReStickifyOpWithPTLx) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ReStickifyOpWithPTLx;
      myDataOpDsc.op->name = OpFuncs::ReStickifyOpWithPTLx;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx);
    }
    ReStickifyOpWithPTLx* myOp = (ReStickifyOpWithPTLx*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
    clearBaseStickOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::ReStickifyOpHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ReStickifyOpHBM;
      myDataOpDsc.op->name = OpFuncs::ReStickifyOpHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ReStickifyOpHBM);
    }
    ReStickifyOpHBM* myOp = (ReStickifyOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    for (auto& coreIDkv : myOp->coreIDtoANInfo)
      coreIDkv.second.loopCountL3SU = coreIDkv.second.loopCount;
    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
    clearBaseStickOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::ReStickifyOpWithPTHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ReStickifyOpWithPTHBM;
      myDataOpDsc.op->name = OpFuncs::ReStickifyOpWithPTHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM);
    }
    ReStickifyOpWithPTHBM* myOp = (ReStickifyOpWithPTHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    for (auto& coreIDkv : myOp->coreIDtoANInfo)
      coreIDkv.second.loopCountL3SU = coreIDkv.second.loopCount;
    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
    clearBaseStickOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::XRFWriteHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new XRFWriteHBM;
      myDataOpDsc.op->name = OpFuncs::XRFWriteHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::XRFWriteHBM);
    }
    XRFWriteHBM* myOp = (XRFWriteHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    for (auto& coreIDkv : myOp->coreIDtoANInfo)
      coreIDkv.second.loopCountL3SU = coreIDkv.second.loopCount;

    myOp->coreIDToPieceDtGp.clear();
  } else if (myDataOpDsc.opName == OpFuncs::XRFWriteLX) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new XRFWriteLX;
      myDataOpDsc.op->name = OpFuncs::XRFWriteLX;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::XRFWriteLX);
    }
    XRFWriteLX* myOp = (XRFWriteLX*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    myOp->coreIDToPieceDtGp.clear();
  } else if (myDataOpDsc.opName == OpFuncs::Nop) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new baseOp;
      myDataOpDsc.op->name = OpFuncs::Nop;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::Nop);
    }

    if (myDataOpDsc.coreIdsUsed_.size() == 0)
      myDataOpDsc.coreIdsUsed_.push_back(0);
  } else if (myDataOpDsc.opName == OpFuncs::StickifyOpHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new StickifyOpHBM;
      myDataOpDsc.op->name = OpFuncs::StickifyOpHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::StickifyOpHBM);
    }
    StickifyOpHBM* myOp = (StickifyOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->inpLds = &myDataOpDsc.labeledDs_[0];
    myOp->outLds = &myDataOpDsc.labeledDs_[1];
    for (auto& coreIDkv : myOp->coreIDtoANInfo)
      coreIDkv.second.loopCountL3SU = coreIDkv.second.loopCount;
    clearBaseStickOpState(myOp);
    myOp->coreIDtol3luOffsetAdj.clear();
    if (myOp->outLds->wordLength == 2) myOp->useImm16 = true;
  } else if (myDataOpDsc.opName == OpFuncs::ConstPadOpLX) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ConstPadOpLX;
      myDataOpDsc.op->name = OpFuncs::ConstPadOpLX;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ConstPadOpLX);
    }
    ConstPadOpLX* myOp = (ConstPadOpLX*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 1);
    myOp->Lds = &myDataOpDsc.labeledDs_[0];
    if (myOp->Lds->wordLength == 2) myOp->useImm16 = true;
  } else if (myDataOpDsc.opName == OpFuncs::ConstPadOpHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ConstPadOpHBM;
      myDataOpDsc.op->name = OpFuncs::ConstPadOpHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ConstPadOpHBM);
    }
    ConstPadOpHBM* myOp = (ConstPadOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 2);
    myOp->Lds = &myDataOpDsc.labeledDs_[0];
    myOp->ldsHBM = &myDataOpDsc.labeledDs_[1];
    if (myOp->Lds->wordLength == 2) myOp->useImm16 = true;
  } else if (myDataOpDsc.opName == OpFuncs::ScatterOpHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ScatterOpHBM;
      myDataOpDsc.op->name = OpFuncs::ScatterOpHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ScatterOpHBM);
    }
    ScatterOpHBM* myOp = (ScatterOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 3);
    myOp->arrayA = &myDataOpDsc.labeledDs_[0];
    myOp->arrayB = &myDataOpDsc.labeledDs_[1];
    myOp->arrayC = &myDataOpDsc.labeledDs_[2];
    clearBaseScatterOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::ITOF) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new baseITOFOp;
      myDataOpDsc.op->name = OpFuncs::ITOF;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ITOF);
    }
    baseITOFOp* myOp = (baseITOFOp*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 5);
    myOp->input_ = &myDataOpDsc.labeledDs_[0];
    myOp->inp_mask_ = &myDataOpDsc.labeledDs_[2];
    myOp->scale_ = &myDataOpDsc.labeledDs_[1];
    myOp->scale_mask_ = &myDataOpDsc.labeledDs_[3];
    myOp->output_ = &myDataOpDsc.labeledDs_[4];
    clearBaseITOFOpState(myOp);
  } else if (myDataOpDsc.opName == OpFuncs::ITOFHBM) {
    if (myDataOpDsc.op == nullptr) {
      myDataOpDsc.op = new ITOFOpHBM;
      myDataOpDsc.op->name = OpFuncs::ITOFHBM;
    } else {
      DT_CHECK(myDataOpDsc.op->name == OpFuncs::ITOFHBM);
    }
    ITOFOpHBM* myOp = (ITOFOpHBM*)myDataOpDsc.op;
    DT_CHECK(myDataOpDsc.labeledDs_.size() == 5);
    myOp->input_ = &myDataOpDsc.labeledDs_[0];
    myOp->inp_mask_ = &myDataOpDsc.labeledDs_[2];
    myOp->scale_ = &myDataOpDsc.labeledDs_[1];
    myOp->scale_mask_ = &myDataOpDsc.labeledDs_[3];
    myOp->output_ = &myDataOpDsc.labeledDs_[4];
    clearBaseITOFOpState(myOp);
  } else {
    DT_CHECK(0);
  }

  myDataOpDsc.op->uniqueID += c;

  for (auto& lds : myDataOpDsc.labeledDs_) {
    if (lds.isHbmPinned() && lds.pieces_.size()) {
      for (auto addr : lds.HbmStartAddress().getAllData()) DT_CHECK(addr != -1);
      if (lds.hbmSize_ == -1)
        lds.hbmSize_ = DCGUtils::getProduct(lds.dimToLayoutSize_) /
                       DCGUtils::getProduct(lds.dimToStickSize_);
    } else {
      lds.hbmSize_ = 0;
      if (lds.HbmStartAddress().hasZeroFoldDim()) {
        lds.HbmStartAddress().insertData(0);
      } else {
        for (auto fcoord : lds.HbmStartAddress().getFlattenedCoordinates())
          lds.HbmStartAddress().insertData(0, fcoord);
      }
    }
  }
}
