/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2018, 2025
 ************************************************************/

/*
 * Description:
 *
 */

#include <dcg/dcg_fe/dcg_frontend.h>

#include "util/dt_exception.hpp"

void DcgFE::createSubPieces(STCDPOpHBM* op) {
  // each piece is a subPiece
  bool inpPinnedHBM = op->inpLds->isHbmPinned();
  bool outPinnedHBM = op->outLds->isHbmPinned();

  std::set<std::string> inpPieceInserted;
  std::set<std::string> outPieceInserted;

  for (auto& anModeInfokv : op->coreIDtoANInfo) {
    int coreID = anModeInfokv.first;
    auto& anModeInfo = anModeInfokv.second;

    bool inp_piece_reuse = false;
    bool out_piece_reuse = false;
    int loop_count = anModeInfo.inpPieceOrder.size();
    if (anModeInfo.inpPieceOrder.size() != anModeInfo.outPieceOrder.size()) {
      DT_CHECK(anModeInfo.isAnalyticalMode);
      if (anModeInfo.inpPieceOrder.size() < anModeInfo.outPieceOrder.size()) {
        inp_piece_reuse = true;
        DT_CHECK(anModeInfo.inpPieceOrder.size() == 1);
        loop_count = anModeInfo.outPieceOrder.size();
      } else {
        out_piece_reuse = true;
        DT_CHECK(anModeInfo.outPieceOrder.size() == 1);
      }
    }

    for (int idx = 0; idx < loop_count; idx++) {
      std::string iPieceName =
          anModeInfo.inpPieceOrder.at(inp_piece_reuse ? 0 : idx);
      std::string oPieceName =
          anModeInfo.outPieceOrder.at(out_piece_reuse ? 0 : idx);

      if (!inp_piece_reuse || (inp_piece_reuse && idx == 0)) {
        if (inpPieceInserted.count(iPieceName)) {
          if (op->outLds->pieces_.size()) {
            DT_CHECK(outPieceInserted.count(oPieceName));
          }
          continue;  // do nothing
        }
      }

      if (!out_piece_reuse || (out_piece_reuse && idx == 0)) {
        if (outPieceInserted.count(oPieceName)) {
          if (op->inpLds->pieces_.size()) {
            DT_CHECK(inpPieceInserted.count(iPieceName));
          }
          continue;  // do nothing
        }
      }

      inpPieceInserted.insert(iPieceName);
      outPieceInserted.insert(oPieceName);

      // form subPiece
      SliceInfo inpSubPiece;
      SliceInfo outSubPiece;

      auto fillSubPiece = [&](SliceInfo& new_sp, const PieceInfo& ref_piece) {
        new_sp.dimToStartCordinate = ref_piece.dimToStartCordinate;

        if (anModeInfo.isAnalyticalMode == false) {
          // check for gaps
          new_sp.dimToSize_ =
              PieceInfo::getTotalValidsInVG(ref_piece.validGap_);
          if (ref_piece.dimToSize_ != new_sp.dimToSize_) {
            // we allow only one gap per dim
            for (auto mapkv1 : ref_piece.validGap_) {
              DT_CHECK(mapkv1.second.size() <= 1);
            }
          }
        } else {
          new_sp.dimToSize_ = ref_piece.dimToSize_;
        }
        new_sp.bigDimToSize_ = ref_piece.dimToSize_;
        DT_CHECK(new_sp.bigDimToSize_ >= new_sp.dimToSize_);
      };

      if (op->inpLds->pieces_.size()) {
        fillSubPiece(inpSubPiece, op->inpLds->pieces_.at(iPieceName));
      }

      if (op->outLds->pieces_.size()) {
        fillSubPiece(outSubPiece, op->outLds->pieces_.at(oPieceName));
      }

      // take care of LX_to_LX transfers
      if (op->inpLds->pieces_.size() && op->outLds->pieces_.size()) {
        // add entry to the dtTable_ : LX-to-LX
        auto& iPlacement =
            op->inpLds->pieces_.at(iPieceName).placement.at(SenComponents::LX);
        auto& oPlacement =
            op->outLds->pieces_.at(oPieceName).placement.at(SenComponents::LX);

        DT_CHECK(iPlacement.MemId().hasZeroFoldDim());
        DT_CHECK(oPlacement.MemId().hasZeroFoldDim());
        auto i_memId_v = iPlacement.MemId().getData();
        auto o_memId_v = oPlacement.MemId().getData();
        DT_CHECK(i_memId_v.size() == o_memId_v.size());

        for (int memIdx = 0; memIdx < i_memId_v.size(); memIdx++) {
          // insert one entry for each memId
          inpSubPiece.placement.clearMemId();
          inpSubPiece.placement.clearStAddr();
          outSubPiece.placement.clearMemId();
          outSubPiece.placement.clearStAddr();

          inpSubPiece.placement.setType(SenComponents::LX);
          outSubPiece.placement.setType(SenComponents::LX);

          fillMemIdAddr(inpSubPiece.placement.MemId(), iPlacement.MemId(),
                        memIdx);
          fillMemIdAddr(inpSubPiece.placement.StartAddr(),
                        iPlacement.StartAddr(), memIdx);
          fillMemIdAddr(outSubPiece.placement.MemId(), oPlacement.MemId(),
                        memIdx);
          fillMemIdAddr(outSubPiece.placement.StartAddr(),
                        oPlacement.StartAddr(), memIdx);

          op->inpSP_.push_back(inpSubPiece);
          int inpSubPieceIdx = op->inpSP_.size() - 1;
          DT_CHECK(op->dtTable_.count(inpSubPieceIdx) == 0);
          op->dtTable_[inpSubPieceIdx];
          op->dtTable_.at(inpSubPieceIdx).pIDX = op->inpSP_.size() - 1;
          op->dtTable_.at(inpSubPieceIdx).pMemID = i_memId_v.at(memIdx);
          op->outSP_.push_back(outSubPiece);
          op->dtTable_.at(inpSubPieceIdx)
              .cIDXs.push_back(op->outSP_.size() - 1);

          if (inp_piece_reuse) {
            // input is bigger (N-Buffer)
            op->inpSP_.back().dimToSize_ = op->outSP_.back().dimToSize_;
          } else if (out_piece_reuse) {
            // output is bigger (N-Buffer)
            op->outSP_.back().dimToSize_ = op->inpSP_.back().dimToSize_;
          }
        }
      }

      if (inpPinnedHBM && op->inpLds->pieces_.size()) {
        // create another entry for HBM to LX
        SliceInfo inpSubPieceHBM = inpSubPiece;
        inpSubPieceHBM.placement =
            op->inpLds->pieces_.at(iPieceName).placement.at(SenComponents::HBM);

        DT_CHECK(inpSubPieceHBM.placement.MemId().hasZeroFoldDim());
        auto memId_inp_sp = inpSubPieceHBM.placement.MemId().getData();

        DT_CHECK(memId_inp_sp.size() == 1);
        inpSubPieceHBM.bigDimToSize_ = op->inpLds->dimToLayoutSize_;

        // add entry to the dtTable_
        op->inpSP_.push_back(inpSubPieceHBM);
        int inpSubPieceIdxHBM = op->inpSP_.size() - 1;
        DT_CHECK(op->dtTable_.count(inpSubPieceIdxHBM) == 0);
        op->dtTable_[inpSubPieceIdxHBM];
        op->dtTable_.at(inpSubPieceIdxHBM).pIDX = op->inpSP_.size() - 1;
        op->dtTable_.at(inpSubPieceIdxHBM).pMemID = memId_inp_sp.at(0);

        auto& iPlacement =
            op->inpLds->pieces_.at(iPieceName).placement.at(SenComponents::LX);

        DT_CHECK(iPlacement.MemId().hasZeroFoldDim());
        auto c_memId = iPlacement.MemId().getData();

        if (c_memId.size() > 1) op->reqMulticast = true;

        for (int memIdx = 0; memIdx < c_memId.size(); memIdx++) {
          PlacementInfo newPlacement;
          newPlacement.setType(SenComponents::LX);

          newPlacement.MemId().insertData({c_memId.at(memIdx)});
          fillMemIdAddr(newPlacement.StartAddr(), iPlacement.StartAddr(),
                        memIdx);

          inpSubPiece.placement = newPlacement;
          op->outSP_.push_back(inpSubPiece);
          op->dtTable_.at(inpSubPieceIdxHBM)
              .cIDXs.push_back(op->outSP_.size() - 1);
        }
      }

      if (op->outLds->pieces_.size())
        outSubPiece.placement =
            op->outLds->pieces_.at(oPieceName).placement.at(SenComponents::LX);

      if (outPinnedHBM && op->outLds->pieces_.size()) {
        // create another entry for LX to HBM
        SliceInfo outSubPieceHBM = outSubPiece;
        outSubPieceHBM.placement =
            op->outLds->pieces_.at(oPieceName).placement.at(SenComponents::HBM);

        DT_CHECK(outSubPiece.placement.MemId().hasZeroFoldDim());
        auto memId_hbm_v = outSubPiece.placement.MemId().getData();
        DT_CHECK(memId_hbm_v.size() == 1);  // we don't support > 1
        outSubPieceHBM.bigDimToSize_ = op->outLds->dimToLayoutSize_;

        // add entry to the dtTable_
        op->inpSP_.push_back(outSubPiece);
        int outSubPieceIdxHBM = op->inpSP_.size() - 1;
        DT_CHECK(op->dtTable_.count(outSubPieceIdxHBM) == 0);
        op->dtTable_[outSubPieceIdxHBM];
        op->dtTable_.at(outSubPieceIdxHBM).pIDX = op->inpSP_.size() - 1;
        op->dtTable_.at(outSubPieceIdxHBM).pMemID = memId_hbm_v.at(0);
        op->outSP_.push_back(outSubPieceHBM);
        op->dtTable_.at(outSubPieceIdxHBM)
            .cIDXs.push_back(op->outSP_.size() - 1);
      }
    }
  }
}

void DcgFE::createSubPieces(STCDPOpLx* op) {
  // each output LDs piece
  for (auto& myOutPiece : op->outLds->pieces_) {
    // check if it subset of an input pieces
    for (auto& myInpPiece : op->inpLds->pieces_) {
      bool hasOverlap =
          doesPiecesOverlap(op, &myInpPiece.second, &myOutPiece.second);
      if (hasOverlap) {
        // insert subPieces, if the exact match is not present
        insertSubPieces(op, myInpPiece.second, myOutPiece.second);
      }
    }
  }

  // Focused diagnostic for Granite P06: dump the exact byte-addressed
  // common-refinement transfer table after logical piece intersections have
  // been lowered to source and destination subpieces.
  const auto& dims = op->inpLds->dimToLayoutSize_;
  const bool isGraniteP06 =
      dims.count("x") && dims.count("y") && dims.count("mb") &&
      dims.count("in") && dims.at("x") == 8 && dims.at("y") == 512 &&
      dims.at("mb") == 4 && dims.at("in") == 128;
  if (isGraniteP06) {
    auto dumpSlice = [](const char* role, int idx, const SliceInfo& slice) {
      std::cout << "P06_STCDP_SLICE role=" << role << " idx=" << idx
                << " mem=";
      for (const auto memId : slice.placement.MemId().getData()) {
        std::cout << memId << ",";
      }
      std::cout << " addr=";
      for (const auto& addresses : slice.placement.StartAddr().getAllData()) {
        for (const auto address : addresses) {
          std::cout << address << ",";
        }
      }
      std::cout << " start=";
      for (const auto& [dim, start] : slice.dimToStartCordinate) {
        std::cout << dim << ":" << start << ",";
      }
      std::cout << " size=";
      for (const auto& [dim, size] : slice.dimToSize_) {
        std::cout << dim << ":" << size << ",";
      }
      std::cout << std::endl;
    };

    std::cout << "P06_STCDP_BEGIN input_ldo=";
    for (const auto& dim : op->inpLds->layoutDimOrder_) {
      std::cout << dim << ",";
    }
    std::cout << " output_ldo=";
    for (const auto& dim : op->outLds->layoutDimOrder_) {
      std::cout << dim << ",";
    }
    std::cout << " entries=" << op->dtTable_.size() << std::endl;
    for (const auto& [key, transfer] : op->dtTable_) {
      std::cout << "P06_STCDP_ENTRY key=" << key
                << " producer=" << transfer.pIDX << " consumers=";
      for (const auto consumer : transfer.cIDXs) {
        std::cout << consumer << ",";
      }
      std::cout << std::endl;
      dumpSlice("src", transfer.pIDX, op->inpSP_.at(transfer.pIDX));
      for (const auto consumer : transfer.cIDXs) {
        dumpSlice("dst", consumer, op->outSP_.at(consumer));
      }
    }
    std::cout << "P06_STCDP_END" << std::endl;
  }
}

void DcgFE::mapDtEntryToSenComponent(STCDPOpLx* op) {
  op->coreIDtoDtKey_L3LU.clear();
  op->coreIDtoDtKey_L3SU.clear();
  op->coreIDtoDtKey_LX.clear();
  for (auto& dtTableEntry : op->dtTable_) {
    int pSpIdx = dtTableEntry.first;

    DT_CHECK(op->inpSP_.at(pSpIdx).placement.MemId().hasZeroFoldDim());
    auto memId_v = op->inpSP_.at(pSpIdx).placement.MemId().getData();
    DT_CHECK(memId_v.size() == 1);

    int pMemID = memId_v.at(0);
    SenComponents pMemType = op->inpSP_[pSpIdx].placement.Type();

    // if consumer with
    bool extConsumerPresent = false;
    for (auto& cSpIdx : dtTableEntry.second.cIDXs) {
      auto cMemIdVec = op->outSP_.at(cSpIdx).placement.MemId().getData();
      SenComponents cMemType = op->outSP_.at(cSpIdx).placement.Type();
      DT_CHECK(cMemType != SenComponents::HBM);
      for (int idx = 0; idx < cMemIdVec.size(); idx++) {
        int cMemID = cMemIdVec[idx];
        if (pMemID != cMemID &&
            !DCGUtils::isDTtableKeyPresent(op->coreIDtoDtKey_L3LU[cMemID],
                                           dtTableEntry.first)) {
          if (!isInpFetchNeigh_)
            insertL3LUSortedpMemID(op, cMemID, dtTableEntry.first);
          else
            insertL3LUSortedCoordinateInpFetch(op, cMemID, dtTableEntry.first);
          extConsumerPresent = true;
        } else if (pMemID == cMemID &&
                   !DCGUtils::isDTtableKeyPresent(op->coreIDtoDtKey_LX[cMemID],
                                                  dtTableEntry.first)) {
          op->coreIDtoDtKey_LX[cMemID].push_back(dtTableEntry.first);
        } else {
          // do nothing
        }
      }
    }
    DT_CHECK(pMemType != SenComponents::HBM);
    if (!DCGUtils::isDTtableKeyPresent(op->coreIDtoDtKey_L3SU[pMemID],
                                       dtTableEntry.first) &&
        extConsumerPresent) {
      if (!isInpFetchNeigh_)
        insertL3SUSortedcMemID(op, pMemID, dtTableEntry.first);
      else
        insertL3SUSortedCoordinateInpFetch(op, pMemID, dtTableEntry.first);
    }
  }
}

void DcgFE::mapDtEntryToSenComponent(STCDPOpHBM* op) {
  for (auto& dtTableEntry : op->dtTable_) {
    int pSpIdx = dtTableEntry.first;
    DT_CHECK(op->inpSP_.at(pSpIdx).placement.MemId().hasZeroFoldDim());
    auto memId_v = op->inpSP_.at(pSpIdx).placement.MemId().getData();
    DT_CHECK(memId_v.size() == 1);

    int pMemID = memId_v.at(0);
    SenComponents pMemType = op->inpSP_[pSpIdx].placement.Type();

    // consumer info
    // DT_CHECK(dtTableEntry.second.cIDXs.size() == 1);
    for (int i = 0; i < dtTableEntry.second.cIDXs.size(); i++) {
      int cSpIdx = dtTableEntry.second.cIDXs[i];
      auto c_memid_v = op->outSP_.at(cSpIdx).placement.MemId().getData();
      DT_CHECK(c_memid_v.size() == 1);
      int cMemID = c_memid_v.at(0);
      SenComponents cMemType = op->outSP_[cSpIdx].placement.Type();

      if (pMemType == SenComponents::HBM) {
        DT_CHECK(cMemType == SenComponents::LX);
        // insert L3LU
        if (!DCGUtils::isDTtableKeyPresent(op->coreIDtoDtKey_L3LU[cMemID],
                                           dtTableEntry.first)) {
          op->coreIDtoDtKey_L3LU[cMemID].push_back(dtTableEntry.first);
        }
      } else if (pMemType == SenComponents::LX) {
        if (cMemType == SenComponents::HBM) {
          // insert L3SU
          if (!DCGUtils::isDTtableKeyPresent(op->coreIDtoDtKey_L3SU[pMemID],
                                             dtTableEntry.first)) {
            op->coreIDtoDtKey_L3SU[pMemID].push_back(dtTableEntry.first);
          }
        } else if (cMemType == SenComponents::LX) {
          if (!DCGUtils::isDTtableKeyPresent(op->coreIDtoDtKey_LX[pMemID],
                                             dtTableEntry.first)) {
            op->coreIDtoDtKey_LX[pMemID].push_back(dtTableEntry.first);
          }
        } else {
          DT_CHECK(0);
        }

      } else {
        DT_CHECK(0);
      }
    }
  }
}

void DcgFE::determineInnerLoopOrder(baseSTCDPOp* op) {
  for (auto& dtTableEntry : op->dtTable_) {
    // loop order is same as producer layout order
    if (op->outSP_[dtTableEntry.second.cIDXs[0]].placement.Type() ==
        SenComponents::HBM) {
      DT_CHECK(op->inpSP_[dtTableEntry.second.pIDX].placement.Type() ==
               SenComponents::LX);
      dtTableEntry.second.loopOrder = op->outLds->layoutDimOrder_;
    } else {
      dtTableEntry.second.loopOrder = op->inpLds->layoutDimOrder_;
      if (op->inpSP_[dtTableEntry.second.pIDX].placement.Type() !=
          SenComponents::HBM) {
        // check if the burst can be used
        if (op->outLds->layoutDimOrder_[0] != op->inpLds->layoutDimOrder_[0]) {
          dtTableEntry.second.useBurst = false;
        }
      }
    }
  }
}

void DcgFE::finalizeBurstInfo(STCDPOpLx* op) {
  // L3SU
  for (auto& mapkv : op->coreIDtoDtKey_L3SU) {
    if (mapkv.second.size() == 0) {
      continue;
    }
    std::string dimName = op->dtTable_.at(mapkv.second[0]).loopOrder[0];
    int dimSize = op->inpSP_[mapkv.second[0]].dimToSize_[dimName];
    bool disableBurst = false;
    for (auto& inpSPIdx : mapkv.second) {
      if (op->inpSP_[inpSPIdx].dimToSize_[dimName] != dimSize) {
        disableBurst = true;
      }
    }
    for (auto& inpSPIdx : mapkv.second) {
      if (disableBurst) op->dtTable_.at(inpSPIdx).useBurst = false;
    }
  }

  // L3LU
  for (auto& mapkv : op->coreIDtoDtKey_L3LU) {
    int coreID = mapkv.first;
    if (mapkv.second.size() == 0) {
      continue;
    }
    std::string dimName = op->dtTable_.at(mapkv.second[0]).loopOrder[0];
    int firstIdx = getIdxForMatchingCMenID(
        op->outSP_, op->dtTable_.at(mapkv.second[0]).cIDXs, coreID);
    int dimSize = op->outSP_[firstIdx].dimToSize_[dimName];
    bool disableBurst = false;
    for (auto& inpSPIdx : mapkv.second) {
      int outSPIdx = getIdxForMatchingCMenID(
          op->outSP_, op->dtTable_.at(inpSPIdx).cIDXs, coreID);
      if (op->outSP_[outSPIdx].dimToSize_[dimName] != dimSize) {
        disableBurst = true;
      }
    }

    for (auto& inpSPIdx : mapkv.second) {
      if (disableBurst) op->dtTable_.at(inpSPIdx).useBurst = false;
    }
  }
}

void DcgFE::checkSubPieceCoverage(STCDPOpLx* op) {
  for (auto& outPiecekv : op->outLds->pieces_) {
    auto& outPiece = outPiecekv.second;
    for (auto& memId :
         outPiece.placement.at(SenComponents::LX).MemId().getData()) {
      for (auto& map1 : outPiece.dimToStartCordinate) {
        long startOP = map1.second;
        if (outPiece.validGap_.at(map1.first).size() == 2)
          startOP += outPiece.validGap_.at(map1.first)[0].second;
        long endOP = map1.second + outPiece.dimToSize_.at(map1.first);
        if (outPiece.validGap_.at(map1.first).size() == 2) {
          endOP -= outPiece.validGap_.at(map1.first)[1].second;
        } else {
          endOP -= outPiece.validGap_.at(map1.first)[0].second;
        }
        std::map<long, bool> isCovered;
        for (long start = startOP; start < endOP; start++) {
          isCovered[start] = false;
        }

        for (auto& mySubPiece : op->outSP_) {
          auto memid_v = mySubPiece.placement.MemId().getData();
          DT_CHECK(memid_v.size() == 1);
          if (memid_v.at(0) == memId) {
            long endSP = mySubPiece.dimToStartCordinate.at(map1.first) +
                         mySubPiece.dimToSize_.at(map1.first);
            for (long startSP = mySubPiece.dimToStartCordinate.at(map1.first);
                 startSP < endSP; startSP++) {
              if (isCovered.count(startSP)) isCovered.at(startSP) = true;
            }
          }
        }
        for (long start = startOP; start < endOP; start++) {
          if (isCovered.at(start) == false) DT_CHECK(0);  // not covered
        }
      }
    }
  }
}

bool DcgFE::isSRQProne(baseSTCDPOp* op) {
  if (op->name == OpFuncs::STCDPOpLx &&
      sysDef.coreArch <= IsaCoreGen::RCUDD1A_ISA && dscGlobal->dtVersion >= 2) {
    auto stcdp_op = static_cast<STCDPOpLx*>(op);
    // check is the program is prone to SRQ-HB-BUG
    if (stcdp_op->is_SRQ_HW_Bug_prone_) {
      return true;
    }
  }
  return false;
}

void DcgFE::createPcfgsSTCDPOp(DataOpDsc* myDataDscPtr,
                               bool forceNoOpt /*= false*/) {
  myDataDscPtr->pcfg_.clear();
  myDataDscPtr->pcfg_.resize(myDataDscPtr->coreIdsUsed_.size());
  baseSTCDPOp* op = (baseSTCDPOp*)myDataDscPtr->op;

  // figure out #cores involved
  for (int idx = 0; idx < myDataDscPtr->coreIdsUsed_.size(); idx++) {
    int coreID = myDataDscPtr->coreIdsUsed_[idx];
    if (verbose_ > 0) {
      std::cout << "Creating pcfg for coreID:" << coreID;
    }

    // L3-SU
    if (op->coreIDtoDtKey_L3SU[coreID].size()) {
      DT_CHECK(op->name != OpFuncs::ResizeNNLX);
      if (verbose_ > 0) {
        std::cout << " : L3SU";
      }
      std::pair<SenPcfg, SenComponents> newPair;
      newPair.second = SenComponents::L3SU;
      newPair.first.dtFormat =
          DCGUtils::getDsFormat(op->outLds->df, op->outLds->wordLength);
      if (op->name == OpFuncs::STCDPOpLx) {
        // if (op->optSTCDP && isOptSTCDPFeasible)
        if (op->optSTCDP && !forceNoOpt)
          transformToPcfgSTCDPLxUnrolled(newPair.first, op, coreID,
                                         SenComponents::L3SU);
        else
          transformToPcfg(newPair.first, op, coreID, SenComponents::L3SU);
      } else {
        bool reqSync = op->coreIDtoDtKey_LX[coreID].size();  // LX comes in pair
        transformToPcfg(newPair.first, (STCDPOpHBM*)op, coreID,
                        SenComponents::L3SU, reqSync);
      }
      myDataDscPtr->pcfg_[idx].push_back(newPair);
    }

    // L3-LU
    if (op->coreIDtoDtKey_L3LU[coreID].size()) {
      if (verbose_ > 0) {
        std::cout << " : L3LU";
      }
      std::pair<SenPcfg, SenComponents> newPair;
      newPair.second = SenComponents::L3LU;
      newPair.first.dtFormat =
          DCGUtils::getDsFormat(op->inpLds->df, op->inpLds->wordLength);
      if (op->name == OpFuncs::STCDPOpLx) {
        // if (op->optSTCDP && isOptSTCDPFeasible)
        if (op->optSTCDP && !forceNoOpt)
          transformToPcfgSTCDPLxUnrolled(newPair.first, op, coreID,
                                         SenComponents::L3LU);
        else
          transformToPcfg(newPair.first, op, coreID, SenComponents::L3LU);
      } else {
        bool reqSync = op->coreIDtoDtKey_LX[coreID].size();  // LX comes in pair
        transformToPcfg(newPair.first, (STCDPOpHBM*)op, coreID,
                        SenComponents::L3LU, reqSync);
      }
      myDataDscPtr->pcfg_[idx].push_back(newPair);
    } else if (isInpFetchNeigh_ && myIFNInfo_.coreIdtoOrderedSP.count(coreID)) {
      if (myIFNInfo_.coreIdtoOrderedSP.at(coreID).size()) {
        if (verbose_ > 0) {
          std::cout << " : L3LU";
        }
        std::pair<SenPcfg, SenComponents> newPair;
        newPair.second = SenComponents::L3LU;
        newPair.first.dtFormat =
            DCGUtils::getDsFormat(op->inpLds->df, op->inpLds->wordLength);
        transformToPcfgSyncInpNeigborFetch(newPair.first, op, coreID,
                                           SenComponents::L3LU);
        myDataDscPtr->pcfg_[idx].push_back(newPair);
      }

    } else if (op->name == OpFuncs::ResizeNNLX ||
               op->name == OpFuncs::ResizeNNHBM) {
      const auto genConstIntr = op->name == OpFuncs::ResizeNNLX
                                    ? ((ResizeNNLX*)op)->genConstIntr
                                    : ((ResizeNNHBM*)op)->genConstIntr;
      if (genConstIntr) {
        if (verbose_ > 0) {
          std::cout << " : L3LU";
        }
        std::pair<SenPcfg, SenComponents> newPair;
        newPair.first.dtFormat =
            DCGUtils::getDsFormat(op->inpLds->df, op->inpLds->wordLength);
        newPair.second = SenComponents::L3LU;
        if (op->name == OpFuncs::ResizeNNLX) {
          transformToPcfgConstInstr(newPair.first, (ResizeNNLX*)op, coreID,
                                    SenComponents::L3LU);
        } else {
          transformToPcfgConstInstr(newPair.first, (ResizeNNHBM*)op, coreID,
                                    SenComponents::L3LU);
        }
        myDataDscPtr->pcfg_[idx].push_back(newPair);
      }
    }

    // LX
    if (op->coreIDtoDtKey_LX[coreID].size()) {
      if (verbose_ > 0) {
        std::cout << " : LX";
      }
      std::pair<SenPcfg, SenComponents> newPairLU;
      std::pair<SenPcfg, SenComponents> newPairSU;
      newPairLU.first.dtFormat =
          DCGUtils::getDsFormat(op->inpLds->df, op->inpLds->wordLength);
      newPairSU.first.dtFormat =
          DCGUtils::getDsFormat(op->outLds->df, op->outLds->wordLength);
      newPairLU.second = SenComponents::LXLU0;
      newPairSU.second = SenComponents::LXSU0;
      if (op->name == OpFuncs::STCDPOpLx) {
        transformToPcfg(newPairLU.first, op, coreID, SenComponents::LXLU0);
        transformToPcfg(newPairSU.first, op, coreID, SenComponents::LXSU0);
      } else {
        // check if we have N-Buffer requirement
        bool req_NBuffer_LXSU = false;  // LXSU is single-ended
        bool req_NBuffer_LXLU = false;  // LXLU is single-ended
        if (is_any_of(op->name, OpFuncs::ResizeNNHBM, OpFuncs::STCDPOpHBM)) {
          auto opHBM = static_cast<STCDPOpHBM*>(op);
          if (opHBM->coreIDtoANInfo.count(coreID)) {
            if (opHBM->coreIDtoANInfo.at(coreID).isAnalyticalMode &&
                opHBM->coreIDtoANInfo.at(coreID).inpPieceOrder.size() !=
                    opHBM->coreIDtoANInfo.at(coreID).outPieceOrder.size()) {
              if (opHBM->coreIDtoANInfo.at(coreID).inpPieceOrder.size() >
                  opHBM->coreIDtoANInfo.at(coreID).outPieceOrder.size()) {
                req_NBuffer_LXSU = true;
                DT_CHECK(
                    opHBM->coreIDtoANInfo.at(coreID).outPieceOrder.size() == 1);
              } else {
                req_NBuffer_LXLU = true;
                DT_CHECK(
                    opHBM->coreIDtoANInfo.at(coreID).inpPieceOrder.size() == 1);
              }
            }
          }
        }

        if (req_NBuffer_LXLU) {
          transformToPcfgNBufferLXUnit(newPairLU.first,
                                       static_cast<STCDPOpHBM*>(op), coreID,
                                       SenComponents::LXLU0);
        } else {
          transformToPcfg(newPairLU.first, op, coreID, SenComponents::LXLU0,
                          op->inpLds->isHbmPinned());
        }

        if (req_NBuffer_LXSU) {
          transformToPcfgNBufferLXUnit(newPairSU.first,
                                       static_cast<STCDPOpHBM*>(op), coreID,
                                       SenComponents::LXSU0);
        } else {
          transformToPcfg(newPairSU.first, op, coreID, SenComponents::LXSU0,
                          op->outLds->isHbmPinned());
        }
      }
      myDataDscPtr->pcfg_[idx].push_back(newPairLU);
      myDataDscPtr->pcfg_[idx].push_back(newPairSU);

      // generate SFP pcfgs
      if (op->useLXSFPLXTransfers) {
        if (verbose_ > 0) {
          std::cout << " : PE0";
        }
        std::pair<SenPcfg, SenComponents> newPairPE;
        newPairPE.first.dtFormat =
            DCGUtils::getDsFormat(op->inpLds->df, op->inpLds->wordLength);
        newPairPE.second = SenComponents::PE0;
        transformToPcfgSfp(newPairPE.first, op, coreID, SenComponents::PE0);
        myDataDscPtr->pcfg_[idx].push_back(newPairPE);
      }
    }
    if (verbose_ > 0) {
      std::cout << " ...\n";
    }
  }
}

template <typename DCGOptype>
void DcgFE::transformToPcfgConstInstr(SenPcfg& newPcfg, DCGOptype* op,
                                      int coreID, SenComponents pcfgType) {
  DT_CHECK(pcfgType == SenComponents::L3LU);
  SenPcfgNode* lastTopNodePcfg = nullptr;
  int count = 0;
  for (auto pNameKv : op->pieceNameToOpConsts) {
    std::string pName = pNameKv.first;
    auto& opConsts = pNameKv.second;
    DT_CHECK(op->inpLds->pieces_.count(pName));
    auto& myPiece = op->inpLds->pieces_.at(pName);
    DT_CHECK(myPiece.placement.count(SenComponents::LX));
    DT_CHECK(myPiece.placement.size() == 1);  // only LX allowed

    auto memId_v = myPiece.placement.at(SenComponents::LX).MemId().getData();
    DT_CHECK(memId_v.size() == 1);
    if (memId_v.at(0) != coreID) continue;
    FoldManager<int64_t> startAddr;
    fillAddr(startAddr, myPiece.placement.at(SenComponents::LX).StartAddr(), 0);
    int product = 1;
    for (auto& val : myPiece.dimToSize_) product *= val.second;

    DT_CHECK(product * op->inpLds->wordLength ==
             128);  // size should be equal to one stick

    // insert mempadconst
    SenPcfgMemPadConstNode* newPadNode =
        (SenPcfgMemPadConstNode*)newPcfg.createPcfgNode(
            SenPcfgNode::Type::MEMPADCONST);
    newPadNode->name = "c" + std::to_string(coreID) + "-" +
                       newPcfg.senComponentsToString.at(pcfgType) +
                       "-ConstPad-" + std::to_string(count) + "-" +
                       std::to_string(op->uniqueID);

    newPadNode->dst = SenComponents::LX;
    newPadNode->stAddr = checkAndGetScalar(startAddr);
    newPadNode->wordLength = op->inpLds->wordLength;
    newPadNode->stickSize = product;
    for (auto val : myPiece.dimToSize_) newPadNode->memPaddedSize.push_back(1);

    newPadNode->coreletId = -1;
    newPadNode->padDimId = 0;
    newPadNode->padDimUnpaddedSize = 0;
    newPadNode->aheadPadSize = 0;
    newPadNode->afterPadSize = 1;

    if (op->useImm16) {
      newPadNode->padValBin = opConsts.at(127) + pow(2, 8) * opConsts.at(126);
      newPadNode->padVal = binToDouble(newPadNode->padValBin, op->inpLds->df);
    } else {
      newPadNode->padValBin = opConsts.at(127);
      newPadNode->padVal = binToDouble(newPadNode->padValBin, op->inpLds->df);
    }

    if (newPcfg.srcNode == nullptr) {
      newPcfg.srcNode = newPadNode;
    } else {
      if (lastTopNodePcfg != nullptr) {
        lastTopNodePcfg->next.push_back(newPadNode);
        newPadNode->prev.push_back(lastTopNodePcfg);
      }
    }
    lastTopNodePcfg = newPadNode;
    count++;
  }

  // insert send-sync node
  SenPcfgSyncNode* newSyncNode =
      (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
  newSyncNode->name = "c" + std::to_string(coreID) + "-" +
                      newPcfg.senComponentsToString.at(pcfgType) +
                      "-sendSync-ConstWrite" + "-" +
                      std::to_string(op->uniqueID);

  newSyncNode->self = SenComponents::L3LU;
  newSyncNode->sendOrRecv = 0;  // 0--> send
  newSyncNode->external.push_back(SenComponents::LXLU0);

  if (lastTopNodePcfg != nullptr) {
    lastTopNodePcfg->next.push_back(newSyncNode);
    newSyncNode->prev.push_back(lastTopNodePcfg);
  }
}

void DcgFE::transformToPcfg(SenPcfg& newPcfg, STCDPOpHBM* op, int coreID,
                            SenComponents pcfgType, bool reqSync /*= true*/) {
  DT_CHECK(pcfgType == SenComponents::L3SU || pcfgType == SenComponents::L3LU);

  // L3SU uses output sub-pieces
  std::vector<std::pair<std::vector<std::string>, std::vector<int>>>
      dtKeysPerInnerLOs =
          pcfgType == SenComponents::L3SU
              ? clusterDtKeysUsingInnerLOs(op->coreIDtoDtKey_L3SU[coreID], op,
                                           -1, false)
              : clusterDtKeysUsingInnerLOs(op->coreIDtoDtKey_L3LU[coreID], op,
                                           coreID, true);

  DT_CHECK(dtKeysPerInnerLOs.size() == 1);
  SenPcfgNode* lastTopNodePcfg = nullptr;
  SenPcfgNode* bottomMostNode = nullptr;

  // insert send-sync node
  if (pcfgType == SenComponents::L3SU && reqSync) {
    SenPcfgSyncNode* newSyncNode =
        (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
    newSyncNode->name = "c" + std::to_string(coreID) + "-" +
                        newPcfg.senComponentsToString.at(pcfgType) +
                        "-sendSync-start" + "-" + std::to_string(op->uniqueID);

    newSyncNode->self = SenComponents::L3SU;
    newSyncNode->sendOrRecv = 0;  // 0--> send
    newSyncNode->external.push_back(SenComponents::LXSU0);

    lastTopNodePcfg = newSyncNode;
    newPcfg.srcNode = newSyncNode;
  }

  const std::vector<std::string>& innerLoopOrder = dtKeysPerInnerLOs[0].first;
  const std::vector<int>& dtKeys = dtKeysPerInnerLOs[0].second;

  // used for non-analytical mode
  SenPcfgMvloopNode* newOuterLoopNode;
  SenPcfgMvloopBranchNode* newOuterLoopBranchNode;

  // used for analytical mode
  SenPcfgNode* lastTopNodeOuter = nullptr;
  SenPcfgNode* firstBotNodeOuter = nullptr;

  std::map<std::string, std::string> dimToLoopNameOuter;

  // checks for non-analytical mode
  bool isNotAN_reqFullUnroll = false;
  if (!op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
    // check all pieces
    // full unroll if 1. both Multicast & unicast transfers
    // 2. piece sizes differs
    bool hasMultiCast = false;
    bool hasUniCast = false;
    bool pSizeDiffers = false;

    if (dtKeys.size() > 1) {
      for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
        int inpSPIDx = dtKeys.at(idx3);

        if (op->dtTable_.at(inpSPIDx).myGTR.numSharers == 1) {
          hasUniCast = true;
        } else if (op->dtTable_.at(inpSPIDx).myGTR.numSharers > 1) {
          hasMultiCast = true;
        } else if (op->dtTable_.at(inpSPIDx).myGTR.numSharers == 0) {
          hasUniCast = true;
          DT_CHECK(!op->reqMulticast);
        } else {
          if (pcfgType == SenComponents::L3SU)
            hasUniCast = true;
          else
            DT_CHECK(0);  // invalid
        }

        if (idx3 == 0) continue;

        if (op->inpSP_[inpSPIDx].dimToSize_ !=
            op->inpSP_.at(dtKeys.at(0)).dimToSize_)
          pSizeDiffers = true;
      }
    }

    if ((hasMultiCast && hasUniCast) || pSizeDiffers || dtKeys.size() > 2) {
      isNotAN_reqFullUnroll = true;  // we will unroll everything
      // we don't allow syncs
      // DT_CHECK(!reqSync);
      // we allow sync in non-analytical mode only for L3SU
      if (reqSync) DT_CHECK(pcfgType == SenComponents::L3SU);
    }
  }

  if (op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
    std::vector<std::string>& layoutDimOrder_ =
        pcfgType == SenComponents::L3LU ? op->inpLds->layoutDimOrder_
        : op->inpLds->isHbmPinned()     ? op->inpLds->layoutDimOrder_
                                        : op->outLds->layoutDimOrder_;

    auto& loopCountANN = pcfgType == SenComponents::L3SU
                             ? op->coreIDtoANInfo.at(coreID).loopCountL3SU
                             : op->coreIDtoANInfo.at(coreID).loopCount;

    DT_CHECK(loopCountANN.size() == layoutDimOrder_.size());

    for (int lpIdx = layoutDimOrder_.size() - 1; lpIdx >= 0; lpIdx--) {
      const auto& loopDimName = layoutDimOrder_[lpIdx];
      SenPcfgMvloopNode* newOuterMvLoopNode =
          (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);
      dimToLoopNameOuter[loopDimName] =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-OL-" + loopDimName +
          "-" + std::to_string(op->uniqueID);
      newOuterMvLoopNode->name = dimToLoopNameOuter[loopDimName];
      newOuterMvLoopNode->loopName = dimToLoopNameOuter[loopDimName];
      DT_CHECK(loopCountANN.count(loopDimName));
      newOuterMvLoopNode->loopCount = loopCountANN.at(loopDimName);

      // inner loop end
      SenPcfgMvloopBranchNode* newOuterLoopBranchNode =
          (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
              SenPcfgNode::Type::MVLOOPBRANCH);

      newOuterLoopBranchNode->name =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-OLBranch-" +
          loopDimName + "-" + std::to_string(op->uniqueID);
      newOuterLoopBranchNode->loopNode = newOuterMvLoopNode;
      newOuterLoopBranchNode->next.push_back(newOuterMvLoopNode);

      // hook in NodeGraph
      if (lpIdx != layoutDimOrder_.size() - 1) {
        newOuterMvLoopNode->prev.push_back(lastTopNodeOuter);
        lastTopNodeOuter->next.push_back(newOuterMvLoopNode);
        firstBotNodeOuter->prev.push_back(newOuterLoopBranchNode);
        newOuterLoopBranchNode->next.push_back(firstBotNodeOuter);
      } else {
        if (newPcfg.srcNode == nullptr) {
          newPcfg.srcNode = newOuterMvLoopNode;
        } else {
          if (lastTopNodePcfg != nullptr) {
            lastTopNodePcfg->next.push_back(newOuterMvLoopNode);
            newOuterMvLoopNode->prev.push_back(lastTopNodePcfg);
          }
        }
      }
      lastTopNodeOuter = newOuterMvLoopNode;
      firstBotNodeOuter = newOuterLoopBranchNode;

      if (lpIdx == layoutDimOrder_.size() - 1) {
        bottomMostNode = newOuterLoopBranchNode;
      }
    }
  } else {
    // outer loop
    newOuterLoopNode =
        (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);

    newOuterLoopNode->name = "c" + std::to_string(coreID) + "-" +
                             newPcfg.senComponentsToString.at(pcfgType) +
                             "-Outer-Loop" + "-" + std::to_string(op->uniqueID);
    newOuterLoopNode->loopName = newOuterLoopNode->name;
    newOuterLoopNode->loopCount = isNotAN_reqFullUnroll ? 1 : dtKeys.size();

    // outer loop end
    newOuterLoopBranchNode = (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
        SenPcfgNode::Type::MVLOOPBRANCH);

    newOuterLoopBranchNode->name = "c" + std::to_string(coreID) + "-" +
                                   newPcfg.senComponentsToString.at(pcfgType) +
                                   "-Outer-LoopBranch" + "-" +
                                   std::to_string(op->uniqueID);
    newOuterLoopBranchNode->loopNode = newOuterLoopNode;
    newOuterLoopBranchNode->next.push_back(newOuterLoopNode);

    if (newPcfg.srcNode == nullptr) {
      newPcfg.srcNode = newOuterLoopNode;
    } else {
      if (lastTopNodePcfg != nullptr) {
        lastTopNodePcfg->next.push_back(newOuterLoopNode);
        newOuterLoopNode->prev.push_back(lastTopNodePcfg);
      }
    }

    lastTopNodeOuter = newOuterLoopNode;
    firstBotNodeOuter = newOuterLoopBranchNode;
    bottomMostNode = newOuterLoopBranchNode;
  }

  if (pcfgType == SenComponents::L3SU && reqSync && !isNotAN_reqFullUnroll) {
    // insert send-sync node
    SenPcfgSyncNode* newSyncNodeSend =
        (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
    newSyncNodeSend->name = "c" + std::to_string(coreID) + "-" +
                            newPcfg.senComponentsToString.at(pcfgType) +
                            "-sendSync-0" + "-" + std::to_string(op->uniqueID);

    newSyncNodeSend->self = pcfgType == SenComponents::L3LU
                                ? SenComponents::L3LU
                                : SenComponents::L3SU;
    newSyncNodeSend->sendOrRecv = 0;  // 0--> send
    newSyncNodeSend->external.push_back(pcfgType == SenComponents::L3LU
                                            ? SenComponents::LXLU0
                                            : SenComponents::LXSU0);

    // hook in NodeGraph
    lastTopNodeOuter->next.push_back(newSyncNodeSend);
    newSyncNodeSend->prev.push_back(lastTopNodeOuter);
    lastTopNodeOuter = newSyncNodeSend;
  }

  // insert recieve-sync node
  bool blockSync = false;
  if (pcfgType == SenComponents::L3SU && reqSync && isNotAN_reqFullUnroll)
    blockSync = true;
  if (reqSync && !blockSync) {
    SenPcfgSyncNode* newSyncNodeRcv =
        (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
    newSyncNodeRcv->name = "c" + std::to_string(coreID) + "-" +
                           newPcfg.senComponentsToString.at(pcfgType) +
                           "-receiveSync-0" + "-" +
                           std::to_string(op->uniqueID);

    newSyncNodeRcv->self = pcfgType == SenComponents::L3LU
                               ? SenComponents::L3LU
                               : SenComponents::L3SU;
    newSyncNodeRcv->sendOrRecv = 1;  // 0--> send
    newSyncNodeRcv->external.push_back(pcfgType == SenComponents::L3LU
                                           ? SenComponents::LXLU0
                                           : SenComponents::LXSU0);

    // hook in NodeGraph
    lastTopNodeOuter->next.push_back(newSyncNodeRcv);
    newSyncNodeRcv->prev.push_back(lastTopNodeOuter);
    lastTopNodeOuter = newSyncNodeRcv;
  }

  // Data Transfer
  SenPcfgDtNode* newDtNode =
      (SenPcfgDtNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::DATATRANSFER);
  DtPair newDtPair;

  if (pcfgType == SenComponents::L3LU) {
    // LXLUSUFIFO transfer: to LX
    newDtPair.src_ = SenComponents::HBM;
    newDtPair.dst_ = SenComponents::LX;
  } else {
    // LXLUSUFIFO transfer: from LX
    newDtPair.src_ = SenComponents::LX;
    newDtPair.dst_ = SenComponents::HBM;
  }

  newDtNode->name = "c" + std::to_string(coreID) + "-" +
                    newPcfg.senComponentsToString.at(pcfgType) + "-ringDT-" +
                    newPcfg.senComponentsToString.at(newDtPair.src_) + "-" +
                    newPcfg.senComponentsToString.at(newDtPair.dst_) + "-" +
                    std::to_string(op->uniqueID);
  newDtNode->coreletId = -1;  // corelet independent
  newDtNode->dtInfo = nullptr;
  newDtNode->srcDest = newDtPair;
  newDtNode->dsInfo = nullptr;
  newDtNode->dimLayoutOrder = (pcfgType == SenComponents::L3LU)
                                  ? op->inpLds->layoutDimOrder_
                                  : op->outLds->layoutDimOrder_;

  // figure out bigDimToSize
  std::map<std::string, double> bigDimToSize_;
  if (pcfgType == SenComponents::L3LU) {
    // bigDimToSize_ = op->inpSP_.at(dtKeys.at(0)).bigDimToSize_;
    bigDimToSize_ = op->inpLds->dimToLayoutSize_;
  } else {
    bigDimToSize_ = op->outLds->dimToLayoutSize_;
  }

  // add burst Info, used in SenProg
  newDtNode->useBurst = true;

  // check if all biDimToSize are same..
  for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
    int inpSPIDx = dtKeys.at(idx3);
    int outSPIDx;
    if (pcfgType == SenComponents::L3LU) {
      DT_CHECK(op->inpSP_.size() > inpSPIDx);
      DT_CHECK(bigDimToSize_ == op->inpSP_[inpSPIDx].bigDimToSize_);
    } else {
      DT_CHECK(op->dtTable_.at(inpSPIDx).cIDXs.size() == 1);
      DT_CHECK(bigDimToSize_ ==
               op->outSP_[op->dtTable_.at(inpSPIDx).cIDXs[0]].bigDimToSize_);
    }

    if (!op->dtTable_.at(inpSPIDx).useBurst) newDtNode->useBurst = false;
  }

  newDtNode->myBigDimSize = bigDimToSize_;
  makeStickLevelAdjustments(
      newDtNode->myBigDimSize,
      (pcfgType == SenComponents::L3LU) ? op->inpLds : op->outLds);

  // set src & dest Addresses
  fillAddr(newDtNode->SrcStartAddr(),
           op->inpSP_.at(dtKeys.at(0)).placement.StartAddr(), 0);
  if (pcfgType == SenComponents::L3LU) {
    auto outSPIDx = getIdxForMatchingCMenID(
        op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
    fillAddr(newDtNode->DestStartAddr(),
             op->outSP_.at(outSPIDx).placement.StartAddr(), 0);
  } else {
    fillAddr(newDtNode->DestStartAddr(),
             op->outSP_.at(op->dtTable_.at(dtKeys.at(0)).cIDXs[0])
                 .placement.StartAddr(),
             0);
  }

  if (!op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
    for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
      int inpSPIDx = dtKeys.at(idx3);
      auto outSPIDx =
          (pcfgType == SenComponents::L3LU)
              ? getIdxForMatchingCMenID(
                    op->outSP_, op->dtTable_.at(dtKeys.at(idx3)).cIDXs, coreID)
              : (op->dtTable_.at(dtKeys.at(idx3)).cIDXs[0]);
      std::pair<PcfgLccrCond, FoldManager<int64_t>> newPcfgLccrCond;
      newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
      newPcfgLccrCond.first.condOp = CondOp::EQ;
      newPcfgLccrCond.first.condVal =
          dtKeys.size() - 1 - idx3;  // order matters

      fillAddr(newPcfgLccrCond.second,
               op->outSP_.at(outSPIDx).placement.StartAddr(), 0);  // count
      newDtNode->DestStartCondAndVal().emplace_back(newPcfgLccrCond);
      fillAddr(newPcfgLccrCond.second,
               op->inpSP_.at(inpSPIDx).placement.StartAddr(), 0);
      newDtNode->SrcStartCondAndVal().emplace_back(newPcfgLccrCond);
    }
  } else {
    // implement toggle op
    auto& layoutDimOrder_ = (pcfgType == SenComponents::L3LU)
                                ? op->inpLds->layoutDimOrder_
                                : op->outLds->layoutDimOrder_;
    for (const auto& loopDimName : layoutDimOrder_) {
      std::pair<PcfgLccrCond, FoldManager<int64_t>> newPcfgLccrCond;
      newPcfgLccrCond.first.loopName = dimToLoopNameOuter.at(loopDimName);
      newPcfgLccrCond.first.condOp = CondOp::TOGGLE;
      newPcfgLccrCond.first.condVal = -1;  // don't care

      FoldManager<int64_t> addrValue;

      if (dtKeys.size() >= 2) {
        auto outSPIDx = getIdxForMatchingCMenID(
            op->outSP_, op->dtTable_[dtKeys[1]].cIDXs, coreID);
        fillAddr(addrValue,
                 pcfgType == SenComponents::L3LU
                     ? op->outSP_.at(outSPIDx).placement.StartAddr()
                     : op->inpSP_[dtKeys[1]].placement.StartAddr(),
                 0);
      } else {
        auto outSPIDx = getIdxForMatchingCMenID(
            op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
        fillAddr(addrValue,
                 pcfgType == SenComponents::L3LU
                     ? op->outSP_.at(outSPIDx).placement.StartAddr()
                     : op->inpSP_.at(dtKeys.at(0)).placement.StartAddr(),
                 0);
      }

      newPcfgLccrCond.second.clone(addrValue);  // count
      if (pcfgType == SenComponents::L3LU) {
        newDtNode->DestStartCondAndVal().emplace_back(newPcfgLccrCond);
      } else {
        // L3SU
        newDtNode->SrcStartCondAndVal().emplace_back(newPcfgLccrCond);
      }
    }
  }

  // no burst for now, add later..
  // for (auto& myDim : newDtNode->myBigDimSize) {
  if (newDtNode->myBigDimSize.size()) {
    // newDtNode->myLitDimSize[myDim.first] = 1;
    // will be piece size
    auto outSPIDx = getIdxForMatchingCMenID(
        op->outSP_, op->dtTable_[dtKeys.back()].cIDXs, coreID);
    newDtNode->myLitDimSize =
        pcfgType == SenComponents::L3SU
            ? op->inpSP_[dtKeys.back()]
                  .bigDimToSize_ /* input sub-piece in LX for L3SU*/
            : op->outSP_.at(outSPIDx)
                  .bigDimToSize_; /* output sub-piece in LX for L3LU*/

    if (pcfgType != SenComponents::L3SU) {
      if (!op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
        for (auto& kv : newDtNode->myLitDimSize) {
          DT_CHECK(op->outSP_.at(outSPIDx).dimToSize_.count(kv.first));
          auto gap =
              kv.second - op->outSP_.at(outSPIDx).dimToSize_.at(kv.first);
          DT_CHECK(gap >= 0);
          newDtNode->myLitDimGap[kv.first] = gap;
        }
      } else {
        DT_CHECK(op->outSP_.at(outSPIDx).dimToSize_ ==
                 op->outSP_.at(outSPIDx).bigDimToSize_);
      }
    } else {
      if (!op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
        for (auto& kv : newDtNode->myLitDimSize) {
          DT_CHECK(op->inpSP_[dtKeys.back()].dimToSize_.count(kv.first));
          auto gap =
              kv.second - op->inpSP_[dtKeys.back()].dimToSize_.at(kv.first);
          DT_CHECK(gap >= 0);
          newDtNode->myLitDimGap[kv.first] = gap;
        }
      } else {
        DT_CHECK(op->inpSP_[dtKeys.back()].dimToSize_ ==
                 op->inpSP_[dtKeys.back()].bigDimToSize_);
      }
    }

    makeStickLevelAdjustments(
        newDtNode->myLitDimSize,
        (pcfgType == SenComponents::L3LU) ? op->inpLds : op->outLds);
  }

  // set bigStAddrOffsets for each outer loop
  if (op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
    auto& layoutDimOrder_ = (pcfgType == SenComponents::L3LU)
                                ? op->inpLds->layoutDimOrder_
                                : op->outLds->layoutDimOrder_;
    for (const auto& loopDimName : layoutDimOrder_) {
      PcfgDtOffsets newDtOffset;
      init(newDtOffset);

      // find location in dimLayoutOrder
      newDtOffset.dimOffset =
          op->coreIDtoANInfo.at(coreID).getAddrInfo(pcfgType)->getOffset(
              loopDimName);

      newDtNode->bigStAddrOffsets[dimToLoopNameOuter[loopDimName]] =
          newDtOffset;
    }
  }

  if (pcfgType == SenComponents::L3LU && op->reqMulticast) {
    GTRBurstInfo newCondGTR;
    newCondGTR.groupID = op->dtTable_.at(dtKeys.at(0)).myGTR.groupID;
    newCondGTR.numSharers = op->dtTable_.at(dtKeys.at(0)).myGTR.numSharers;
    newCondGTR.count = op->dtTable_.at(dtKeys.at(0)).myGTR.count;
    newCondGTR.srcNodeID = op->dtTable_.at(dtKeys.at(0)).myGTR.srcNodeID;
    newCondGTR.useBurst = newDtNode->useBurst;
    DT_CHECK(newCondGTR.srcNodeID == -1);

    if (op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
      for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
        int inpSPIDx = dtKeys.at(idx3);
        DT_CHECK(newCondGTR.numSharers ==
                 op->dtTable_.at(dtKeys.at(idx3)).myGTR.numSharers);
      }

      // fill GTR info if needed
      if (newCondGTR.numSharers > 1) {
        // DT_CHECK(dtKeys.size() ==
        //         1);  // we don't support more than one piece right now
        DT_CHECK(op->inpLds->layoutDimOrder_.size());

        auto loopDimName = op->inpLds->layoutDimOrder_.front();
        PcfgLccrCond newPcfgLccrCond;
        newPcfgLccrCond.loopName = dimToLoopNameOuter.at(loopDimName);
        newPcfgLccrCond.condOp = CondOp::ALWAYS;
        newPcfgLccrCond.condVal = -1;  // don't care
        newDtNode->GTRAndBurstCondAndVal.push_back(
            std::make_pair(newPcfgLccrCond, newCondGTR));
      }
    } else if (!isNotAN_reqFullUnroll) {
      DT_CHECK(!op->coreIDtoANInfo.at(coreID).isAnalyticalMode);
      for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
        int inpSPIDx = dtKeys.at(idx3);
        if (op->dtTable_.at(dtKeys.at(idx3)).myGTR.numSharers > 1) {
          GTRBurstInfo newCondGTR;
          newCondGTR.groupID = op->dtTable_.at(inpSPIDx).myGTR.groupID;
          newCondGTR.numSharers = op->dtTable_.at(inpSPIDx).myGTR.numSharers;
          newCondGTR.count = op->dtTable_.at(inpSPIDx).myGTR.count;
          newCondGTR.srcNodeID = op->dtTable_.at(inpSPIDx).myGTR.srcNodeID;
          newCondGTR.useBurst = op->dtTable_.at(inpSPIDx).useBurst;

          PcfgLccrCond newPcfgLccrCond;
          newPcfgLccrCond.loopName = newOuterLoopNode->name;
          newPcfgLccrCond.condOp = CondOp::EQ;
          newPcfgLccrCond.condVal = dtKeys.size() - 1 - idx3;  // order matters

          newDtNode->GTRAndBurstCondAndVal.push_back(
              std::make_pair(newPcfgLccrCond, newCondGTR));
        }
      }
    }
  }

  if (!isNotAN_reqFullUnroll) {
    // optimize newDtNode burst -- loop coaleascing
    DT_CHECK(newDtNode->dimLayoutOrder.size());
    std::string merge_dim = newDtNode->dimLayoutOrder.at(0);

    int litDimSize = 1;
    int bigDimSize = 1;
    if (newDtNode->myLitDimSize.at(merge_dim) ==
        newDtNode->myBigDimSize.at(merge_dim)) {
      int mergeCount = 1;
      litDimSize = newDtNode->myLitDimSize.at(merge_dim);
      bigDimSize = newDtNode->myBigDimSize.at(merge_dim);
      for (int idx = 1; idx < newDtNode->dimLayoutOrder.size(); idx++) {
        mergeCount++;
        auto new_dim = newDtNode->dimLayoutOrder.at(idx);
        merge_dim += new_dim;
        litDimSize *= newDtNode->myLitDimSize.at(new_dim);
        bigDimSize *= newDtNode->myBigDimSize.at(new_dim);

        if (newDtNode->myLitDimGap.count(new_dim))
          DT_CHECK(newDtNode->myLitDimGap.at(new_dim) == 0);

        if (newDtNode->myLitDimSize.at(new_dim) !=
            newDtNode->myBigDimSize.at(new_dim))
          break;
      }

      if (mergeCount > 1) {
        newDtNode->myLitDimSize[merge_dim] = litDimSize;
        newDtNode->myBigDimSize[merge_dim] = bigDimSize;

        if (newDtNode->myLitDimGap.count(newDtNode->dimLayoutOrder.at(0)))
          newDtNode->myLitDimGap[merge_dim] =
              newDtNode->myLitDimGap.at(newDtNode->dimLayoutOrder.at(0));

        for (int idx = 0; idx < mergeCount; idx++) {
          newDtNode->myLitDimSize.erase(newDtNode->dimLayoutOrder.at(idx));
          newDtNode->myBigDimSize.erase(newDtNode->dimLayoutOrder.at(idx));
          if (newDtNode->myLitDimGap.count(newDtNode->dimLayoutOrder.at(idx)))
            DT_CHECK(newDtNode->myLitDimGap.erase(
                newDtNode->dimLayoutOrder.at(idx)));
        }
        std::vector<std::string> newdimLayout;
        newdimLayout.push_back(merge_dim);
        for (int idx = mergeCount; idx < newDtNode->dimLayoutOrder.size();
             idx++)
          newdimLayout.push_back(newDtNode->dimLayoutOrder.at(idx));
        newDtNode->dimLayoutOrder = newdimLayout;
      }
    }

    // hook in NodeGraph
    lastTopNodeOuter->next.push_back(newDtNode);
    newDtNode->prev.push_back(lastTopNodeOuter);
    lastTopNodeOuter = newDtNode;
  } else {
    // check for EBR sharing
    bool enEBRSharing = false;

    if (dtKeys.size() >
        sysDef.regInfoPerUnit.at(pcfgType).at(RegType::EBR).maxNum / 2) {
      auto totalSticks = (pcfgType == SenComponents::L3SU)
                             ? op->outLds->hbmSize_
                             : op->inpLds->hbmSize_;
      DT_CHECK(
          totalSticks <
          pow(2, sysDef.regInfoPerUnit.at(pcfgType).at(RegType::EAR).bitSize));
      enEBRSharing = true;
    }

    for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
      int inpSPIDx = dtKeys.at(idx3);
      int outSPIDx;
      if (pcfgType == SenComponents::L3SU) {
        outSPIDx = getIdxForMatchingCMenID(op->outSP_,
                                           op->dtTable_.at(inpSPIDx).cIDXs, -1);
      } else {
        outSPIDx = getIdxForMatchingCMenID(
            op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
      }
      SenPcfgDtNode* newUnrolledDtNode = (SenPcfgDtNode*)newPcfg.createPcfgNode(
          SenPcfgNode::Type::DATATRANSFER);

      newUnrolledDtNode->name =
          newDtNode->name + "-Unroll-" + std::to_string(idx3);
      newUnrolledDtNode->dimLayoutOrder = newDtNode->dimLayoutOrder;
      newUnrolledDtNode->srcDest = newDtNode->srcDest;
      newUnrolledDtNode->useBurst = newDtNode->useBurst;

      newUnrolledDtNode->useBurst = newDtNode->useBurst;
      newUnrolledDtNode->myBigDimSize = newDtNode->myBigDimSize;
      newUnrolledDtNode->dtInfo = nullptr;
      newUnrolledDtNode->dsInfo = nullptr;

      DT_CHECK(newDtNode->bigStAddrOffsets.size() == 0);

      if (op->dtTable_.at(inpSPIDx).myGTR.numSharers > 1) {
        GTRBurstInfo newCondGTR;
        newCondGTR.groupID = op->dtTable_.at(inpSPIDx).myGTR.groupID;
        newCondGTR.numSharers = op->dtTable_.at(inpSPIDx).myGTR.numSharers;
        newCondGTR.count = op->dtTable_.at(inpSPIDx).myGTR.count;
        newCondGTR.srcNodeID = op->dtTable_.at(inpSPIDx).myGTR.srcNodeID;
        newCondGTR.useBurst = op->dtTable_.at(inpSPIDx).useBurst;

        PcfgLccrCond newPcfgLccrCond;
        newPcfgLccrCond.loopName = newOuterLoopNode->name;
        newPcfgLccrCond.condOp = CondOp::ALWAYS;
        newPcfgLccrCond.condVal = dtKeys.size() - 1 - idx3;

        newUnrolledDtNode->GTRAndBurstCondAndVal.push_back(
            std::make_pair(newPcfgLccrCond, newCondGTR));
      }

      // no burst for now, add later..
      // newUnrolledDtNode->myLitDimSize = newDtNode->myLitDimSize; //FIXME
      // for (auto& myDim : newUnrolledDtNode->myBigDimSize) {
      if (newUnrolledDtNode->myBigDimSize.size()) {
        newUnrolledDtNode->myLitDimSize =
            pcfgType == SenComponents::L3SU
                ? op->inpSP_[inpSPIDx]
                      .bigDimToSize_ /* input sub-piece in LX for L3SU*/
                : op->outSP_.at(outSPIDx)
                      .bigDimToSize_; /* output sub-piece in LX for L3LU*/

        if (pcfgType != SenComponents::L3SU) {
          if (!op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
            for (auto& kv : newUnrolledDtNode->myLitDimSize) {
              DT_CHECK(op->outSP_.at(outSPIDx).dimToSize_.count(kv.first));
              auto gap =
                  kv.second - op->outSP_.at(outSPIDx).dimToSize_.at(kv.first);
              DT_CHECK(gap >= 0);
              newUnrolledDtNode->myLitDimGap[kv.first] = gap;
            }
          } else {
            DT_CHECK(op->outSP_.at(outSPIDx).dimToSize_ ==
                     op->outSP_.at(outSPIDx).bigDimToSize_);
          }
        } else {
          if (!op->coreIDtoANInfo.at(coreID).isAnalyticalMode) {
            for (auto& kv : newUnrolledDtNode->myLitDimSize) {
              DT_CHECK(op->inpSP_.at(inpSPIDx).dimToSize_.count(kv.first));
              auto gap =
                  kv.second - op->inpSP_.at(inpSPIDx).dimToSize_.at(kv.first);
              DT_CHECK(gap >= 0);
              newUnrolledDtNode->myLitDimGap[kv.first] = gap;
            }
          } else {
            DT_CHECK(op->inpSP_[inpSPIDx].dimToSize_ ==
                     op->inpSP_[inpSPIDx].bigDimToSize_);
          }
        }

        makeStickLevelAdjustments(
            newUnrolledDtNode->myLitDimSize,
            (pcfgType == SenComponents::L3LU) ? op->inpLds : op->outLds);
      }

      {
        fillAddr(newUnrolledDtNode->SrcStartAddr(),
                 op->inpSP_[inpSPIDx].placement.StartAddr(), 0);
        fillAddr(newUnrolledDtNode->DestStartAddr(),
                 op->outSP_[outSPIDx].placement.StartAddr(), 0);

        std::pair<PcfgLccrCond, FoldManager<int64_t>> newPcfgLccrCond;
        newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
        newPcfgLccrCond.first.condOp = CondOp::ALWAYS;
        newPcfgLccrCond.first.condVal = 0;

        if (pcfgType == SenComponents::L3LU) {
          newPcfgLccrCond.second.clone(
              newUnrolledDtNode->DestStartAddr());  // count
          newUnrolledDtNode->DestStartCondAndVal().emplace_back(
              newPcfgLccrCond);
        } else {
          newPcfgLccrCond.second.clone(newUnrolledDtNode->SrcStartAddr());
          newUnrolledDtNode->SrcStartCondAndVal().emplace_back(newPcfgLccrCond);
        }
      }

      //  use collapseFactor
      if (op->dtTable_.at(inpSPIDx).collapseFactor > 1) {
        std::string mergedDim = newUnrolledDtNode->dimLayoutOrder.at(0);
        std::string dimZero = newUnrolledDtNode->dimLayoutOrder.at(0);
        for (int i = 1; i < op->dtTable_.at(inpSPIDx).collapseFactor; i++) {
          std::string presentDim = newUnrolledDtNode->dimLayoutOrder.at(i);
          mergedDim += presentDim;
          newUnrolledDtNode->myBigDimSize.at(dimZero) *=
              newUnrolledDtNode->myBigDimSize.at(presentDim);
          if (i == op->dtTable_.at(inpSPIDx).collapseFactor - 1) {
            DT_CHECK(newUnrolledDtNode->myLitDimGap.at(dimZero) == 0);
            newUnrolledDtNode->myLitDimGap.at(dimZero) =
                newUnrolledDtNode->myLitDimSize.at(dimZero) *
                newUnrolledDtNode->myLitDimGap.at(presentDim);
          } else {
            DT_CHECK(newUnrolledDtNode->myLitDimGap.at(presentDim) == 0);
          }
          newUnrolledDtNode->myLitDimSize.at(dimZero) *=
              newUnrolledDtNode->myLitDimSize.at(presentDim);

          newUnrolledDtNode->myLitDimSize.erase(presentDim);
          newUnrolledDtNode->myLitDimGap.erase(presentDim);
          newUnrolledDtNode->myBigDimSize.erase(presentDim);
        }
        newUnrolledDtNode->myBigDimSize[mergedDim] =
            newUnrolledDtNode->myBigDimSize.at(dimZero);
        newUnrolledDtNode->myLitDimSize[mergedDim] =
            newUnrolledDtNode->myLitDimSize.at(dimZero);
        newUnrolledDtNode->myLitDimGap[mergedDim] =
            newUnrolledDtNode->myLitDimGap.at(dimZero);
        newUnrolledDtNode->myLitDimSize.erase(dimZero);
        newUnrolledDtNode->myBigDimSize.erase(dimZero);
        newUnrolledDtNode->myLitDimGap.erase(dimZero);

        std::vector<std::string> newdimLayout;
        newdimLayout.push_back(mergedDim);
        for (int dimIdx = op->dtTable_.at(inpSPIDx).collapseFactor;
             dimIdx < newUnrolledDtNode->dimLayoutOrder.size(); dimIdx++)
          newdimLayout.push_back(newDtNode->dimLayoutOrder.at(dimIdx));

        newUnrolledDtNode->dimLayoutOrder = newdimLayout;
      }

      if (enEBRSharing) {
        newUnrolledDtNode->EBRTag = std::make_pair(
            true, (pcfgType == SenComponents::L3SU) ? op->outLds->ldsName_
                                                    : op->inpLds->ldsName_);
      }

      if (pcfgType == SenComponents::L3SU && reqSync) {
        // insert send-sync node
        SenPcfgSyncNode* newSyncNodeSend =
            (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
        newSyncNodeSend->name =
            "c" + std::to_string(coreID) + "-" +
            newPcfg.senComponentsToString.at(pcfgType) + "-sendSync-0" + "-" +
            "-Unroll-" + std::to_string(idx3) + std::to_string(op->uniqueID);

        newSyncNodeSend->self = pcfgType;
        newSyncNodeSend->sendOrRecv = 0;  // 0--> send
        newSyncNodeSend->external.push_back(SenComponents::LXSU0);

        // hook in NodeGraph
        lastTopNodeOuter->next.push_back(newSyncNodeSend);
        newSyncNodeSend->prev.push_back(lastTopNodeOuter);
        lastTopNodeOuter = newSyncNodeSend;

        SenPcfgSyncNode* newSyncNodeRcv =
            (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
        newSyncNodeRcv->name = "c" + std::to_string(coreID) + "-" +
                               newPcfg.senComponentsToString.at(pcfgType) +
                               "-receiveSync-0" + "-" + "-Unroll-" +
                               std::to_string(idx3) +
                               std::to_string(op->uniqueID);

        newSyncNodeRcv->self = pcfgType;
        newSyncNodeRcv->sendOrRecv = 1;  // 0--> send
        newSyncNodeRcv->external.push_back(SenComponents::LXSU0);

        // hook in NodeGraph
        lastTopNodeOuter->next.push_back(newSyncNodeRcv);
        newSyncNodeRcv->prev.push_back(lastTopNodeOuter);
        lastTopNodeOuter = newSyncNodeRcv;
      }

      // hook in NodeGraph
      lastTopNodeOuter->next.push_back(newUnrolledDtNode);
      newUnrolledDtNode->prev.push_back(lastTopNodeOuter);
      lastTopNodeOuter = newUnrolledDtNode;
    }
  }

  if (pcfgType != SenComponents::L3SU && reqSync) {
    // insert send-sync node
    SenPcfgSyncNode* newSyncNodeSend =
        (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
    newSyncNodeSend->name = "c" + std::to_string(coreID) + "-" +
                            newPcfg.senComponentsToString.at(pcfgType) +
                            "-sendSync-0" + "-" + std::to_string(op->uniqueID);

    newSyncNodeSend->self = pcfgType == SenComponents::L3LU
                                ? SenComponents::L3LU
                                : SenComponents::L3SU;
    newSyncNodeSend->sendOrRecv = 0;  // 0--> send
    newSyncNodeSend->external.push_back(pcfgType == SenComponents::L3LU
                                            ? SenComponents::LXLU0
                                            : SenComponents::LXSU0);

    // hook in NodeGraph
    lastTopNodeOuter->next.push_back(newSyncNodeSend);
    newSyncNodeSend->prev.push_back(lastTopNodeOuter);
    lastTopNodeOuter = newSyncNodeSend;
  }

  // hook in NodeGraph
  lastTopNodeOuter->next.push_back(firstBotNodeOuter);
  firstBotNodeOuter->prev.push_back(lastTopNodeOuter);

  // extra receive for L3SU
  if (pcfgType == SenComponents::L3LU && reqSync) {
    DT_CHECK(bottomMostNode != nullptr);
    SenPcfgSyncNode* newSyncNodeExtra =
        (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
    newSyncNodeExtra->name = "c" + std::to_string(coreID) + "-" +
                             newPcfg.senComponentsToString.at(pcfgType) +
                             "-receiveSync-extra" + "-" +
                             std::to_string(op->uniqueID);

    newSyncNodeExtra->self = SenComponents::L3LU;
    newSyncNodeExtra->sendOrRecv = 1;  // 0--> send
    newSyncNodeExtra->external.push_back(SenComponents::LXLU0);

    // hook in NodeGraph
    bottomMostNode->next.push_back(newSyncNodeExtra);
    newSyncNodeExtra->prev.push_back(bottomMostNode);
    bottomMostNode = newSyncNodeExtra;
  }
}

void DcgFE::transformToPcfg(SenPcfg& newPcfg, baseSTCDPOp* op, int coreID,
                            SenComponents pcfgType, bool reqSync /* = false*/) {
  DT_CHECK(is_any_of(pcfgType, SenComponents::L3SU, SenComponents::L3LU,
                     SenComponents::LXSU0, SenComponents::LXLU0));

  if (is_any_of(pcfgType, SenComponents::L3SU, SenComponents::L3LU)) {
    DT_CHECK(!isSRQProne(op));
  }

  bool needConstRcvSync = false;
  if (op->name == OpFuncs::ResizeNNLX || op->name == OpFuncs::ResizeNNHBM) {
    const auto genConstIntr = op->name == OpFuncs::ResizeNNLX
                                  ? ((ResizeNNLX*)op)->genConstIntr
                                  : ((ResizeNNHBM*)op)->genConstIntr;
    if (genConstIntr && pcfgType == SenComponents::LXLU0) {
      needConstRcvSync = true;
    }
  }

  // L3SU uses input sub-pieces
  std::vector<std::pair<std::vector<std::string>, std::vector<int>>>
      dtKeysPerInnerLOs =
          pcfgType == SenComponents::L3SU
              ? clusterDtKeysUsingInnerLOs(op->coreIDtoDtKey_L3SU[coreID], op,
                                           coreID, true)
          : pcfgType == SenComponents::L3LU
              ? clusterDtKeysUsingInnerLOs(op->coreIDtoDtKey_L3LU[coreID], op,
                                           coreID, false)
              : clusterDtKeysUsingInnerLOs(op->coreIDtoDtKey_LX[coreID], op,
                                           coreID,
                                           (pcfgType == SenComponents::LXLU0));

  SenPcfgNode* lastTopNodePcfg = nullptr;

  if (needConstRcvSync) {
    SenPcfgSyncNode* newSyncNode =
        (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
    newSyncNode->name = "c" + std::to_string(coreID) + "-" +
                        newPcfg.senComponentsToString.at(pcfgType) +
                        "-sendSync-ConstWrite" + "-" +
                        std::to_string(op->uniqueID);

    newSyncNode->self = SenComponents::LXLU0;
    newSyncNode->sendOrRecv = 1;  // 1--> rcv
    newSyncNode->external.push_back(SenComponents::L3LU);

    lastTopNodePcfg = newSyncNode;
    newPcfg.srcNode = newSyncNode;
  }

  // insert send-sync node
  if (reqSync && pcfgType == SenComponents::LXLU0) {
    DT_CHECK(op->name == OpFuncs::STCDPOpHBM ||
             op->name == OpFuncs::ResizeNNHBM);
    SenPcfgSyncNode* newSyncNode =
        (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
    newSyncNode->name = "c" + std::to_string(coreID) + "-" +
                        newPcfg.senComponentsToString.at(pcfgType) +
                        "-sendSync-start" + "-" + std::to_string(op->uniqueID);

    newSyncNode->self = SenComponents::LXLU0;
    newSyncNode->sendOrRecv = 0;  // 0--> send
    newSyncNode->external.push_back(SenComponents::L3LU);

    lastTopNodePcfg = newSyncNode;
    newPcfg.srcNode = newSyncNode;
  }

  for (int idx = 0; idx < dtKeysPerInnerLOs.size(); idx++) {
    SenPcfgNode* lastNodethisLoop = nullptr;
    const std::vector<std::string>& loopOrder = dtKeysPerInnerLOs[idx].first;
    const std::vector<int>& dtKeys = dtKeysPerInnerLOs[idx].second;

    bool has_innermost_dyn_loop = false;
    // outer loop
    SenPcfgMvloopNode* newOuterLoopNode =
        (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);

    newOuterLoopNode->name = "c" + std::to_string(coreID) + "-" +
                             newPcfg.senComponentsToString.at(pcfgType) +
                             "-Outer-Loop-" + std::to_string(idx) + "-" +
                             std::to_string(op->uniqueID);
    newOuterLoopNode->loopName = newOuterLoopNode->name;
    newOuterLoopNode->loopCount = dtKeys.size();

    // outer loop end
    SenPcfgMvloopBranchNode* newOuterLoopBranchNode =
        (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
            SenPcfgNode::Type::MVLOOPBRANCH);

    newOuterLoopBranchNode->name = "c" + std::to_string(coreID) + "-" +
                                   newPcfg.senComponentsToString.at(pcfgType) +
                                   "-Outer-LoopBranch-" + std::to_string(idx) +
                                   "-" + std::to_string(op->uniqueID);
    newOuterLoopBranchNode->loopNode = newOuterLoopNode;
    newOuterLoopBranchNode->next.push_back(newOuterLoopNode);

    if (newPcfg.srcNode == nullptr) {
      newPcfg.srcNode = newOuterLoopNode;
    } else {
      if (lastTopNodePcfg != nullptr) {
        lastTopNodePcfg->next.push_back(newOuterLoopNode);
        newOuterLoopNode->prev.push_back(lastTopNodePcfg);
      }
    }

    // inner loops top and bottom nodes
    SenPcfgNode* lastTopNodeInner = newOuterLoopNode;
    SenPcfgNode* firstBotNodeInner = newOuterLoopBranchNode;
    lastNodethisLoop = newOuterLoopBranchNode;

    // insert recieve-sync & send-sync node
    if (reqSync && (pcfgType == SenComponents::LXLU0 ||
                    pcfgType == SenComponents::LXSU0)) {
      if (dtKeysPerInnerLOs.size() > 1) {
        // check if pieces bigDimSizes are different
        bool size_differs = false;
        for (int nidx = 1; nidx < dtKeysPerInnerLOs.size(); nidx++) {
          const std::vector<int>& dtKeys_next = dtKeysPerInnerLOs[nidx].second;
          DT_CHECK(dtKeys_next.size());
          DT_CHECK(dtKeys.size());
          if (op->inpSP_.at(dtKeys.at(0)).bigDimToSize_ !=
              op->inpSP_.at(dtKeys_next.at(0)).bigDimToSize_) {
            size_differs = true;
            break;
          }
        }
        if (size_differs)
          DT_ERROR(
              "Cannot have unequal buffer sizes across subpieces during "
              "STCDPOpHBM");
        else
          DT_ERROR("Unexpected error");  // this should not happen, so it is
                                         // unexpected & issue is unknown
      } else if (dtKeysPerInnerLOs.size() != 1) {
        DT_ERROR("Unexpected error");  // this should never happen
      }

      // send sync
      SenPcfgSyncNode* newSyncNodeSend =
          (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
      newSyncNodeSend->name = "c" + std::to_string(coreID) + "-" +
                              newPcfg.senComponentsToString.at(pcfgType) +
                              "-sendSync-0" + "-" + std::to_string(idx) + "-" +
                              std::to_string(op->uniqueID);

      newSyncNodeSend->self = pcfgType == SenComponents::LXLU0
                                  ? SenComponents::LXLU0
                                  : SenComponents::LXSU0;
      newSyncNodeSend->sendOrRecv = 0;  // 0--> send
      newSyncNodeSend->external.push_back(pcfgType == SenComponents::LXLU0
                                              ? SenComponents::L3LU
                                              : SenComponents::L3SU);

      // hook in NodeGraph
      if (pcfgType == SenComponents::LXLU0) {
        // sync will come before wait-for-sync
        lastTopNodeInner->next.push_back(newSyncNodeSend);
        newSyncNodeSend->prev.push_back(lastTopNodeInner);
        lastTopNodeInner = newSyncNodeSend;
      } else {
        // sync will come at the end of the loop
        firstBotNodeInner->prev.push_back(newSyncNodeSend);
        newSyncNodeSend->next.push_back(firstBotNodeInner);
        firstBotNodeInner = newSyncNodeSend;
      }

      SenPcfgSyncNode* newSyncNodeRcv =
          (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
      newSyncNodeRcv->name = "c" + std::to_string(coreID) + "-" +
                             newPcfg.senComponentsToString.at(pcfgType) +
                             "-receiveSync-0" + "-" + std::to_string(idx) +
                             "-" + std::to_string(op->uniqueID);

      newSyncNodeRcv->self = pcfgType == SenComponents::LXLU0
                                 ? SenComponents::LXLU0
                                 : SenComponents::LXSU0;
      newSyncNodeRcv->sendOrRecv = 1;  // 0--> send
      newSyncNodeRcv->external.push_back(pcfgType == SenComponents::LXLU0
                                             ? SenComponents::L3LU
                                             : SenComponents::L3SU);

      // hook in NodeGraph
      lastTopNodeInner->next.push_back(newSyncNodeRcv);
      newSyncNodeRcv->prev.push_back(lastTopNodeInner);
      lastTopNodeInner = newSyncNodeRcv;

      // extra receive for LXSU
      if (pcfgType == SenComponents::LXSU0) {
        SenPcfgSyncNode* newSyncNodeExtra =
            (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
        newSyncNodeExtra->name =
            "c" + std::to_string(coreID) + "-" +
            newPcfg.senComponentsToString.at(pcfgType) + "-receiveSync-extra" +
            "-" + std::to_string(idx) + "-" + std::to_string(op->uniqueID);

        newSyncNodeExtra->self = SenComponents::LXSU0;
        newSyncNodeExtra->sendOrRecv = 1;  // 0--> send
        newSyncNodeExtra->external.push_back(SenComponents::L3SU);

        // hook in NodeGraph
        lastNodethisLoop->next.push_back(newSyncNodeExtra);
        newSyncNodeExtra->prev.push_back(lastNodethisLoop);
        lastNodethisLoop = newSyncNodeExtra;
      }
    }

    // find minLoopCollapseFactor (mlCf)
    int mlCf = -1;
    std::map<std::string, std::vector<long>> dimToLoopCount;  // compact form
    for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
      int inpSPIDx = dtKeys.at(idx3);
      DT_CHECK(op->dtTable_.at(inpSPIDx).collapseFactor > 0);
      if (mlCf == -1)
        mlCf = op->dtTable_.at(inpSPIDx).collapseFactor;
      else if (op->dtTable_.at(inpSPIDx).collapseFactor < mlCf)
        mlCf = op->dtTable_.at(inpSPIDx).collapseFactor;

      // find loop values
      for (int lpIdx = loopOrder.size() - 1; lpIdx >= 0; lpIdx--) {
        const auto& loopDimName = loopOrder[lpIdx];
        int outSPIDx;
        if (pcfgType == SenComponents::L3LU) {
          outSPIDx = getIdxForMatchingCMenID(
              op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
          DT_CHECK(outSPIDx >= 0);
          DT_CHECK(op->outSP_.size() > outSPIDx);
        } else {
          DT_CHECK(op->inpSP_.size() > inpSPIDx);
        }
        int loopCount = pcfgType == SenComponents::L3LU
                            ? op->outSP_[outSPIDx].dimToSize_.at(loopDimName)
                            : op->inpSP_[inpSPIDx].dimToSize_.at(loopDimName);
        // stick-level adjustments
        LdsInfo* lds =
            (pcfgType == SenComponents::L3LU) ? op->outLds : op->inpLds;
        if (DCGUtils::isValPresent(lds->stickDimOrder_, loopDimName)) {
          loopCount = ceil(loopCount / lds->dimToStickSize_[loopDimName]);
        }

        if (dimToLoopCount.count(loopDimName)) {
          if (dimToLoopCount.at(loopDimName).back() != loopCount)
            dimToLoopCount.at(loopDimName).push_back(loopCount);
        } else {
          dimToLoopCount[loopDimName].push_back(loopCount);
        }
      }
    }
    // if (pcfgType == SenComponents::L3SU || pcfgType == SenComponents::L3LU ||
    //     op->name == OpFuncs::ResizeNNLX || op->name == OpFuncs::ResizeNNHBM)

    if (op->name == OpFuncs::ResizeNNLX || op->name == OpFuncs::ResizeNNHBM)
      mlCf = 1;

    std::map<std::string, std::string> dimToLoopName;
    for (int lpIdx = loopOrder.size() - 1; lpIdx >= (mlCf > 1 ? mlCf : 0);
         lpIdx--) {
      const auto& loopDimName = loopOrder[lpIdx];
      SenPcfgMvloopNode* newInnerMvLoopNode =
          (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);
      dimToLoopName[loopDimName] =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-IL-" + loopDimName +
          "-" + std::to_string(idx) + "-" + std::to_string(op->uniqueID);
      newInnerMvLoopNode->name = dimToLoopName[loopDimName];
      newInnerMvLoopNode->loopName = dimToLoopName[loopDimName];
      DT_CHECK(dimToLoopCount.count(loopDimName));
      DT_CHECK(dimToLoopCount.at(loopDimName).size());
      newInnerMvLoopNode->loopCount = dimToLoopCount.at(loopDimName)[0];

      // check if dynLoopCondAndVal is required
      if (dimToLoopCount.at(loopDimName).size() > 1) {
        if (lpIdx == 0) has_innermost_dyn_loop = true;
        // if (true) {  // to force enable dynmvloop
        // add dynLoopCondAndVal for each outer-loop-idx
        for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
          int inpSPIDx = dtKeys.at(idx3);
          int outSPIDx;
          if (pcfgType == SenComponents::L3LU) {
            outSPIDx = getIdxForMatchingCMenID(
                op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
          } else {
            DT_CHECK(op->inpSP_.size() > inpSPIDx);
          }
          std::pair<PcfgLccrCond, int> newPcfgLccrCond;
          newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
          newPcfgLccrCond.first.condOp = CondOp::EQ;
          newPcfgLccrCond.first.condVal =
              dtKeys.size() - 1 - idx3;  // order matters
          int dynLoopCount =
              pcfgType == SenComponents::L3LU
                  ? op->outSP_[outSPIDx].dimToSize_.at(loopDimName)
                  : op->inpSP_[inpSPIDx].dimToSize_.at(loopDimName);
          // stick-level adjustments
          LdsInfo* lds =
              (pcfgType == SenComponents::L3LU) ? op->outLds : op->inpLds;
          if (DCGUtils::isValPresent(lds->stickDimOrder_, loopDimName)) {
            dynLoopCount =
                ceil(dynLoopCount / lds->dimToStickSize_[loopDimName]);
          }
          newPcfgLccrCond.second = dynLoopCount;  // count
          newInnerMvLoopNode->dynLoopCondAndVal.push_back(newPcfgLccrCond);
          if (idx3 == dtKeys.size() - 1) {
            // push default case
            newPcfgLccrCond.first.loopName = newInnerMvLoopNode->name;
            newPcfgLccrCond.first.condVal = -1;
            newInnerMvLoopNode->dynLoopCondAndVal.push_back(newPcfgLccrCond);

            // set default values
            newInnerMvLoopNode->loopCount = dynLoopCount;
          }
        }
      }

      // inner loop end
      SenPcfgMvloopBranchNode* newInnerLoopBranchNode =
          (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
              SenPcfgNode::Type::MVLOOPBRANCH);

      newInnerLoopBranchNode->name =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-ILBranch-" +
          loopDimName + "-" + std::to_string(idx) + "-" +
          std::to_string(op->uniqueID);
      newInnerLoopBranchNode->loopNode = newInnerMvLoopNode;
      newInnerLoopBranchNode->next.push_back(newInnerMvLoopNode);

      // hook in NodeGraph
      newInnerMvLoopNode->prev.push_back(lastTopNodeInner);
      lastTopNodeInner->next.push_back(newInnerMvLoopNode);
      firstBotNodeInner->prev.push_back(newInnerLoopBranchNode);
      newInnerLoopBranchNode->next.push_back(firstBotNodeInner);

      lastTopNodeInner = newInnerMvLoopNode;
      firstBotNodeInner = newInnerLoopBranchNode;
    }

    std::string mergedDimName;
    if (mlCf > 1) {
      int collapsedLC_0 = 1;
      bool req_dyn_loop = false;
      for (int lpIdx = mlCf - 1; lpIdx >= 0; lpIdx--) {
        const auto& loopDimName = loopOrder[lpIdx];
        mergedDimName += loopDimName;
        DT_CHECK(dimToLoopCount.count(loopDimName));
        DT_CHECK(dimToLoopCount.at(loopDimName).size());
        // DT_CHECK(dimToLoopCount.at(loopDimName).size() == 1); // relaxed, we
        // will make use of dynamic loop
        if (dimToLoopCount.at(loopDimName).size() != 1) req_dyn_loop = true;
        collapsedLC_0 *= dimToLoopCount.at(loopDimName).at(0);
      }

      SenPcfgMvloopNode* newInnerMvLoopNode =
          (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);
      dimToLoopName[mergedDimName] =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-IL-" + mergedDimName +
          "-OL-" + std::to_string(idx) + "-" + std::to_string(op->uniqueID);
      newInnerMvLoopNode->name = dimToLoopName[mergedDimName];
      newInnerMvLoopNode->loopName = dimToLoopName[mergedDimName];
      newInnerMvLoopNode->loopCount = collapsedLC_0;

      // inner loop end
      SenPcfgMvloopBranchNode* newInnerLoopBranchNode =
          (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
              SenPcfgNode::Type::MVLOOPBRANCH);

      newInnerLoopBranchNode->name =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-ILBranch-" +
          mergedDimName + "-OL-" + std::to_string(idx) + "-" +
          std::to_string(op->uniqueID);
      newInnerLoopBranchNode->loopNode = newInnerMvLoopNode;
      newInnerLoopBranchNode->next.push_back(newInnerMvLoopNode);

      // hook in NodeGraph
      newInnerMvLoopNode->prev.push_back(lastTopNodeInner);
      lastTopNodeInner->next.push_back(newInnerMvLoopNode);
      firstBotNodeInner->prev.push_back(newInnerLoopBranchNode);
      newInnerLoopBranchNode->next.push_back(firstBotNodeInner);

      lastTopNodeInner = newInnerMvLoopNode;
      firstBotNodeInner = newInnerLoopBranchNode;

      // check if dynLoopCondAndVal is required
      if (req_dyn_loop) {
        has_innermost_dyn_loop = true;
        for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
          int dynLoopCount = 1;
          for (int lpIdx = mlCf - 1; lpIdx >= 0; lpIdx--) {
            const auto& loopDimName = loopOrder[lpIdx];

            int inpSPIDx = dtKeys.at(idx3);
            int outSPIDx;
            if (pcfgType == SenComponents::L3LU) {
              outSPIDx = getIdxForMatchingCMenID(
                  op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
            } else {
              DT_CHECK(op->inpSP_.size() > inpSPIDx);
            }
            int lp = pcfgType == SenComponents::L3LU
                         ? op->outSP_[outSPIDx].dimToSize_.at(loopDimName)
                         : op->inpSP_[inpSPIDx].dimToSize_.at(loopDimName);
            // stick-level adjustments
            LdsInfo* lds =
                (pcfgType == SenComponents::L3LU) ? op->outLds : op->inpLds;
            if (DCGUtils::isValPresent(lds->stickDimOrder_, loopDimName)) {
              lp = ceil(lp / lds->dimToStickSize_[loopDimName]);
            }
            dynLoopCount *= lp;
          }

          std::pair<PcfgLccrCond, int> newPcfgLccrCond;
          newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
          newPcfgLccrCond.first.condOp = CondOp::EQ;
          newPcfgLccrCond.first.condVal =
              dtKeys.size() - 1 - idx3;  // order matters

          newPcfgLccrCond.second = dynLoopCount;  // count
          newInnerMvLoopNode->dynLoopCondAndVal.emplace_back(newPcfgLccrCond);

          if (idx3 == dtKeys.size() - 1) {
            // push default case
            newPcfgLccrCond.first.loopName = newInnerMvLoopNode->name;
            newPcfgLccrCond.first.condVal = -1;
            newInnerMvLoopNode->dynLoopCondAndVal.emplace_back(newPcfgLccrCond);

            // set default values
            newInnerMvLoopNode->loopCount = dynLoopCount;
          }
        }
      }
    }

    // take care of ResizeNNHBM op
    if ((op->name == OpFuncs::ResizeNNHBM || op->name == OpFuncs::ResizeNNLX)) {
      baseSTCDPOp* myOpResize = (baseSTCDPOp*)op;
      std::map<std::string, int> upSizeFactor;
      int dimReqUpsize = 0;
      if (op->name == OpFuncs::ResizeNNHBM) {
        ResizeNNHBM* myOptemp = (ResizeNNHBM*)op;
        upSizeFactor = myOptemp->upSizeFactor;
      } else {
        ResizeNNLX* myOptemp = (ResizeNNLX*)op;
        upSizeFactor = myOptemp->upSizeFactor;
      }
      int totalUpSizeFactor = 1;
      for (auto& mapkv : upSizeFactor) {
        if (mapkv.second > 1) dimReqUpsize++;
        totalUpSizeFactor *= mapkv.second;
      }
      // DT_CHECK(dimReqUpsize >=
      //        1); /* atleast one dim should have upsizefactor > 1*/

      if (pcfgType == SenComponents::LXLU0) {
        SenPcfgMvloopNode* newMvLoopNode =
            (SenPcfgMvloopNode*)newPcfg.createPcfgNode(
                SenPcfgNode::Type::MVLOOP);
        std::string resizeLoopName =
            "c" + std::to_string(coreID) + "-" +
            newPcfg.senComponentsToString.at(pcfgType) + "-IL-resizeCollapsed" +
            std::to_string(idx) + "-" + "-" + std::to_string(op->uniqueID);
        newMvLoopNode->name = resizeLoopName;
        newMvLoopNode->loopName = resizeLoopName;
        newMvLoopNode->loopCount = totalUpSizeFactor;

        // inner loop end
        SenPcfgMvloopBranchNode* newLoopBranchNode =
            (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
                SenPcfgNode::Type::MVLOOPBRANCH);

        newLoopBranchNode->name = "c" + std::to_string(coreID) + "-" +
                                  newPcfg.senComponentsToString.at(pcfgType) +
                                  "-ILBranch-resizeCollapsed" + "-" +
                                  std::to_string(idx) + "-" +
                                  std::to_string(op->uniqueID);
        newLoopBranchNode->loopNode = newMvLoopNode;
        newLoopBranchNode->next.push_back(newMvLoopNode);

        // hook in NodeGraph
        newMvLoopNode->prev.push_back(lastTopNodeInner);
        lastTopNodeInner->next.push_back(newMvLoopNode);
        firstBotNodeInner->prev.push_back(newLoopBranchNode);
        newLoopBranchNode->next.push_back(firstBotNodeInner);

        lastTopNodeInner = newMvLoopNode;
        firstBotNodeInner = newLoopBranchNode;
      } else if (pcfgType == SenComponents::LXSU0) {
        for (int idx_r = loopOrder.size() - 1; idx_r >= 0; idx_r--) {
          const auto& loopDimName = loopOrder[idx_r];
          DT_CHECK(upSizeFactor.count(loopDimName));

          if (upSizeFactor.at(loopDimName) == 1) continue;

          SenPcfgMvloopNode* newMvLoopNode =
              (SenPcfgMvloopNode*)newPcfg.createPcfgNode(
                  SenPcfgNode::Type::MVLOOP);
          std::string tag = loopDimName;
          std::string resizeLoopName =
              "c" + std::to_string(coreID) + "-" +
              newPcfg.senComponentsToString.at(pcfgType) + "-IL-resizeL-" +
              tag + "-" + std::to_string(idx) + "-" +
              std::to_string(op->uniqueID);
          newMvLoopNode->name = resizeLoopName;
          newMvLoopNode->loopName = resizeLoopName;
          newMvLoopNode->loopCount = upSizeFactor.at(loopDimName);

          // inner loop end
          SenPcfgMvloopBranchNode* newLoopBranchNode =
              (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
                  SenPcfgNode::Type::MVLOOPBRANCH);

          newLoopBranchNode->name = "c" + std::to_string(coreID) + "-" +
                                    newPcfg.senComponentsToString.at(pcfgType) +
                                    "-ILBranch-resize-" + tag + "-" +
                                    std::to_string(idx) + "-" +
                                    std::to_string(op->uniqueID);
          newLoopBranchNode->loopNode = newMvLoopNode;
          newLoopBranchNode->next.push_back(newMvLoopNode);

          // hook in NodeGraph
          newMvLoopNode->prev.push_back(lastTopNodeInner);
          lastTopNodeInner->next.push_back(newMvLoopNode);
          firstBotNodeInner->prev.push_back(newLoopBranchNode);
          newLoopBranchNode->next.push_back(firstBotNodeInner);

          lastTopNodeInner = newMvLoopNode;
          firstBotNodeInner = newLoopBranchNode;
        }
      }
    }
    // Data Transfer
    // figure out bigDimToSize
    std::map<std::string, double> bigDimToSize_;
    if (pcfgType == SenComponents::L3LU || pcfgType == SenComponents::LXSU0) {
      int outSPIDx = getIdxForMatchingCMenID(
          op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
      bigDimToSize_ = op->outSP_[outSPIDx].bigDimToSize_;
    } else {
      bigDimToSize_ = op->inpSP_.at(dtKeys.at(0)).bigDimToSize_;
    }

    // check if all biDimToSize are same..
    for (int idx3 = 1; idx3 < dtKeys.size(); idx3++) {
      int inpSPIDx = dtKeys.at(idx3);
      int outSPIDx;
      if (pcfgType == SenComponents::L3LU || pcfgType == SenComponents::LXSU0) {
        outSPIDx = getIdxForMatchingCMenID(
            op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
        DT_CHECK(bigDimToSize_ == op->outSP_[outSPIDx].bigDimToSize_);
      } else {
        DT_CHECK(op->inpSP_.size() > inpSPIDx);
        if (!(op->name == OpFuncs::ResizeNNHBM ||
              op->name == OpFuncs::ResizeNNLX))
          DT_CHECK(bigDimToSize_ == op->inpSP_[inpSPIDx].bigDimToSize_);
      }
    }

    if (pcfgType == SenComponents::L3SU || pcfgType == SenComponents::L3LU) {
      if (op->name == OpFuncs::ResizeNNLX || op->name == OpFuncs::ResizeNNHBM)
        DT_CHECK(mlCf == 1);
      SenPcfgRingDtNode* newRingDtNode =
          (SenPcfgRingDtNode*)newPcfg.createPcfgNode(
              SenPcfgNode::Type::RINGDATATRANSFER);
      DtPair newDtPair;
      if (pcfgType == SenComponents::L3SU) {
        // RING transfer: send to ring
        newDtPair.src_ = SenComponents::LX;
        newDtPair.dst_ = SenComponents::RING;
      } else {
        // RING transfer: send to LX
        newDtPair.src_ = SenComponents::RING;
        newDtPair.dst_ = SenComponents::LX;
      }

      newRingDtNode->name =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-ringDT-" +
          newPcfg.senComponentsToString.at(newDtPair.src_) + "-" +
          newPcfg.senComponentsToString.at(newDtPair.dst_) + "-" +
          std::to_string(idx) + "-" + std::to_string(op->uniqueID);
      newRingDtNode->coreletId = -1;  // corelet independent
      newRingDtNode->dtInfo = nullptr;
      newRingDtNode->srcDest = newDtPair;
      newRingDtNode->dsInfo = nullptr;
      newRingDtNode->dimLayoutOrder = (pcfgType == SenComponents::L3SU)
                                          ? op->inpLds->layoutDimOrder_
                                          : op->outLds->layoutDimOrder_;
      newRingDtNode->myBigDimSize = bigDimToSize_;
      makeStickLevelAdjustments(
          newRingDtNode->myBigDimSize,
          (pcfgType == SenComponents::L3LU) ? op->outLds : op->inpLds);

      // handling of collapsed loops
      if (mlCf > 1) {
        newRingDtNode->myBigDimSize[mergedDimName] = 1;
        for (int lpIdx = mlCf - 1; lpIdx >= 0; lpIdx--) {
          const auto& loopDimName = loopOrder[lpIdx];
          newRingDtNode->myBigDimSize.at(mergedDimName) *=
              newRingDtNode->myBigDimSize.at(loopDimName);
          newRingDtNode->myBigDimSize.erase(loopDimName);
        }
        std::vector<std::string> newdimLayout;
        newdimLayout.push_back(mergedDimName);

        for (int dimIdx = mlCf; dimIdx < newRingDtNode->dimLayoutOrder.size();
             dimIdx++)
          newdimLayout.push_back(newRingDtNode->dimLayoutOrder.at(dimIdx));

        newRingDtNode->dimLayoutOrder = newdimLayout;
      }

      // add coreIds
      bool useUnicast = false;
      if (op->name == OpFuncs::STCDPOpLx)
        useUnicast = ((STCDPOpLx*)op)->useUnicast;

      bool canUseBurst = true;
      for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
        std::vector<int64_t> coreIdForRing;
        int inpSPIDx = dtKeys.at(idx3);
        if (pcfgType == SenComponents::L3LU) {
          // procuderID
          coreIdForRing.push_back(
              op->inpSP_.at(inpSPIDx).placement.MemId().getData().at(0));
        } else {
          // consumerID
          for (const auto& cIDX : op->dtTable_.at(inpSPIDx).cIDXs) {
            if (op->outSP_.at(cIDX).placement.MemId().getData().at(0) !=
                op->inpSP_.at(inpSPIDx).placement.MemId().getData().at(0))
              coreIdForRing.push_back(
                  op->outSP_.at(cIDX).placement.MemId().getData().at(0));
          }
        }

        std::pair<PcfgLccrCond, std::vector<int64_t>> newPcfgLccrCond;
        newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
        newPcfgLccrCond.first.condOp = CondOp::EQ;
        newPcfgLccrCond.first.condVal =
            dtKeys.size() - 1 - idx3;            // order matters
        newPcfgLccrCond.second = coreIdForRing;  // coreIDs

        newRingDtNode->coreIDForRingCondAndVal.push_back(newPcfgLccrCond);
        if (idx3 == dtKeys.size() - 1) {
          // set default values
          newRingDtNode->coreIdForRing = coreIdForRing;
        }

        if (!op->dtTable_.at(inpSPIDx).useBurst) canUseBurst = false;

        // fill GTR info
        if (useUnicast == false) {
          GTRBurstInfo newCondGTR;
          newCondGTR.groupID = op->dtTable_.at(inpSPIDx).myGTR.groupID;
          newCondGTR.numSharers = op->dtTable_.at(inpSPIDx).myGTR.numSharers;
          newCondGTR.count = op->dtTable_.at(inpSPIDx).myGTR.count;
          newCondGTR.srcNodeID = op->dtTable_.at(inpSPIDx).myGTR.srcNodeID;
          newCondGTR.useBurst = op->dtTable_.at(inpSPIDx).useBurst;
          newRingDtNode->GTRAndBurstCondAndVal.push_back(
              std::make_pair(newPcfgLccrCond.first, newCondGTR));

          newRingDtNode->gtr_imm_opt_en = false;
        }
      }

      if (useUnicast) newRingDtNode->useBurst = canUseBurst;

      // set start Address
      if (pcfgType == SenComponents::L3SU) {
        fillAddr(newRingDtNode->SrcStartAddr(),
                 op->inpSP_.at(dtKeys.at(0)).placement.StartAddr(), 0);
      } else {
        int outSPIDx = getIdxForMatchingCMenID(
            op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
        fillAddr(newRingDtNode->DestStartAddr(),
                 op->outSP_[outSPIDx].placement.StartAddr(), 0);
      }

      for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
        int inpSPIDx = dtKeys.at(idx3);
        int outSPIDx;
        if (pcfgType == SenComponents::L3LU) {
          outSPIDx = getIdxForMatchingCMenID(
              op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
          DT_CHECK(outSPIDx >= 0);
          DT_CHECK(op->outSP_.size() > outSPIDx);
        } else {
          DT_CHECK(op->inpSP_.size() > inpSPIDx);
        }
        std::pair<PcfgLccrCond, FoldManager<int64_t>> newPcfgLccrCond;
        newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
        newPcfgLccrCond.first.condOp = CondOp::EQ;
        newPcfgLccrCond.first.condVal =
            dtKeys.size() - 1 - idx3;  // order matters

        fillAddr(newPcfgLccrCond.second,
                 pcfgType == SenComponents::L3LU
                     ? op->outSP_[outSPIDx].placement.StartAddr()
                     : op->inpSP_[inpSPIDx].placement.StartAddr(),
                 0);  // count

        if (pcfgType == SenComponents::L3LU) {
          newRingDtNode->DestStartCondAndVal().emplace_back(newPcfgLccrCond);
        } else {
          newRingDtNode->SrcStartCondAndVal().emplace_back(newPcfgLccrCond);
        }
      }

      // no burst for now, add later..
      for (auto& myDim : newRingDtNode->myBigDimSize) {
        newRingDtNode->myLitDimSize[myDim.first] = 1;
      }

      // set bigStAddrOffsets for each inner loop
      // for (int lpIdx = loopOrder.size() - 1; lpIdx >= 0; lpIdx--) {
      for (int lpIdx = loopOrder.size() - 1; lpIdx >= (mlCf > 1 ? mlCf : 0);
           lpIdx--) {
        const auto& loopDimName = loopOrder[lpIdx];
        PcfgDtOffsets newDtOffset;
        init(newDtOffset);

        // find location in dimLayoutOrder
        for (auto& dimName : newRingDtNode->dimLayoutOrder) {
          if (dimName == loopDimName) {
            break;
          } else {
            newDtOffset.dimOffset *= newRingDtNode->myBigDimSize[dimName];
          }
        }

        newRingDtNode->bigStAddrOffsets[dimToLoopName[loopDimName]] =
            newDtOffset;
      }

      // inner most loop offset
      if (mlCf > 1) {
        PcfgDtOffsets newDtOffset;
        init(newDtOffset);
        // find location in dimLayoutOrder
        for (auto& dimName : newRingDtNode->dimLayoutOrder) {
          if (dimName == mergedDimName) {
            break;
          } else {
            newDtOffset.dimOffset *= newRingDtNode->myBigDimSize[dimName];
          }
        }
        DT_CHECK(dimToLoopName.count(mergedDimName));
        newRingDtNode->bigStAddrOffsets[dimToLoopName.at(mergedDimName)] =
            newDtOffset;
      }

      // fill multicast mode info
      if (op->name == OpFuncs::STCDPOpLx)
        newRingDtNode->forceModeMC = ((STCDPOpLx*)op)->forceModeMC;

      // hook in NodeGraph
      lastTopNodeInner->next.push_back(newRingDtNode);
      newRingDtNode->prev.push_back(lastTopNodeInner);
      lastTopNodeInner = newRingDtNode;

      // check if last loop is dynamic, if so disable burst
      if (has_innermost_dyn_loop && newRingDtNode->useBurst)
        DT_CHECK(0);  // unexpected

    } else if (pcfgType == SenComponents::LXSU0 ||
               pcfgType == SenComponents::LXLU0) {
      SenPcfgDtNode* newDtNode = (SenPcfgDtNode*)newPcfg.createPcfgNode(
          SenPcfgNode::Type::DATATRANSFER);
      DtPair newDtPair;
      if (pcfgType == SenComponents::LXSU0) {
        // LXLUSUFIFO transfer: to LX
        newDtPair.src_ = op->useLXSFPLXTransfers ? SenComponents::PE0
                                                 : SenComponents::LXLUSUFIFO;
        newDtPair.dst_ = SenComponents::LX;
      } else {
        // LXLUSUFIFO transfer: from LX
        newDtPair.src_ = SenComponents::LX;
        newDtPair.dst_ = op->useLXSFPLXTransfers ? SenComponents::PE0
                                                 : SenComponents::LXLUSUFIFO;
      }

      newDtNode->name =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-ringDT-" +
          newPcfg.senComponentsToString.at(newDtPair.src_) + "-" +
          newPcfg.senComponentsToString.at(newDtPair.dst_) + "-" +
          std::to_string(idx) + "-" + std::to_string(op->uniqueID);
      newDtNode->coreletId = -1;  // corelet independent
      newDtNode->dtInfo = nullptr;
      newDtNode->srcDest = newDtPair;
      newDtNode->dsInfo = nullptr;
      newDtNode->dimLayoutOrder = (pcfgType == SenComponents::LXLU0)
                                      ? op->inpLds->layoutDimOrder_
                                      : op->outLds->layoutDimOrder_;
      newDtNode->myBigDimSize = bigDimToSize_;
      makeStickLevelAdjustments(
          newDtNode->myBigDimSize,
          (pcfgType == SenComponents::LXLU0) ? op->outLds : op->inpLds);

      // handling of collapsed loops
      if (mlCf > 1) {
        newDtNode->myBigDimSize[mergedDimName] = 1;
        for (int lpIdx = mlCf - 1; lpIdx >= 0; lpIdx--) {
          const auto& loopDimName = loopOrder[lpIdx];
          newDtNode->myBigDimSize.at(mergedDimName) *=
              newDtNode->myBigDimSize.at(loopDimName);
          newDtNode->myBigDimSize.erase(loopDimName);
        }
        std::vector<std::string> newdimLayout;
        newdimLayout.push_back(mergedDimName);

        for (int dimIdx = mlCf; dimIdx < newDtNode->dimLayoutOrder.size();
             dimIdx++)
          newdimLayout.push_back(newDtNode->dimLayoutOrder.at(dimIdx));

        newDtNode->dimLayoutOrder = newdimLayout;
      }

      // set start Address
      if (pcfgType == SenComponents::LXLU0) {
        fillAddr(newDtNode->SrcStartAddr(),
                 op->inpSP_.at(dtKeys.at(0)).placement.StartAddr(), 0);
      } else {
        int outSPIDx = getIdxForMatchingCMenID(
            op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
        fillAddr(newDtNode->DestStartAddr(),
                 op->outSP_[outSPIDx].placement.StartAddr(), 0);
      }

      // add burst Info, used in SenProg
      newDtNode->useBurst = true;
      if (op->name == OpFuncs::ResizeNNLX || op->name == OpFuncs::ResizeNNHBM) {
        if (pcfgType == SenComponents::LXSU0 ||
            pcfgType == SenComponents::LXLU0) {
          newDtNode->useBurst = false;
        } else {
          DT_CHECK(0);  // this case shouldn't happen
        }
      }

      for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
        int inpSPIDx = dtKeys.at(idx3);
        int outSPIDx;
        if (pcfgType == SenComponents::LXSU0) {
          outSPIDx = getIdxForMatchingCMenID(
              op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
          DT_CHECK(outSPIDx >= 0);
          DT_CHECK(op->outSP_.size() > outSPIDx);
        } else {
          DT_CHECK(op->inpSP_.size() > inpSPIDx);
        }

        if (dtKeys.size() > 1) {
          std::pair<PcfgLccrCond, FoldManager<int64_t>> newPcfgLccrCond;
          newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
          newPcfgLccrCond.first.condOp = CondOp::EQ;
          newPcfgLccrCond.first.condVal =
              dtKeys.size() - 1 - idx3;  // order matters
          fillAddr(newPcfgLccrCond.second,
                   pcfgType == SenComponents::LXSU0
                       ? op->outSP_[outSPIDx].placement.StartAddr()
                       : op->inpSP_[inpSPIDx].placement.StartAddr(),
                   0);  // count
          if (pcfgType == SenComponents::LXSU0) {
            newDtNode->DestStartCondAndVal().emplace_back(newPcfgLccrCond);
          } else {
            newDtNode->SrcStartCondAndVal().emplace_back(newPcfgLccrCond);
          }
        }

        if (!op->dtTable_.at(inpSPIDx).useBurst) newDtNode->useBurst = false;
      }

      // no burst for now, add later..
      for (auto& myDim : newDtNode->myBigDimSize) {
        newDtNode->myLitDimSize[myDim.first] = 1;
      }

      // set bigStAddrOffsets for each inner loop
      for (int lpIdx = loopOrder.size() - 1; lpIdx >= (mlCf > 1 ? mlCf : 0);
           lpIdx--) {
        const auto& loopDimName = loopOrder[lpIdx];
        PcfgDtOffsets newDtOffset;
        init(newDtOffset);

        // find location in dimLayoutOrder
        for (auto& dimName : newDtNode->dimLayoutOrder) {
          if (dimName == loopDimName) {
            break;
          } else {
            newDtOffset.dimOffset *= newDtNode->myBigDimSize[dimName];
          }
        }

        // special case: ResizeNNHBM Op: Insert another conditional
        if (pcfgType == SenComponents::LXSU0 &&
            (op->name == OpFuncs::ResizeNNHBM ||
             op->name == OpFuncs::ResizeNNLX)) {
          DT_CHECK(mlCf == 1);
          std::map<std::string, int> upSizeFactor;
          if (op->name == OpFuncs::ResizeNNHBM) {
            ResizeNNHBM* myOptemp = (ResizeNNHBM*)op;
            upSizeFactor = myOptemp->upSizeFactor;
          } else {
            ResizeNNLX* myOptemp = (ResizeNNLX*)op;
            upSizeFactor = myOptemp->upSizeFactor;
          }

          baseSTCDPOp* myOpResize = (baseSTCDPOp*)op;
          const auto& loopDimName = loopOrder[lpIdx];
          DT_CHECK(upSizeFactor.count(loopDimName));
          std::string tag = loopDimName;
          if (upSizeFactor.at(loopDimName) > 1) {
            std::string resizeLoopName =
                "c" + std::to_string(coreID) + "-" +
                newPcfg.senComponentsToString.at(pcfgType) + "-IL-resizeL-" +
                tag + "-" + std::to_string(idx) + "-" +
                std::to_string(op->uniqueID);
            newDtNode->bigStAddrOffsets[resizeLoopName] = newDtOffset;
            // modify offset for regular case
            newDtOffset.dimOffset *= upSizeFactor.at(loopDimName);
          }
        }

        newDtNode->bigStAddrOffsets[dimToLoopName[loopDimName]] = newDtOffset;
      }

      // inner most loop offset
      if (mlCf > 1) {
        PcfgDtOffsets newDtOffset;
        init(newDtOffset);
        // find location in dimLayoutOrder
        for (auto& dimName : newDtNode->dimLayoutOrder) {
          if (dimName == mergedDimName) {
            break;
          } else {
            newDtOffset.dimOffset *= newDtNode->myBigDimSize[dimName];
          }
        }
        DT_CHECK(dimToLoopName.count(mergedDimName));
        newDtNode->bigStAddrOffsets[dimToLoopName.at(mergedDimName)] =
            newDtOffset;
      }

      // hook in NodeGraph
      lastTopNodeInner->next.push_back(newDtNode);
      newDtNode->prev.push_back(lastTopNodeInner);
      lastTopNodeInner = newDtNode;

      // check if last loop is dynamic, if so disable burst
      if (has_innermost_dyn_loop) newDtNode->useBurst = false;
    } else {
      DT_CHECK(0);
    }

    // hook in NodeGraph
    lastTopNodeInner->next.push_back(firstBotNodeInner);
    firstBotNodeInner->prev.push_back(lastTopNodeInner);

    lastTopNodePcfg = lastNodethisLoop;
  }
}

std::vector<std::pair<std::vector<std::string>, std::vector<int>>>
DcgFE::clusterDtKeysUsingInnerLOs(std::vector<int>& dtKeys, baseSTCDPOp* op,
                                  int coreId, bool useInpSubPiece) {
  std::vector<std::pair<std::vector<std::string>, std::vector<int>>>
      dtKeysPerInnerLOs;

  bool skip_dimSize_check = op->name == OpFuncs::STCDPOpHBM ? true : false;

  for (int idx = 0; idx < dtKeys.size(); idx++) {
    auto& myDtKey = dtKeys.at(idx);
    bool canMerge = false;
    if (dtKeysPerInnerLOs.size()) {
      if (dtKeysPerInnerLOs.back().first ==
          op->dtTable_.at(myDtKey).loopOrder) {
        // check if piece dimensions are same
        DT_CHECK(dtKeysPerInnerLOs.back().second.size());
        int inpSPIDx = dtKeysPerInnerLOs.back().second.back();

        // if (op->inpSP_[inpSPIDx].dimToSize_ ==
        // op->inpSP_[myDtKey].dimToSize_) {
        if (useInpSubPiece) {
          if (op->inpSP_[inpSPIDx].bigDimToSize_ ==
                  op->inpSP_[myDtKey].bigDimToSize_ &&
              op->dtTable_.at(inpSPIDx).collapseFactor ==
                  op->dtTable_.at(myDtKey).collapseFactor &&
              (skip_dimSize_check || op->inpSP_[inpSPIDx].dimToSize_ ==
                                         op->inpSP_[myDtKey].dimToSize_))
            canMerge = true;
        } else {
          int outSPIdx = getIdxForMatchingCMenID(
              op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreId);
          int out_dtkey = getIdxForMatchingCMenID(
              op->outSP_, op->dtTable_.at(myDtKey).cIDXs, coreId);
          if (op->outSP_[outSPIdx].bigDimToSize_ ==
                  op->outSP_[out_dtkey].bigDimToSize_ &&
              op->dtTable_.at(inpSPIDx).collapseFactor ==
                  op->dtTable_.at(myDtKey).collapseFactor &&
              (skip_dimSize_check || op->outSP_[outSPIdx].dimToSize_ ==
                                         op->outSP_[out_dtkey].dimToSize_))
            canMerge = true;
        }
        //}
      }
    }

    if (canMerge) {
      dtKeysPerInnerLOs.back().second.push_back(myDtKey);
    } else {
      // create new entry
      dtKeysPerInnerLOs.resize(dtKeysPerInnerLOs.size() + 1);
      dtKeysPerInnerLOs.back().first = op->dtTable_.at(myDtKey).loopOrder;
      dtKeysPerInnerLOs.back().second.push_back(myDtKey);
    }
  }
  return dtKeysPerInnerLOs;
}

void DcgFE::computerGTRInfo(baseSTCDPOp* op, int ddsc_idx /*= 0 */) {
  int grpID = 0;

  for (auto& dtEntry : op->dtTable_) {
    int pIDX = dtEntry.second.pIDX;

    auto memId_v_isp = op->inpSP_.at(pIDX).placement.MemId().getData();
    DT_CHECK(memId_v_isp.size() == 1);
    dtEntry.second.myGTR.srcNodeID = memId_v_isp.at(0);
    dtEntry.second.myGTR.count = 0;
    dtEntry.second.myGTR.numSharers = 0;
    for (const auto& cIDX : dtEntry.second.cIDXs) {
      auto memId_v_osp = op->outSP_.at(cIDX).placement.MemId().getData();
      dtEntry.second.myGTR.numSharers += memId_v_osp.size();
      if (DCGUtils::isValPresent(memId_v_osp, memId_v_isp.at(0))) {
        dtEntry.second.myGTR.numSharers -= 1;  // remove self
      }
    }
    if (dtEntry.second.myGTR.numSharers > 0) {
      if (use_fixed_group_id_map) {
        DT_CHECK(datadsc_idx_to_group_id.count(ddsc_idx));
        dtEntry.second.myGTR.groupID = datadsc_idx_to_group_id.at(ddsc_idx);
      } else {
        dtEntry.second.myGTR.groupID =
            (firstAvailGlobalGrpId_ + grpID) % sysDef.maxGroupID;
        op->gtrIdsUsed.insert(dtEntry.second.myGTR.groupID);
        grpID++;
      }
    }
  }

  if (grpID <= sysDef.maxGroupID) {
    // do nothing
    // DCGUtils::message(",Passed,grpID-max," + std::to_string(grpID));
    firstAvailGlobalGrpId_ += grpID;
    firstAvailGlobalGrpId_ = firstAvailGlobalGrpId_ % sysDef.maxGroupID;
  } else {
    // optimize grpID :heuristic-1 : Each producer starts from 0 and assigns
    // uniqueID for each transfer
    op->gtrIdsUsed.clear();
    std::map<int, int> coreToGrpIDCount;

    // hbm multicast specific sharing
    std::vector<std::pair<std::set<int>, int>> hbmSharerToGroupID;

    for (auto& dtEntry : op->dtTable_) {
      int pIDX = dtEntry.second.pIDX;
      DT_CHECK(op->inpSP_[pIDX].placement.getMemId().size() == 1);
      int coreID = op->inpSP_[pIDX].placement.getMemId().at(0);

      // optimization for non analytical mode STCDP-HBM
      if (op->name == OpFuncs::STCDPOpHBM && coreID == -1) {
        // find a consumer core
        DT_CHECK(dtEntry.second.cIDXs.size());
        int outIdx = dtEntry.second.cIDXs.back();
        int consumer_coreId = op->outSP_.at(outIdx).placement.getMemId().back();
        DT_CHECK(((STCDPOpHBM*)op)->coreIDtoANInfo.count(consumer_coreId));
        if (!((STCDPOpHBM*)op)
                 ->coreIDtoANInfo.at(consumer_coreId)
                 .isAnalyticalMode &&
            ((STCDPOpHBM*)op)->reqMulticast) {
          // unicast transfer need not be allocated a groupId as we unroll
          // everything once this condition is achieved
          if (dtEntry.second.myGTR.numSharers == 1) continue;
        }
      }

      if (coreToGrpIDCount.count(coreID) == 0) coreToGrpIDCount[coreID] = 0;
      if (dtEntry.second.myGTR.numSharers > 0) {
        if (op->name == OpFuncs::STCDPOpHBM) {
          auto cMemIDs = getCMenIDs(op->outSP_, dtEntry.second.cIDXs);
          // see if the Consumer core group exists
          bool found_gtrEntry = false;
          int grpId_hbm = -1;
          for (auto& item : hbmSharerToGroupID)
            if (item.first == cMemIDs) {
              grpId_hbm = item.second;
              found_gtrEntry = true;
              break;
            }

          if (found_gtrEntry) {
            DT_CHECK(grpId_hbm >= 0);
            dtEntry.second.myGTR.groupID = grpId_hbm;
          } else {
            // assign new groupId
            dtEntry.second.myGTR.groupID =
                (firstAvailGlobalGrpId_ + coreToGrpIDCount.at(coreID)) %
                sysDef.maxGroupID;
            coreToGrpIDCount.at(coreID)++;
            op->gtrIdsUsed.insert(dtEntry.second.myGTR.groupID);
            hbmSharerToGroupID.push_back(
                std::make_pair(cMemIDs, dtEntry.second.myGTR.groupID));
          }

        } else {
          dtEntry.second.myGTR.groupID =
              (firstAvailGlobalGrpId_ + coreToGrpIDCount.at(coreID)) %
              sysDef.maxGroupID;
          coreToGrpIDCount.at(coreID)++;
          op->gtrIdsUsed.insert(dtEntry.second.myGTR.groupID);
        }
      }
    }

    // check conditions
    int maxGrpId = 0;
    for (auto& mapkv : coreToGrpIDCount)
      maxGrpId = maxGrpId < mapkv.second ? mapkv.second : maxGrpId;

    firstAvailGlobalGrpId_ += maxGrpId;
    firstAvailGlobalGrpId_ = firstAvailGlobalGrpId_ % sysDef.maxGroupID;
    DT_CHECK(maxGrpId <= sysDef.maxGroupID);
    // if (maxGrpId <= maxGrpIDL3)
    //  DCGUtils::message(",Passed-WithOpt,grpID-max," +
    //  std::to_string(maxGrpId));
    // else
    //  DCGUtils::warning(",Failed,grpID-max, " +
    //  std::to_string(maxGrpId));
  }
}

void DcgFE::checkConvertToUnicast(STCDPOpLx* op) {
  bool useUnicast = true;
  for (auto& dtEntry : op->dtTable_)
    if (dtEntry.second.myGTR.numSharers > 1) useUnicast = false;
  op->useUnicast = useUnicast;
}

void DcgFE::setPlacementInfoSubPiece(const LdsInfo* ldsPtr, PieceInfo& myPiece,
                                     SliceInfo& subPiece,
                                     bool useLdsCoordinates /*= false*/) {
  DT_CHECK(myPiece.placement.count(SenComponents::LX));
  subPiece.placement.setType(myPiece.placement.at(SenComponents::LX).Type());

  subPiece.placement.MemId().clone(
      myPiece.placement.at(SenComponents::LX).MemId());
  subPiece.placement.StartAddr().clone(
      myPiece.placement.at(SenComponents::LX).StartAddr());

  // compute offset
  long offset = 0;  // in elements
  for (int idx = ldsPtr->layoutDimOrder_.size() - 1; idx >= 0; idx--) {
    std::string dimName = ldsPtr->layoutDimOrder_[idx];
    int diff;

    if (!useLdsCoordinates) {
      diff = subPiece.dimToStartCordinate.at(dimName) -
             myPiece.dimToStartCordinate.at(dimName);
    } else {
      DT_CHECK(ldsPtr->pieces_.count(myPiece.key_));
      diff = subPiece.dimToStartCordinate.at(dimName) -
             ldsPtr->pieces_.at(myPiece.key_).dimToStartCordinate.at(dimName);
    }

    if (DCGUtils::isValPresent(ldsPtr->stickDimOrder_, dimName)) {
      diff =
          ceil(diff / ldsPtr->dimToStickSize_.at(dimName));  // stick adjusted
    }

    auto dimToSize = PieceInfo::getTotalElementsInVG(myPiece.validGap_);
    // DT_CHECK( dimToSize == myPiece.dimToSize_);
    DT_CHECK(diff >= 0);
    if (diff > 0) {
      int eleToSkip = 1;  // in sticks
      for (int idx2 = idx - 1; idx2 >= 0; idx2--) {
        std::string dimName2 = ldsPtr->layoutDimOrder_[idx2];
        // adjust for stick
        if (DCGUtils::isValPresent(ldsPtr->stickDimOrder_, dimName2)) {
          // eleToSkip *= ceil(myPiece.dimToSize_[dimName2] /
          //                  ldsPtr->dimToStickSize_.at(dimName2));
          eleToSkip *= ceil(dimToSize.at(dimName2) /
                            ldsPtr->dimToStickSize_.at(dimName2));

        } else {
          // eleToSkip *= myPiece.dimToSize_[dimName2];
          eleToSkip *= dimToSize.at(dimName2);
        }
      }
      offset += diff * eleToSkip * sysDef.bytesPerStick;  // stick adjusted
    }
  }

  subPiece.placement.StartAddr().apply({}, [&](auto&& data) {
    for (auto& startAddr : data) startAddr += offset;
    return data;
  });
}

bool DcgFE::doesPiecesOverlap(STCDPOpLx* op, PieceInfo* inpPiece,
                              PieceInfo* outPiece) {
  for (auto key_ : outPiece->dimToStartCordinate) {
    // shifting the cartesian coordinates
    int outStartGap = 0;
    int outStart = outPiece->dimToStartCordinate.at(key_.first);
    if (op->outLds->validGap_.at(key_.first).size() == 2) {
      // outStartGap = outPiece->validGap_.at(key_.first)[0].second; //BUG-sj
      outStartGap = op->outLds->validGap_.at(key_.first)[0].second;
      if (outPiece->validGap_.at(key_.first).size() == 2)
        DT_CHECK(outPiece->validGap_.at(key_.first)[0].second == outStartGap);
    } else if (op->outLds->validGap_.at(key_.first).size() == 1) {
      if (outPiece->validGap_.at(key_.first).size() == 2)
        outStartGap = outPiece->validGap_.at(key_.first)[0].second;
    } else {
      DT_CHECK(0);
    }
    outStart -= outStartGap;

    int outEndGap = (outPiece->validGap_.at(key_.first).size() == 2)
                        ? outPiece->validGap_.at(key_.first)[1].second
                        : outPiece->validGap_.at(key_.first)[0].second;
    int outEnd = outStart + outPiece->dimToSize_.at(key_.first) - 1 - outEndGap;

    if (op->name == OpFuncs::ResizeNNLX) {
      auto op_resize = (ResizeNNLX*)op;
      if (op_resize->upSizeFactor.count(key_.first)) {
        int upsizeFactor = op_resize->upSizeFactor.at(key_.first);
        DT_CHECK(outStart % upsizeFactor == 0);
        outStart /= upsizeFactor;
        DT_CHECK((outEnd + 1) % upsizeFactor == 0);
        outEnd = ((outEnd + 1) / upsizeFactor) - 1;
      }
    }

    int inpStartGap = 0;
    int inpStart = inpPiece->dimToStartCordinate.at(key_.first);
    if (op->inpLds->validGap_.at(key_.first).size() == 2) {
      // inpStartGap = inpPiece->validGap_.at(key_.first)[0].second; //BUG-sj
      inpStartGap = op->inpLds->validGap_.at(key_.first)[0].second;
      if (inpPiece->validGap_.at(key_.first).size() == 2)
        DT_CHECK(inpPiece->validGap_.at(key_.first)[0].second == outStartGap);
    } else if (op->inpLds->validGap_.at(key_.first).size() == 1) {
      if (inpPiece->validGap_.at(key_.first).size() == 2)
        inpStartGap = inpPiece->validGap_.at(key_.first)[0].second;
    } else {
      DT_CHECK(0);
    }
    inpStart -= inpStartGap;

    int inpEndGap = (inpPiece->validGap_.at(key_.first).size() == 2)
                        ? inpPiece->validGap_.at(key_.first)[1].second
                        : inpPiece->validGap_.at(key_.first)[0].second;
    int inpEnd = inpStart + inpPiece->dimToSize_.at(key_.first) - 1 - inpEndGap;

    if (inpPiece->dimToStartCordinate.count(key_.first)) {
      if (outEnd < inpStart || inpEnd < outStart) {
        return false;
      }
    } else {
      // use DimRelationInfo
      return false;
    }
  }
  return true;
}

void DcgFE::insertSubPieces(STCDPOpLx* op, PieceInfo& inpPiece,
                            PieceInfo& outPiece) {
  // form subPiece
  SliceInfo inpSubPiece;
  SliceInfo outSubPiece;
  for (auto key_ : outPiece.dimToStartCordinate) {
    // shifting the cartesian coordinates
    int outStartGap = 0;
    int outStart = outPiece.dimToStartCordinate.at(key_.first);
    if (op->outLds->validGap_.at(key_.first).size() == 2) {
      // outStartGap = outPiece.validGap_.at(key_.first)[0].second; // BUG-sj
      outStartGap = op->outLds->validGap_.at(key_.first)[0].second;
      if (outPiece.validGap_.at(key_.first).size() == 2)
        DT_CHECK(outPiece.validGap_.at(key_.first)[0].second == outStartGap);
    } else if (op->outLds->validGap_.at(key_.first).size() == 1) {
      if (outPiece.validGap_.at(key_.first).size() == 2)
        outStartGap = outPiece.validGap_.at(key_.first)[0].second;
    } else {
      DT_CHECK(0);
    }
    outStart -= outStartGap;

    int outEndGap = (outPiece.validGap_.at(key_.first).size() == 2)
                        ? outPiece.validGap_.at(key_.first)[1].second
                        : outPiece.validGap_.at(key_.first)[0].second;

    int outEnd = outStart + outPiece.dimToSize_.at(key_.first) - 1 - outEndGap;
    int upsizeFactor = 1;
    if (op->name == OpFuncs::ResizeNNLX) {
      auto op_resize = (ResizeNNLX*)op;
      if (op_resize->upSizeFactor.count(key_.first)) {
        upsizeFactor = op_resize->upSizeFactor.at(key_.first);
        DT_CHECK(outStart % upsizeFactor == 0);
        outStart /= upsizeFactor;
        DT_CHECK((outEnd + 1) % upsizeFactor == 0);
        outEnd = ((outEnd + 1) / upsizeFactor) - 1;
      }
    }

    int inpStartGap = 0;
    int inpStart = inpPiece.dimToStartCordinate.at(key_.first);
    if (op->inpLds->validGap_.at(key_.first).size() == 2) {
      // inpStartGap = inpPiece.validGap_.at(key_.first)[0].second; //BUG-sj
      inpStartGap = op->inpLds->validGap_.at(key_.first)[0].second;
      if (inpPiece.validGap_.at(key_.first).size() == 2)
        DT_CHECK(inpPiece.validGap_.at(key_.first)[0].second == outStartGap);
    } else if (op->inpLds->validGap_.at(key_.first).size() == 1) {
      if (inpPiece.validGap_.at(key_.first).size() == 2)
        inpStartGap = inpPiece.validGap_.at(key_.first)[0].second;
    } else {
      DT_CHECK(0);
    }
    inpStart -= inpStartGap;

    int inpEndGap = (inpPiece.validGap_.at(key_.first).size() == 2)
                        ? inpPiece.validGap_.at(key_.first)[1].second
                        : inpPiece.validGap_.at(key_.first)[0].second;
    int inpEnd = inpStart + inpPiece.dimToSize_.at(key_.first) - 1 - inpEndGap;

    int startCoor = inpStart < outStart ? outStart : inpStart;
    if (inpStart <= outStart) {
      // re-adjust dimToStartCordinate
      inpSubPiece.dimToStartCordinate[key_.first] = outStart + inpStartGap;
      outSubPiece.dimToStartCordinate[key_.first] =
          outStart * upsizeFactor + outStartGap;
    } else {
      inpSubPiece.dimToStartCordinate[key_.first] = inpStart + inpStartGap;
      outSubPiece.dimToStartCordinate[key_.first] = inpStart + outStartGap;
    }

    if (outEnd <= inpEnd) {
      inpSubPiece.dimToSize_[key_.first] = outEnd - startCoor + 1;
      outSubPiece.dimToSize_[key_.first] =
          (outEnd - startCoor + 1) * upsizeFactor;
    } else {
      inpSubPiece.dimToSize_[key_.first] = inpEnd - startCoor + 1;
      outSubPiece.dimToSize_[key_.first] = inpEnd - startCoor + 1;
    }
  }

  inpSubPiece.bigDimToSize_ =
      PieceInfo::getTotalElementsInVG(inpPiece.validGap_);
  setPlacementInfoSubPiece(op->inpLds, inpPiece, inpSubPiece);
  DT_CHECK(inpSubPiece.placement.getMemId().size() >= 1);

  if (inpSubPiece.placement.getMemId().size() > 1) {
    auto all_cores = inpSubPiece.placement.MemId().getData();
    inpSubPiece.placement.MemId().insertData({all_cores.at(0)});

    // we have redundancy, pick the first one
    for (auto& [fcoord, data] :
         inpSubPiece.placement.StartAddr().getDataAndFoldCoordinates()) {
      DT_CHECK(data.size() == all_cores.size());
      inpSubPiece.placement.StartAddr().insertData({data.at(0)}, fcoord);
    }
  }

  int inpSubPieceIdx = -1;
  if (op->enSubPieceReuse)
    inpSubPieceIdx = getSubPieceIDX(op->inpSP_, inpSubPiece);

  outSubPiece.bigDimToSize_ =
      PieceInfo::getTotalElementsInVG(outPiece.validGap_);
  if (isInpFetchNeigh_)
    setPlacementInfoSubPiece(op->outLds, outPiece, outSubPiece, true);
  else
    setPlacementInfoSubPiece(op->outLds, outPiece, outSubPiece);

  // need to check if all consumers are compatible
  if (inpSubPieceIdx != -1) {
    auto numSticksOutDims = [&](std::map<std::string, double> dimSize,
                                std::string oddDim) {
      makeStickLevelAdjustments(dimSize, op->outLds);
      int numSticks = 1;
      for (int idx = op->outLds->layoutDimOrder_.size() - 1; idx >= 0; idx--) {
        if (oddDim == op->outLds->layoutDimOrder_.at(idx)) {
          break;
        }
        numSticks *= dimSize.at(op->outLds->layoutDimOrder_.at(idx));
      }
      return numSticks;
    };

    bool disbaleMerge = false;
    for (auto& priorOutSP : op->dtTable_.at(inpSubPieceIdx).cIDXs) {
      // check polarity
      int addr_div = sysDef.bytesPerStick;

      std::set<int> prior_porality;
      std::set<int> curr_porality;

      for (auto addr_vec :
           op->outSP_.at(priorOutSP).placement.StartAddr().getAllData()) {
        DT_CHECK(addr_vec.size() == 1);
        prior_porality.insert((int64_t)(addr_vec.back() / addr_div) % 2);
      }

      for (auto addr_vec : outSubPiece.placement.StartAddr().getAllData()) {
        for (auto addr : addr_vec) {
          curr_porality.insert((int64_t)(addr / addr_div) % 2);
        }
      }

      DT_CHECK(prior_porality.size() == 1);
      DT_CHECK(curr_porality.size() == 1);

      if (prior_porality != curr_porality) {
        disbaleMerge = true;
      } else if (outSubPiece.bigDimToSize_ !=
                 op->outSP_.at(priorOutSP).bigDimToSize_) {
        // check if bigDims are all even
        for (auto kv : outSubPiece.bigDimToSize_)
          if ((int64_t)kv.second % 2 != 0)
            if (numSticksOutDims(outSubPiece.dimToSize_, kv.first) > 1) {
              disbaleMerge = true;
              break;
            }
        if (!disbaleMerge) {
          for (auto kv : op->outSP_.at(priorOutSP).bigDimToSize_)
            if ((int64_t)kv.second % 2 != 0)
              if (numSticksOutDims(outSubPiece.dimToSize_, kv.first) > 1) {
                disbaleMerge = true;
                break;
              }
        }
      }
      if (disbaleMerge) {
        inpSubPieceIdx = -1;
        break;
      }
    }
  }

  // insert subpiece if it is not present..
  if (inpSubPieceIdx == -1) {
    // Relaxed for Input Fetch Neighbor, but should be true for STCDPLX
    // DT_CHECK( PieceInfo::getTotalElementsInVG(inpPiece.validGap_) ==
    // inpPiece.dimToSize_);
    op->inpSP_.push_back(inpSubPiece);
    inpSubPieceIdx = op->inpSP_.size() - 1;
    // add entry to the dtTable_
    DT_CHECK(op->dtTable_.count(inpSubPieceIdx) == 0);
    op->dtTable_[inpSubPieceIdx].pIDX = op->inpSP_.size() - 1;
    op->dtTable_[inpSubPieceIdx].pMemID =
        inpSubPiece.placement.getMemId().at(0);
  }

  if (getSubPieceIDX(op->outSP_, outSubPiece) == -1 || !op->enSubPieceReuse) {
    // Relaxed for Input Fetch Neighbor, but should be true for STCDPLX
    // DT_CHECK( PieceInfo::getTotalElementsInVG(outPiece.validGap_) ==
    // outPiece.dimToSize_);

    // Create separate entry for each placementInfo (memId)
    PlacementInfo placeInfoCopy = outSubPiece.placement;
    for (int idx = 0; idx < placeInfoCopy.getMemId().size(); idx++) {
      PlacementInfo newCopyPlaceInfo;
      newCopyPlaceInfo.setType(placeInfoCopy.Type());

      fillMemIdAddr(newCopyPlaceInfo.MemId(), placeInfoCopy.MemId(), idx);
      fillMemIdAddr(newCopyPlaceInfo.StartAddr(), placeInfoCopy.StartAddr(),
                    idx);

      outSubPiece.placement = newCopyPlaceInfo;
      DT_CHECK(outSubPiece.placement.getMemId().size() == 1);
      op->outSP_.emplace_back(outSubPiece);

      // input fetch neighbor tabulating
      if (isInpFetchNeigh_)
        // if (op->dtTable_[inpSubPieceIdx].pMemID !=
        //    outSubPiece.placement.getMemId().at(0))
        myIFNInfo_.dtTableIdxInWSlice.back().insert(inpSubPieceIdx);

      // add entry to the dtTable_
      op->dtTable_[inpSubPieceIdx].cIDXs.push_back(op->outSP_.size() - 1);
      if (op->dtTable_[inpSubPieceIdx].cIDXs.size() == 1) {
        op->dtTable_[inpSubPieceIdx].minCMemID =
            outSubPiece.placement.getMemId().at(0);
      } else {
        if (op->dtTable_[inpSubPieceIdx].minCMemID >
            outSubPiece.placement.getMemId().at(0))
          op->dtTable_[inpSubPieceIdx].minCMemID =
              outSubPiece.placement.getMemId().at(0);
      }
    }
  }
}

int DcgFE::getSubPieceIDX(std::vector<SliceInfo>& subPVec,
                          SliceInfo& subPiece) {
  for (int idx = 0; idx < subPVec.size(); idx++) {
    auto& currSubPiece = subPVec[idx];
    bool match = true;
    for (auto& key_ : currSubPiece.dimToStartCordinate) {
      DT_CHECK(currSubPiece.dimToSize_.count(key_.first));
      DT_CHECK(subPiece.dimToSize_.count(key_.first));
      DT_CHECK(subPiece.dimToStartCordinate.count(key_.first));
      if (currSubPiece.dimToStartCordinate.at(key_.first) !=
          subPiece.dimToStartCordinate.at(key_.first)) {
        match = false;
        break;
      }
      if (currSubPiece.dimToSize_.at(key_.first) !=
          subPiece.dimToSize_.at(key_.first)) {
        match = false;
        break;
      }
      if (currSubPiece.bigDimToSize_.at(key_.first) !=
          subPiece.bigDimToSize_.at(key_.first)) {
        match = false;
        break;
      }
      if (currSubPiece.placement.Type() != subPiece.placement.Type() ||
          currSubPiece.placement.getMemId() != subPiece.placement.getMemId() ||
          !(currSubPiece.placement.StartAddr() ==
            subPiece.placement.StartAddr())) {
        match = false;
        break;
      }
    }
    if (match) return idx;
  }
  return -1;
}

int DcgFE::getIdxForMatchingCMenID(const std::vector<SliceInfo>& outSP_,
                                   const std::vector<int>& myVec, int value) {
  for (int idx = 0; idx < myVec.size(); idx++) {
    DT_CHECK(outSP_[myVec.at(idx)].placement.getMemId().size() == 1);
    if (outSP_[myVec.at(idx)].placement.getMemId().at(0) == value)
      return myVec.at(idx);
  }
  return -1;
}

std::set<int> DcgFE::getCMenIDs(const std::vector<SliceInfo>& outSP_,
                                const std::vector<int>& myVec) {
  std::set<int> cMenIDs;
  for (int idx = 0; idx < myVec.size(); idx++) {
    DT_CHECK(outSP_[myVec.at(idx)].placement.getMemId().size() == 1);
    cMenIDs.insert(outSP_[myVec.at(idx)].placement.getMemId().at(0));
  }
  return cMenIDs;
}

void DcgFE::insertL3LUSortedpMemID(STCDPOpLx* op, int cMemID, int inpSPIDx) {
  bool inserted = false;
  if (op->coreIDtoDtKey_L3LU[cMemID].size()) {
    for (auto it = op->coreIDtoDtKey_L3LU[cMemID].begin();
         it != op->coreIDtoDtKey_L3LU[cMemID].end(); it++) {
      DT_CHECK(op->coreIDtoTrRank.count(op->dtTable_.at(inpSPIDx).pMemID));
      DT_CHECK(op->coreIDtoTrRank.count(op->dtTable_.at(*it).pMemID));
      if (op->coreIDtoTrRank.at(op->dtTable_.at(inpSPIDx).pMemID) <
          op->coreIDtoTrRank.at(op->dtTable_.at(*it).pMemID)) {
        op->coreIDtoDtKey_L3LU[cMemID].insert(it, inpSPIDx);
        inserted = true;
        break;
      }
    }
  }
  if (inserted == false) {
    op->coreIDtoDtKey_L3LU[cMemID].push_back(inpSPIDx);
  }
}

void DcgFE::insertL3SUSortedcMemID(STCDPOpLx* op, int pMemID, int inpSPIDx) {
  bool inserted = false;
  if (op->coreIDtoDtKey_L3SU[pMemID].size()) {
    for (auto it = op->coreIDtoDtKey_L3SU[pMemID].begin();
         it != op->coreIDtoDtKey_L3SU[pMemID].end(); it++) {
      // if (op->dtTable_.at(inpSPIDx).minCMemID < op->dtTable_[*it].minCMemID)
      // {
      if (getMinRankCoreID(op, inpSPIDx) < getMinRankCoreID(op, *it)) {
        op->coreIDtoDtKey_L3SU[pMemID].insert(it, inpSPIDx);
        inserted = true;
        break;
      }
    }
  }

  if (inserted == false) {
    op->coreIDtoDtKey_L3SU[pMemID].push_back(inpSPIDx);
  }
}

int DcgFE::getMinRankCoreID(STCDPOpLx* op, int inpSPIDx) {
  int minRankCoreID =
      op->coreIDtoTrRank.at(op->dtTable_.at(inpSPIDx).minCMemID);
  for (auto outSPIDx : op->dtTable_.at(inpSPIDx).cIDXs) {
    DT_CHECK(op->outSP_[outSPIDx].placement.getMemId().size());
    int coreID = op->outSP_[outSPIDx].placement.getMemId().back();
    DT_CHECK(op->coreIDtoTrRank.count(coreID));
    DT_CHECK(op->coreIDtoTrRank.count(minRankCoreID));
    if (op->coreIDtoTrRank.count(minRankCoreID) <
        op->coreIDtoTrRank.count(coreID))
      minRankCoreID = coreID;
  }
  return minRankCoreID;
}

/**
 * @brief This pass eliminates redundant transactions in dt table.
 *
 * @param op
 */
void DcgFE::eliminateRedundantTransaction(STCDPOpLx* op) {
  std::set<int> redundant_entires;
  for (const auto& kv : op->dtTable_) {
    const auto& transfer_info = kv.second;
    if (transfer_info.cIDXs.size() == 1) {
      // only within Lx transfers can be redundant
      const auto& inp_sp = op->inpSP_.at(transfer_info.pIDX);
      const auto& out_sp = op->outSP_.at(transfer_info.cIDXs.back());

      if (inp_sp.dimToStartCordinate == out_sp.dimToStartCordinate &&
          inp_sp.dimToSize_ == out_sp.dimToSize_ &&
          inp_sp.placement == out_sp.placement &&
          op->inpLds->dimToStickSize_ == op->outLds->dimToStickSize_ &&
          op->inpLds->layoutDimOrder_ == op->outLds->layoutDimOrder_) {
        if (inp_sp.bigDimToSize_ == out_sp.bigDimToSize_) {
          redundant_entires.insert(kv.first);  // this is redundant
        } else {
          // we can skip outer dims of size 1
          bool is_equal = true;
          for (int idx = op->outLds->layoutDimOrder_.size() - 1; idx >= 0;
               idx--) {
            auto& dimname = op->outLds->layoutDimOrder_.at(idx);
            if (inp_sp.dimToSize_.at(dimname) == 1) continue;
            if (inp_sp.dimToSize_.at(dimname) !=
                out_sp.dimToSize_.at(dimname)) {
              is_equal = false;
              break;
            }
          }

          if (is_equal)
            redundant_entires.insert(kv.first);  // this is redundant
        }
      }
    }
  }

  for (auto inp_sp_idx : redundant_entires) op->dtTable_.erase(inp_sp_idx);
}

void DcgFE::populateCoreProdConsumerList(STCDPOpLx* op) {
  auto& prodCons = op->prodConsList;
  for (auto& kv : op->dtTable_) {
    if (prodCons.find(kv.second.pMemID) == prodCons.end()) {
      prodCons[kv.second.pMemID];
    }
    auto& consSet = prodCons.at(kv.second.pMemID);
    for (auto& cidx : kv.second.cIDXs) {
      auto& csp = op->outSP_.at(cidx);
      DT_CHECK(csp.placement.Type() == SenComponents::LX);
      for (auto& conscore : csp.placement.getMemId()) {
        consSet.insert(conscore);
      }
    }
  }
  if (verbose_ > 0) {
    int maxConsumers = 0;
    for (auto& kv : prodCons) {
      if (kv.second.size() > maxConsumers) {
        maxConsumers = kv.second.size();
      }
    }
    for (auto& kv : prodCons) {
      std::cout << kv.first << " --> [ ";
      for (auto& entry : kv.second) {
        std::cout << entry << " ";
      }
      std::cout << "]" << std::endl;
    }
    std::cout << "maxConsumers: " << maxConsumers << std::endl;
  }
}

void DcgFE::computeInferredSegGroups(STCDPOpLx* op) {
  for (const auto& mapkv : op->prodConsList) {
    auto pID = mapkv.first;
    auto cList = mapkv.second;
    bool found = false;
    for (auto& segGropus : op->inferredSegGroups) {
      if (segGropus.count(pID)) {
        found = true;
      }

      if (!found) {
        for (auto cID : cList) {
          if (segGropus.count(cID)) {
            found = true;
            break;
          }
        }
      }

      if (found) {
        segGropus.insert(pID);
        for (auto cID : cList) {
          segGropus.insert(cID);
        }
        break;
      }
    }

    if (!found) {
      std::set<int> segGropus = {pID};
      for (auto cID : cList) {
        segGropus.insert(cID);
      }
      op->inferredSegGroups.push_back(segGropus);
    }
  }
}

void DcgFE::transformToPcfgSTCDPLxUnrolled(SenPcfg& newPcfg, baseSTCDPOp* op,
                                           int coreID, SenComponents pcfgType,
                                           bool reqSync /* = false*/) {
  DT_CHECK(pcfgType == SenComponents::L3SU || pcfgType == SenComponents::L3LU);

  std::vector<std::pair<std::vector<std::string>, std::vector<int>>>
      dtKeysPerInnerLOs;

  if (isInpFetchNeigh_) {
    dtKeysPerInnerLOs =
        pcfgType == SenComponents::L3SU
            ? clusterDtKeysUsingInnerLOsPartialUnRollInputfetch(
                  op->coreIDtoDtKey_L3SU[coreID], op,
                  !((STCDPOpLx*)op)->useUnicast, coreID, false)
            : clusterDtKeysUsingInnerLOsPartialUnRollInputfetch(
                  op->coreIDtoDtKey_L3LU[coreID], op, coreID, false,
                  (((STCDPOpLx*)op)->interLeaveDtTableIdx.size() > 0), true);
  } else {
    dtKeysPerInnerLOs =
        pcfgType == SenComponents::L3SU
            ? clusterDtKeysUsingInnerLOsPartialUnRoll(
                  op->coreIDtoDtKey_L3SU[coreID], op, false, coreID,
                  !((STCDPOpLx*)op)->useUnicast, false)
            : clusterDtKeysUsingInnerLOsPartialUnRoll(
                  op->coreIDtoDtKey_L3LU[coreID], op, true, coreID, false,
                  (((STCDPOpLx*)op)->interLeaveDtTableIdx.size() > 0));
  }

  SenPcfgNode* lastTopNodePcfg = nullptr;

  // extra initial sync for input neighbor fetch
  if (isInpFetchNeigh_ && pcfgType == SenComponents::L3LU) {
    auto key_first = std::make_pair(coreID, -1);
    if (myIFNInfo_.cIDDtIdxToDummyLXSyncCount.count(key_first)) {
      lastTopNodePcfg = createSyncWithinLoop(
          newPcfg, coreID, pcfgType,
          myIFNInfo_.cIDDtIdxToDummyLXSyncCount.at(key_first), newPcfg.srcNode,
          true, "-InpFetch-" + std::to_string(op->uniqueID) + "-0");
    }
  }

  if (pcfgType == SenComponents::L3SU && isInpFetchNeigh_ &&
      dtKeysPerInnerLOs.size() && myIFNInfo_.enRingFairnessComp) {
    double ringToCoreFreq = sysDef.coreFreq;
    DT_CHECK(sysDef.coreFreq == sysDef.lxCoreletBw / sysDef.ringBw);
    DT_CHECK(
        myIFNInfo_.inpSPIdxToChunkRank.count(dtKeysPerInnerLOs[0].second[0]));
    int chunkId =
        myIFNInfo_.inpSPIdxToChunkRank.at(dtKeysPerInnerLOs[0].second[0]);
    if (chunkId > 0) {
      DT_CHECK(myIFNInfo_.linkTraffic.size() > chunkId);
      int base_delay =
          myIFNInfo_.linkTraffic.at(chunkId - 1).datatr_prechunk_both;

      int num_segs = ((STCDPOpLx*)op)->inferredSegGroups.size();
      base_delay = (base_delay * ringToCoreFreq) / num_segs;

      lastTopNodePcfg = createDelayPcfgGraph(
          newPcfg, coreID, pcfgType, base_delay, newPcfg.srcNode, false,
          std::to_string(op->uniqueID) + "-initial-");
      if (myIFNInfo_.fairCompLevel > 4) {
        DT_CHECK(0);  // unsupported
      }
    }
  }

  std::map<std::string, std::pair<SenPcfgMvloopNode*, SenPcfgMvloopBranchNode*>>
      srq_loops_bes;
  if (isSRQProne(op)) {
    DT_CHECK(!isInpFetchNeigh_);
    DT_CHECK(lastTopNodePcfg == nullptr);
    auto stcdp_op = dynamic_cast<STCDPOpLx*>(op);

    for (const auto& kv_split : stcdp_op->split_factor_per_dim) {
      if (kv_split.second > 1) {
        auto loop_be = createLoopAndBranchEnd(
            newPcfg,
            "c" + std::to_string(coreID) + "-" +
                newPcfg.senComponentsToString.at(pcfgType) + "_SRQ_LOOP_" +
                kv_split.first + "_" + std::to_string(op->uniqueID),
            kv_split.second);
        srq_loops_bes[kv_split.first] = loop_be;
      }
    }
  }

  for (int idx = 0; idx < dtKeysPerInnerLOs.size(); idx++) {
    SenPcfgNode* lastNodethisLoop = nullptr;
    const std::vector<std::string>& loopOrder = dtKeysPerInnerLOs[idx].first;
    const std::vector<int>& dtKeys = dtKeysPerInnerLOs[idx].second;

    std::set<std::string> dimsToDrop;
    std::set<std::string> dimsToDropCandidate;
    // Data Transfer
    // figure out bigDimToSize
    std::map<std::string, double> orgBigDimToSize;
    std::map<std::string, double> orgDimToSize;
    if (pcfgType == SenComponents::L3LU || pcfgType == SenComponents::LXSU0) {
      int outSPIDx = getIdxForMatchingCMenID(
          op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
      orgBigDimToSize = op->outSP_[outSPIDx].bigDimToSize_;
      orgDimToSize = op->outSP_[outSPIDx].dimToSize_;
    } else {
      orgBigDimToSize = op->inpSP_.at(dtKeys.at(0)).bigDimToSize_;
      orgDimToSize = op->inpSP_.at(dtKeys.at(0)).dimToSize_;
    }

    // check for interleaving
    bool interleave = false;

    if (pcfgType == SenComponents::L3LU) {
      if (((STCDPOpLx*)op)->interLeaveDtTableIdx.size() > 0 &&
          dtKeys.size() > 1)
        if (((STCDPOpLx*)op)->interLeaveDtTableIdx.count(dtKeys.at(0))) {
          const auto& il_table =
              ((STCDPOpLx*)op)->interLeaveDtTableIdx.at(dtKeys.at(0));
          if (dtKeys.size() == il_table.size()) {
            // all keys should be found
            interleave = true;
            for (int i = 1; i < dtKeys.size(); i++) {
              if (!DCGUtils::isValPresent(il_table, dtKeys.at(i)))
                interleave = false;

              // if (((STCDPOpLx*)op)
              //        ->interLeaveDtTableIdx.at(dtKeys.at(0))
              //        .at(i) != dtKeys.at(i))
              //  interleave = false;
            }
          } else if (isInpFetchNeigh_) {
            // we allow skipping of one idx due to within LX-Transfers
            // all keys should be found
            interleave = true;
            bool foundSkipedIdx = false;
            for (int i = 1; i < dtKeys.size(); i++) {
              // int currILIdx = il_table.at(i);
              // if (op->dtTable_.at(currILIdx).pMemID == coreID) {
              //   DT_CHECK(!foundSkipedIdx);
              //   foundSkipedIdx = true;
              // }
              // if (((STCDPOpLx*)op)
              //         ->interLeaveDtTableIdx.at(dtKeys.at(0))
              //         .at(i + (foundSkipedIdx == true)) != dtKeys.at(i)) {
              //   interleave = false;
              //   break;
              // }

              if (!DCGUtils::isValPresent(il_table, dtKeys.at(i))) {
                interleave = false;
                break;
              }
            }
          }
        }
    }

    // check inpFetch sync
    bool reqInpFetchSyncToLX = false;
    bool isSyncAfterEachDt = false;

    if (interleave)
      DT_CHECK(dtKeys.size() <= sysDef.regInfoPerUnit.at(pcfgType)
                                    .at(RegType::GTR)
                                    .maxNum);  // due to GTR limitations

    if (pcfgType == SenComponents::L3LU && isInpFetchNeigh_) {
      DT_CHECK(myIFNInfo_.cIDtoL3toLXSyncDtIdx.count(coreID));
      if (myIFNInfo_.cIDtoL3toLXSyncDtIdx.at(coreID).count(dtKeys.back())) {
        int syncCount = 0;
        for (const auto inpSPIDx : dtKeys)
          if (myIFNInfo_.cIDtoL3toLXSyncDtIdx.at(coreID).count(inpSPIDx))
            syncCount++;

        if (syncCount == dtKeys.size())
          isSyncAfterEachDt = true;
        else
          DT_CHECK(syncCount == 1);

        DT_CHECK(pcfgType == SenComponents::L3LU);
        reqInpFetchSyncToLX = true;
      }
    }

    // outer loop
    SenPcfgMvloopNode* newOuterLoopNode =
        (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);

    newOuterLoopNode->name = "c" + std::to_string(coreID) + "-" +
                             newPcfg.senComponentsToString.at(pcfgType) +
                             "-Outer-Loop-" + std::to_string(idx) + "-" +
                             std::to_string(op->uniqueID);
    newOuterLoopNode->loopName = newOuterLoopNode->name;
    newOuterLoopNode->loopCount = interleave ? 1 : dtKeys.size();

    // outer loop end
    SenPcfgMvloopBranchNode* newOuterLoopBranchNode =
        (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
            SenPcfgNode::Type::MVLOOPBRANCH);

    newOuterLoopBranchNode->name = "c" + std::to_string(coreID) + "-" +
                                   newPcfg.senComponentsToString.at(pcfgType) +
                                   "-Outer-LoopBranch-" + std::to_string(idx) +
                                   "-" + std::to_string(op->uniqueID);
    newOuterLoopBranchNode->loopNode = newOuterLoopNode;
    newOuterLoopBranchNode->next.push_back(newOuterLoopNode);

    if (newPcfg.srcNode == nullptr) {
      newPcfg.srcNode = newOuterLoopNode;
    } else {
      if (lastTopNodePcfg != nullptr) {
        lastTopNodePcfg->next.push_back(newOuterLoopNode);
        newOuterLoopNode->prev.push_back(lastTopNodePcfg);
      }
    }

    // inner loops top and bottom nodes
    SenPcfgNode* lastTopNodeInner = newOuterLoopNode;
    SenPcfgNode* firstBotNodeInner = newOuterLoopBranchNode;
    lastNodethisLoop = newOuterLoopBranchNode;

    if (reqInpFetchSyncToLX) {
      // insert send-sync node
      SenPcfgSyncNode* newSyncNode =
          (SenPcfgSyncNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::SYNC);
      newSyncNode->name = "c" + std::to_string(coreID) + "-" +
                          newPcfg.senComponentsToString.at(pcfgType) +
                          "-sendSync-InpFetch" + "-" + std::to_string(idx) +
                          "-" + std::to_string(op->uniqueID);

      newSyncNode->self = SenComponents::L3LU;
      newSyncNode->sendOrRecv = 0;  // 0--> send
      newSyncNode->external.push_back(SenComponents::LXLU0);
      newSyncNode->external.push_back(SenComponents::LXLU1);

      DT_CHECK(senCompToISAptr != nullptr);
      if ((*senCompToISAptr).at(SenComponents::L3LU).sysDef.coreArch >=
          IsaCoreGen::RCUDD1A_ISA)
        newSyncNode->isSoft = true;
      else
        newSyncNode->isSoft = false;

      if (isSyncAfterEachDt) {
        // need to insert mvloop
        // firstBotNodeInner->prev.push_back(newSyncNode);
        // newSyncNode->next.push_back(firstBotNodeInner);
        // firstBotNodeInner = newSyncNode;

        // outer loop
        SenPcfgMvloopNode* newSyncLoopNode =
            (SenPcfgMvloopNode*)newPcfg.createPcfgNode(
                SenPcfgNode::Type::MVLOOP);

        newSyncLoopNode->name = "c" + std::to_string(coreID) + "-" +
                                newPcfg.senComponentsToString.at(pcfgType) +
                                "-Sync-Loop-" + std::to_string(idx) + "-" +
                                std::to_string(op->uniqueID);
        newSyncLoopNode->loopName = newSyncLoopNode->name;
        newSyncLoopNode->loopCount = dtKeys.size();

        // outer loop end
        SenPcfgMvloopBranchNode* newSyncLoopBranchNode =
            (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
                SenPcfgNode::Type::MVLOOPBRANCH);

        newSyncLoopBranchNode->name =
            "c" + std::to_string(coreID) + "-" +
            newPcfg.senComponentsToString.at(pcfgType) + "-Sync-LoopBranch-" +
            std::to_string(idx) + "-" + std::to_string(op->uniqueID);
        newSyncLoopBranchNode->loopNode = newSyncLoopNode;
        newSyncLoopBranchNode->next.push_back(newSyncLoopNode);

        lastNodethisLoop->next.push_back(newSyncLoopNode);
        newSyncLoopNode->prev.push_back(lastNodethisLoop);
        newSyncLoopNode->next.push_back(newSyncNode);
        newSyncNode->prev.push_back(newSyncLoopNode);
        newSyncNode->next.push_back(newSyncLoopBranchNode);
        newSyncLoopBranchNode->prev.push_back(newSyncNode);
        lastNodethisLoop = newSyncLoopBranchNode;
      } else {
        lastNodethisLoop->next.push_back(newSyncNode);
        newSyncNode->prev.push_back(lastNodethisLoop);
        lastNodethisLoop = newSyncNode;
      }
    }

    // find minLoopCollapseFactor (mlCf)
    int mlCf = -1;
    std::map<std::string, long> dimToLoopCount;
    int mcMode = op->dtTable_.at(dtKeys.at(0)).selectedMCMode;
    for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
      int inpSPIDx = dtKeys.at(idx3);
      DT_CHECK(op->dtTable_.at(inpSPIDx).collapseFactor > 0);
      if (mlCf == -1)
        mlCf = op->dtTable_.at(inpSPIDx).collapseFactor;
      else if (op->dtTable_.at(inpSPIDx).collapseFactor < mlCf)
        mlCf = op->dtTable_.at(inpSPIDx).collapseFactor;

      // find loop values
      for (int lpIdx = loopOrder.size() - 1; lpIdx >= 0; lpIdx--) {
        const auto& loopDimName = loopOrder[lpIdx];
        int outSPIDx;
        if (pcfgType == SenComponents::L3LU) {
          outSPIDx = getIdxForMatchingCMenID(
              op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
          DT_CHECK(outSPIDx >= 0);
          DT_CHECK(op->outSP_.size() > outSPIDx);
        } else {
          DT_CHECK(op->inpSP_.size() > inpSPIDx);
        }
        int loopCount = pcfgType == SenComponents::L3LU
                            ? op->outSP_[outSPIDx].dimToSize_.at(loopDimName)
                            : op->inpSP_[inpSPIDx].dimToSize_.at(loopDimName);
        // stick-level adjustments
        LdsInfo* lds =
            (pcfgType == SenComponents::L3LU) ? op->outLds : op->inpLds;

        int refBigDim = lds->dimToLayoutSize_.at(loopDimName);
        if (DCGUtils::isValPresent(lds->stickDimOrder_, loopDimName)) {
          loopCount = ceil(loopCount / lds->dimToStickSize_.at(loopDimName));
          refBigDim /= lds->dimToStickSize_.at(loopDimName);
        }

        if (dimToLoopCount.count(loopDimName))
          DT_CHECK(dimToLoopCount.at(loopDimName) == loopCount);
        else
          dimToLoopCount[loopDimName] = loopCount;

        if (dimToLoopCount.at(loopDimName) == 1 && refBigDim == 1 &&
            op->name == OpFuncs::STCDPOpLx)
          dimsToDropCandidate.insert(loopDimName);
      }

      if (pcfgType == SenComponents::L3SU)
        if (!((STCDPOpLx*)op)->useUnicast)
          DT_CHECK(mcMode == op->dtTable_.at(inpSPIDx).selectedMCMode);
    }

    std::map<std::string, std::string> dimToLoopName;
    bool canDropOutterDim = loopOrder.size() > mlCf ? true : false;

    for (int lpIdx = loopOrder.size() - 1; lpIdx >= (mlCf > 1 ? mlCf : 0);
         lpIdx--) {
      const auto& loopDimName = loopOrder[lpIdx];

      if (canDropOutterDim && dimToLoopCount.at(loopDimName) == 1 &&
          lpIdx > 0 && orgBigDimToSize.at(loopDimName) == 1) {
        // we can't drop the innermost dim yet, need backend enhancements
        dimsToDrop.insert(loopDimName);
        continue;
      } else {
        canDropOutterDim = false;
      }

      if (dimsToDropCandidate.count(loopDimName) && lpIdx > 0) {
        dimsToDrop.insert(loopDimName);
        continue;  // skip this dim
      }

      SenPcfgMvloopNode* newInnerMvLoopNode =
          (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);
      dimToLoopName[loopDimName] =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-IL-" + loopDimName +
          "-OL-" + std::to_string(idx) + "-" + std::to_string(op->uniqueID);
      newInnerMvLoopNode->name = dimToLoopName[loopDimName];
      newInnerMvLoopNode->loopName = dimToLoopName[loopDimName];
      DT_CHECK(dimToLoopCount.count(loopDimName));
      newInnerMvLoopNode->loopCount = dimToLoopCount.at(loopDimName);

      // inner loop end
      SenPcfgMvloopBranchNode* newInnerLoopBranchNode =
          (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
              SenPcfgNode::Type::MVLOOPBRANCH);

      newInnerLoopBranchNode->name =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-ILBranch-" +
          loopDimName + "-OL-" + std::to_string(idx) + "-" +
          std::to_string(op->uniqueID);
      newInnerLoopBranchNode->loopNode = newInnerMvLoopNode;
      newInnerLoopBranchNode->next.push_back(newInnerMvLoopNode);

      // hook in NodeGraph
      newInnerMvLoopNode->prev.push_back(lastTopNodeInner);
      lastTopNodeInner->next.push_back(newInnerMvLoopNode);
      firstBotNodeInner->prev.push_back(newInnerLoopBranchNode);
      newInnerLoopBranchNode->next.push_back(firstBotNodeInner);

      lastTopNodeInner = newInnerMvLoopNode;
      firstBotNodeInner = newInnerLoopBranchNode;
    }

    std::string mergedDimName;
    if (mlCf > 1) {
      int collapsedLC = 1;
      for (int lpIdx = mlCf - 1; lpIdx >= 0; lpIdx--) {
        const auto& loopDimName = loopOrder[lpIdx];
        mergedDimName += loopDimName;
        DT_CHECK(dimToLoopCount.count(loopDimName));
        collapsedLC *= dimToLoopCount.at(loopDimName);
      }

      SenPcfgMvloopNode* newInnerMvLoopNode =
          (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);
      dimToLoopName[mergedDimName] =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-IL-" + mergedDimName +
          "-OL-" + std::to_string(idx) + "-" + std::to_string(op->uniqueID);
      newInnerMvLoopNode->name = dimToLoopName[mergedDimName];
      newInnerMvLoopNode->loopName = dimToLoopName[mergedDimName];
      newInnerMvLoopNode->loopCount = collapsedLC;
      dimToLoopCount[mergedDimName] = collapsedLC;
      // inner loop end
      SenPcfgMvloopBranchNode* newInnerLoopBranchNode =
          (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
              SenPcfgNode::Type::MVLOOPBRANCH);

      newInnerLoopBranchNode->name =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-ILBranch-" +
          mergedDimName + "-OL-" + std::to_string(idx) + "-" +
          std::to_string(op->uniqueID);
      newInnerLoopBranchNode->loopNode = newInnerMvLoopNode;
      newInnerLoopBranchNode->next.push_back(newInnerMvLoopNode);

      // hook in NodeGraph
      newInnerMvLoopNode->prev.push_back(lastTopNodeInner);
      lastTopNodeInner->next.push_back(newInnerMvLoopNode);
      firstBotNodeInner->prev.push_back(newInnerLoopBranchNode);
      newInnerLoopBranchNode->next.push_back(firstBotNodeInner);

      lastTopNodeInner = newInnerMvLoopNode;
      firstBotNodeInner = newInnerLoopBranchNode;
    }

    // check if all biDimToSize are same..
    for (int idx3 = 1; idx3 < dtKeys.size(); idx3++) {
      int inpSPIDx = dtKeys.at(idx3);
      int outSPIDx;
      if (pcfgType == SenComponents::L3LU) {
        outSPIDx = getIdxForMatchingCMenID(
            op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
        DT_CHECK(orgBigDimToSize == op->outSP_[outSPIDx].bigDimToSize_);
      } else {
        DT_CHECK(op->inpSP_.size() > inpSPIDx);
        if (!(op->name == OpFuncs::ResizeNNHBM ||
              op->name == OpFuncs::ResizeNNLX))
          DT_CHECK(orgBigDimToSize == op->inpSP_[inpSPIDx].bigDimToSize_);
      }
    }

    if (pcfgType == SenComponents::L3SU || pcfgType == SenComponents::L3LU) {
      SenPcfgRingDtNode* newRingDtNode =
          (SenPcfgRingDtNode*)newPcfg.createPcfgNode(
              SenPcfgNode::Type::RINGDATATRANSFER);
      DtPair newDtPair;
      if (pcfgType == SenComponents::L3SU) {
        // RING transfer: send to ring
        newDtPair.src_ = SenComponents::LX;
        newDtPair.dst_ = SenComponents::RING;
      } else {
        // RING transfer: send to LX
        newDtPair.src_ = SenComponents::RING;
        newDtPair.dst_ = SenComponents::LX;
      }

      newRingDtNode->name =
          "c" + std::to_string(coreID) + "-" +
          newPcfg.senComponentsToString.at(pcfgType) + "-ringDT-" +
          newPcfg.senComponentsToString.at(newDtPair.src_) + "-" +
          newPcfg.senComponentsToString.at(newDtPair.dst_) + "-OL-" +
          std::to_string(idx) + "-" + std::to_string(op->uniqueID);
      newRingDtNode->coreletId = -1;  // corelet independent
      newRingDtNode->dtInfo = nullptr;
      newRingDtNode->srcDest = newDtPair;
      newRingDtNode->dsInfo = nullptr;
      newRingDtNode->dimLayoutOrder = (pcfgType == SenComponents::L3SU)
                                          ? op->inpLds->layoutDimOrder_
                                          : op->outLds->layoutDimOrder_;
      newRingDtNode->myBigDimSize = orgBigDimToSize;
      makeStickLevelAdjustments(
          newRingDtNode->myBigDimSize,
          (pcfgType == SenComponents::L3LU) ? op->outLds : op->inpLds);

      // handling of collapsed loops
      if (mlCf > 1) {
        newRingDtNode->myBigDimSize[mergedDimName] = 1;
        for (int lpIdx = mlCf - 1; lpIdx >= 0; lpIdx--) {
          const auto& loopDimName = loopOrder[lpIdx];
          newRingDtNode->myBigDimSize.at(mergedDimName) *=
              newRingDtNode->myBigDimSize.at(loopDimName);
          newRingDtNode->myBigDimSize.erase(loopDimName);
        }
        std::vector<std::string> newdimLayout;
        newdimLayout.push_back(mergedDimName);

        for (int dimIdx = mlCf; dimIdx < newRingDtNode->dimLayoutOrder.size();
             dimIdx++)
          newdimLayout.push_back(newRingDtNode->dimLayoutOrder.at(dimIdx));

        newRingDtNode->dimLayoutOrder = newdimLayout;
      }

      // handling droped dims
      if (dimsToDrop.size()) {
        for (auto& loopDimName : dimsToDrop) {
          newRingDtNode->myBigDimSize.erase(loopDimName);
        }

        std::vector<std::string> newdimLayout;
        for (int dimIdx = 0; dimIdx < newRingDtNode->dimLayoutOrder.size();
             dimIdx++)
          if (dimsToDrop.count(newRingDtNode->dimLayoutOrder.at(dimIdx)) == 0)
            newdimLayout.push_back(newRingDtNode->dimLayoutOrder.at(dimIdx));

        newRingDtNode->dimLayoutOrder = newdimLayout;
      }

      // add coreIds
      bool useUnicast = false;
      if (op->name == OpFuncs::STCDPOpLx)
        useUnicast = ((STCDPOpLx*)op)->useUnicast;

      bool canUseBurst = true;
      for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
        std::vector<int64_t> coreIdForRing;
        int inpSPIDx = dtKeys.at(idx3);
        if (pcfgType == SenComponents::L3LU) {
          // procuderID
          coreIdForRing.push_back(
              op->inpSP_[inpSPIDx].placement.getMemId().at(0));
        } else {
          // consumerID
          for (const auto& cIDX : op->dtTable_.at(inpSPIDx).cIDXs) {
            if (op->outSP_[cIDX].placement.getMemId().at(0) !=
                op->inpSP_[inpSPIDx].placement.getMemId().at(0))
              coreIdForRing.push_back(
                  op->outSP_[cIDX].placement.getMemId().at(0));
          }
        }

        std::pair<PcfgLccrCond, std::vector<int64_t>> newPcfgLccrCond;
        newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
        newPcfgLccrCond.first.condOp =
            dtKeys.size() == 1 ? CondOp::ALWAYS : CondOp::EQ;
        newPcfgLccrCond.first.condVal =
            (interleave ? 0 : dtKeys.size() - 1 - idx3);  // order matters
        newPcfgLccrCond.second = coreIdForRing;           // coreIDs

        if (dtKeys.size() > 1 || dscGlobal->doPatchProg) {
          newRingDtNode->coreIDForRingCondAndVal.push_back(newPcfgLccrCond);
        } else {
          newRingDtNode->coreIdForRing = coreIdForRing;  // set default values
        }

        if (idx3 == dtKeys.size() - 1) {
          // set default values
          newRingDtNode->coreIdForRing = coreIdForRing;
        }

        if (!op->dtTable_.at(inpSPIDx).useBurst) canUseBurst = false;

        // fill GTR info
        if (useUnicast == false) {
          GTRBurstInfo newCondGTR;
          newCondGTR.groupID = op->dtTable_.at(inpSPIDx).myGTR.groupID;
          newCondGTR.numSharers = op->dtTable_.at(inpSPIDx).myGTR.numSharers;
          newCondGTR.count = op->dtTable_.at(inpSPIDx).myGTR.count;
          newCondGTR.srcNodeID = op->dtTable_.at(inpSPIDx).myGTR.srcNodeID;
          newCondGTR.useBurst = op->dtTable_.at(inpSPIDx).useBurst;
          newRingDtNode->GTRAndBurstCondAndVal.push_back(
              std::make_pair(newPcfgLccrCond.first, newCondGTR));

          if (isInpFetchNeigh_)
            if (myIFNInfo_.forceGTRIMM) newRingDtNode->gtr_imm_opt_en = false;
        }
      }

      if (useUnicast) newRingDtNode->useBurst = canUseBurst;

      // set start Address
      if (pcfgType == SenComponents::L3SU) {
        fillAddr(newRingDtNode->SrcStartAddr(),
                 op->inpSP_.at(dtKeys.at(0)).placement.StartAddr(), 0);
      } else {
        int outSPIDx = getIdxForMatchingCMenID(
            op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
        fillAddr(newRingDtNode->DestStartAddr(),
                 op->outSP_[outSPIDx].placement.StartAddr(), 0);
      }

      if (dtKeys.size() > 1) {
        for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
          int inpSPIDx = dtKeys.at(idx3);
          int outSPIDx;
          if (pcfgType == SenComponents::L3LU) {
            outSPIDx = getIdxForMatchingCMenID(
                op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
            DT_CHECK(outSPIDx >= 0);
            DT_CHECK(op->outSP_.size() > outSPIDx);
          } else {
            DT_CHECK(op->inpSP_.size() > inpSPIDx);
          }
          std::pair<PcfgLccrCond, FoldManager<int64_t>> newPcfgLccrCond;
          newPcfgLccrCond.first.loopName = newOuterLoopNode->name;
          newPcfgLccrCond.first.condOp = CondOp::EQ;
          newPcfgLccrCond.first.condVal =
              (interleave ? 0 : dtKeys.size() - 1 - idx3);  // order matters
          fillAddr(newPcfgLccrCond.second,
                   pcfgType == SenComponents::L3LU
                       ? op->outSP_[outSPIDx].placement.StartAddr()
                       : op->inpSP_[inpSPIDx].placement.StartAddr(),
                   0);  // count
          if (pcfgType == SenComponents::L3LU) {
            newRingDtNode->DestStartCondAndVal().emplace_back(newPcfgLccrCond);
          } else {
            newRingDtNode->SrcStartCondAndVal().emplace_back(newPcfgLccrCond);
          }
        }
      }

      // no burst for now, add later..
      for (auto& myDim : newRingDtNode->myBigDimSize) {
        newRingDtNode->myLitDimSize[myDim.first] = 1;
      }

      // set bigStAddrOffsets for each inner loop
      int64_t addrSpan = -1;
      bool computedAddrSpan = false;
      for (int lpIdx = loopOrder.size() - 1; lpIdx >= mlCf; lpIdx--) {
        const auto& loopDimName = loopOrder[lpIdx];
        PcfgDtOffsets newDtOffset;
        init(newDtOffset);

        if (dimsToDrop.count(loopDimName)) continue;  // skip this dim

        // find location in dimLayoutOrder
        for (auto& dimName : newRingDtNode->dimLayoutOrder) {
          if (dimName == loopDimName) {
            break;
          } else {
            newDtOffset.dimOffset *= newRingDtNode->myBigDimSize[dimName];
          }
        }

        newRingDtNode->bigStAddrOffsets[dimToLoopName[loopDimName]] =
            newDtOffset;
        // compute lastAddr for LAR optimization
        if (!computedAddrSpan) {
          computedAddrSpan = true;
          addrSpan = newDtOffset.dimOffset * dimToLoopCount.at(loopDimName);
        }
      }

      // inner most loop offset
      {
        PcfgDtOffsets newDtOffset;
        init(newDtOffset);
        // find location in dimLayoutOrder
        const auto& loopDimName =
            mlCf > 1 ? mergedDimName : loopOrder.at(mlCf - 1);

        if (dimsToDrop.count(loopDimName) == 0) {
          for (auto& dimName : newRingDtNode->dimLayoutOrder) {
            if (dimName == loopDimName) {
              break;
            } else {
              newDtOffset.dimOffset *= newRingDtNode->myBigDimSize[dimName];
            }
          }
          DT_CHECK(dimToLoopName.count(loopDimName));
          newRingDtNode->bigStAddrOffsets[dimToLoopName.at(loopDimName)] =
              newDtOffset;
          // compute lastAddr for LAR optimization
          if (!computedAddrSpan) {
            computedAddrSpan = true;
            addrSpan = newDtOffset.dimOffset * dimToLoopCount.at(loopDimName);
          }
        }
      }
      addrSpan *= sysDef.bytesPerStick;

      // fill multicast mode info
      newRingDtNode->forceModeMC = {true, mcMode};

      if (srq_loops_bes.size()) {
        int offset = 1;
        std::map<std::string, int64_t> offset_map;
        auto lds = (pcfgType == SenComponents::L3SU) ? op->inpLds : op->outLds;
        for (int lp_idx = 0; lp_idx < lds->layoutDimOrder_.size(); lp_idx++) {
          auto dimname = lds->layoutDimOrder_.at(lp_idx);
          offset_map[dimname] = offset * orgDimToSize.at(dimname);
          offset *= orgBigDimToSize.at(dimname);
          if (lds->dimToStickSize_.count(dimname)) {
            offset_map.at(dimname) /= lds->dimToStickSize_.at(dimname);
            offset /= lds->dimToStickSize_.at(dimname);
          }
        }
        PcfgDtOffsets newDtOffset;
        init(newDtOffset);

        for (auto& kv_split : srq_loops_bes) {
          newDtOffset.dimOffset = offset_map.at(kv_split.first);
          newRingDtNode->bigStAddrOffsets[kv_split.second.first->loopName] =
              newDtOffset;
        }
      }

      if (interleave) {
        // hook in NodeGraph
        lastTopNodeInner->next.push_back(newRingDtNode);
        newRingDtNode->prev.push_back(lastTopNodeInner);
        lastTopNodeInner = newRingDtNode;

        // account for GTR sharing for input fetch neighbor
        if (isInpFetchNeigh_) {
          auto coreUnitPair = std::make_pair(coreID, SenComponents::L3LU);
          DT_CHECK(myIFNInfo_.dtIdxToringDtName[coreUnitPair].count(
                       dtKeys.at(0)) == 0);
          myIFNInfo_.dtIdxToringDtName[coreUnitPair][dtKeys.at(0)] =
              newRingDtNode->name;

          // add lastAddr for LAR sharing
          newRingDtNode->addrSpan = addrSpan;
        }

        if (useUnicast == false) {
          DT_CHECK(newRingDtNode->GTRAndBurstCondAndVal.size() ==
                   dtKeys.size());
        }

        // N-ways interleaving
        for (int il = 1; il < dtKeys.size(); il++) {
          SenPcfgRingDtNode* newRingDtNode2 =
              (SenPcfgRingDtNode*)newPcfg.createPcfgNode(
                  SenPcfgNode::Type::RINGDATATRANSFER);

          // copy
          newRingDtNode2->name =
              newRingDtNode->name + "IL-" + std::to_string(il);
          newRingDtNode2->dimLayoutOrder = newRingDtNode->dimLayoutOrder;
          newRingDtNode2->srcDest = {SenComponents::RING, SenComponents::LX};
          newRingDtNode2->useBurst = newRingDtNode->useBurst;
          newRingDtNode2->forceModeMC = newRingDtNode->forceModeMC;
          newRingDtNode2->myBigDimSize = newRingDtNode->myBigDimSize;
          newRingDtNode2->myLitDimSize = newRingDtNode->myLitDimSize;
          newRingDtNode2->bigStAddrOffsets = newRingDtNode->bigStAddrOffsets;
          newRingDtNode2->dtInfo = nullptr;
          newRingDtNode2->dsInfo = nullptr;
          newRingDtNode2->coreIdForRing =
              newRingDtNode->coreIDForRingCondAndVal.at(il).second;

          if (dscGlobal->doPatchProg) {
            newRingDtNode2->coreIDForRingCondAndVal.push_back(
                newRingDtNode->coreIDForRingCondAndVal.at(il));
            newRingDtNode2->coreIDForRingCondAndVal.back().first.condOp =
                CondOp::ALWAYS;
          }

          if (useUnicast == false) {
            newRingDtNode2->GTRAndBurstCondAndVal.push_back(
                newRingDtNode->GTRAndBurstCondAndVal.at(il));
            newRingDtNode2->GTRAndBurstCondAndVal.back().first.condOp =
                CondOp::ALWAYS;
            if (isInpFetchNeigh_)
              if (myIFNInfo_.forceGTRIMM)
                newRingDtNode2->gtr_imm_opt_en = false;
          }

          DT_CHECK(newRingDtNode->DestStartCondAndVal().size() ==
                   dtKeys.size());
          newRingDtNode2->DestStartAddr().clone(
              newRingDtNode->DestStartCondAndVal().at(il).second);

          // hook in NodeGraph
          lastTopNodeInner->next.push_back(newRingDtNode2);
          newRingDtNode2->prev.push_back(lastTopNodeInner);
          lastTopNodeInner = newRingDtNode2;

          // account for GTR sharing for input fetch neighbor
          if (isInpFetchNeigh_) {
            auto coreUnitPair = std::make_pair(coreID, SenComponents::L3LU);
            DT_CHECK(myIFNInfo_.dtIdxToringDtName[coreUnitPair].count(
                         dtKeys.at(il)) == 0);
            myIFNInfo_.dtIdxToringDtName[coreUnitPair][dtKeys.at(il)] =
                newRingDtNode2->name;

            // add lastAddr for LAR sharing
            newRingDtNode2->addrSpan = addrSpan;
          }
        }

        for (int il = 1; il < dtKeys.size(); il++) {
          if (useUnicast == false)
            newRingDtNode->GTRAndBurstCondAndVal.pop_back();
        }
        newRingDtNode->coreIdForRing =
            newRingDtNode->coreIDForRingCondAndVal.front().second;

        if (dscGlobal->doPatchProg) {
          auto temp = newRingDtNode->coreIDForRingCondAndVal.front();
          newRingDtNode->coreIDForRingCondAndVal.clear();
          newRingDtNode->coreIDForRingCondAndVal.push_back(temp);
          newRingDtNode->coreIDForRingCondAndVal.back().first.condOp =
              CondOp::ALWAYS;
        } else {
          newRingDtNode->coreIDForRingCondAndVal.clear();
        }
        if (useUnicast == false) {
          newRingDtNode->GTRAndBurstCondAndVal.back().first.condOp =
              CondOp::ALWAYS;
        }
        newRingDtNode->DestStartCondAndVal().clear();
      } else {
        // hook in NodeGraph
        lastTopNodeInner->next.push_back(newRingDtNode);
        newRingDtNode->prev.push_back(lastTopNodeInner);
        lastTopNodeInner = newRingDtNode;

        // account for GTR sharing for input fetch neighbor
        if (isInpFetchNeigh_) {
          DT_CHECK(dtKeys.size() == 1);
          auto coreUnitPair = std::make_pair(coreID, SenComponents::L3LU);
          DT_CHECK(myIFNInfo_.dtIdxToringDtName[coreUnitPair].count(
                       dtKeys.back()) == 0);
          myIFNInfo_.dtIdxToringDtName[coreUnitPair][dtKeys.back()] =
              newRingDtNode->name;

          // add lastAddr for LAR sharing
          // newRingDtNode->addrSpan = addrSpan;
        }
      }
    } else {
      DT_CHECK(0);
    }

    // hook in NodeGraph
    lastTopNodeInner->next.push_back(firstBotNodeInner);
    firstBotNodeInner->prev.push_back(lastTopNodeInner);

    lastTopNodePcfg = lastNodethisLoop;

    // extra after sync for input neighbor fetch
    if (isInpFetchNeigh_ && pcfgType == SenComponents::L3LU) {
      auto key_last = std::make_pair(coreID, dtKeys.back());
      if (myIFNInfo_.cIDDtIdxToDummyLXSyncCount.count(key_last)) {
        lastTopNodePcfg = createSyncWithinLoop(
            newPcfg, coreID, pcfgType,
            myIFNInfo_.cIDDtIdxToDummyLXSyncCount.at(key_last), lastTopNodePcfg,
            true,
            "-InpFetch-" + std::to_string(op->uniqueID) + "-" +
                std::to_string(dtKeys.back()));
      }
    }

    if (pcfgType == SenComponents::L3SU && isInpFetchNeigh_ &&
        myIFNInfo_.enRingFairnessComp) {
      bool enSync = false;
      double ringToCoreFreq = sysDef.coreFreq;
      DT_CHECK(sysDef.coreFreq == sysDef.lxCoreletBw / sysDef.ringBw);
      DT_CHECK(myIFNInfo_.inpSPIdxToChunkRank.count(dtKeys.back()));
      int chunkId = myIFNInfo_.inpSPIdxToChunkRank.at(dtKeys.back());
      DT_CHECK(myIFNInfo_.linkTraffic.size() > chunkId);
      int base_delay = std::max(myIFNInfo_.linkTraffic.at(chunkId).total_ccw,
                                myIFNInfo_.linkTraffic.at(chunkId).total_cw);
      if (idx != dtKeysPerInnerLOs.size() - 1) {
        int num_segs = ((STCDPOpLx*)op)->inferredSegGroups.size();
        int delta_delay = 0;
        if (myIFNInfo_.fairCompLevel == 1) {
          enSync = false;
        } else if (myIFNInfo_.fairCompLevel == 2) {
          // simple subtract delta_delay = max(pass_ccw,pass_cw)
          enSync = true;
          delta_delay = std::max(
              myIFNInfo_.linkTraffic.at(chunkId).core_ccw_pass.at(coreID),
              myIFNInfo_.linkTraffic.at(chunkId).core_cw_pass.at(coreID));
        } else if (myIFNInfo_.fairCompLevel == 3) {
          enSync = true;
          delta_delay = std::max(
              myIFNInfo_.linkTraffic.at(chunkId).core_ccw_pass.at(coreID) -
                  myIFNInfo_.linkTraffic.at(chunkId).min_flit_ccw,
              myIFNInfo_.linkTraffic.at(chunkId).core_cw_pass.at(coreID) -
                  myIFNInfo_.linkTraffic.at(chunkId).min_flit_cw);
          DT_CHECK(delta_delay >= 0);
        } else if (myIFNInfo_.fairCompLevel == 4) {
          enSync = true;
          delta_delay = std::max(
              myIFNInfo_.linkTraffic.at(chunkId).core_ccw_pass.at(coreID) -
                  myIFNInfo_.linkTraffic.at(chunkId).min_flit_ccw,
              myIFNInfo_.linkTraffic.at(chunkId).core_cw_pass.at(coreID) -
                  myIFNInfo_.linkTraffic.at(chunkId).min_flit_cw);

          int max_delta_delay =
              std::max(myIFNInfo_.linkTraffic.at(chunkId).max_flit_ccw -
                           myIFNInfo_.linkTraffic.at(chunkId).min_flit_ccw,
                       myIFNInfo_.linkTraffic.at(chunkId).max_flit_ccw -
                           myIFNInfo_.linkTraffic.at(chunkId).min_flit_cw);

          auto demoninator = max_delta_delay * num_segs * myIFNInfo_.kapa;
          if (demoninator >= 0)
            delta_delay = (delta_delay * 1.0 * base_delay) / demoninator;
          else
            delta_delay = 0;
        } else {
          DT_CHECK(0);  // unsupported
        }

        int total_delay = (base_delay * ringToCoreFreq) / num_segs -
                          (delta_delay * ringToCoreFreq);
        total_delay = total_delay < 0 ? 0 : total_delay;
        lastTopNodePcfg = createDelayPcfgGraph(
            newPcfg, coreID, pcfgType, total_delay, lastTopNodePcfg, enSync,
            std::to_string(op->uniqueID) + std::to_string(dtKeys.back()));
      }
    }
  }

  // attach outer loops
  if (srq_loops_bes.size()) {
    auto tail_pcfg = newPcfg.getPcfgGraphTailNode();
    for (auto& kv_split : srq_loops_bes) {
      auto loop_be = kv_split.second;
      tail_pcfg->next.push_back(loop_be.second);
      loop_be.second->prev.push_back(tail_pcfg);
      tail_pcfg = loop_be.second;

      loop_be.first->next.push_back(newPcfg.srcNode);
      newPcfg.srcNode->prev.push_back(loop_be.first);
      newPcfg.srcNode = loop_be.first;
    }
  }
}

std::vector<std::pair<std::vector<std::string>, std::vector<int>>>
DcgFE::clusterDtKeysUsingInnerLOsPartialUnRoll(
    std::vector<int>& dtKeys, baseSTCDPOp* op, bool isL3LU, int coreID,
    bool checkMCMode /*= false*/, bool checkInterLeaving /*= false*/) {
  std::vector<std::pair<std::vector<std::string>, std::vector<int>>>
      dtKeysPerInnerLOs;

  bool foundILPairFirst = false;
  bool foundILPairSecond = false;

  int maxValOuterLoopStcdp =
      ceil(dtKeys.size() / 10.0);  // if more then 16 then coalesce

  for (int idx = 0; idx < dtKeys.size(); idx++) {
    auto& myDtKey = dtKeys.at(idx);
    bool canMerge = false;
    if (dtKeysPerInnerLOs.size()) {
      if (dtKeysPerInnerLOs.back().first ==
          op->dtTable_.at(myDtKey).loopOrder) {
        // check if piece dimensions are same
        DT_CHECK(dtKeysPerInnerLOs.back().second.size());
        int inpSPIDx = dtKeysPerInnerLOs.back().second.back();
        if (op->inpSP_[inpSPIDx].dimToSize_ == op->inpSP_[myDtKey].dimToSize_) {
          bool bigDimMatch = true;
          if (isL3LU) {
            int outSPIDx1 = getIdxForMatchingCMenID(
                op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
            int outSPIDx2 = getIdxForMatchingCMenID(
                op->outSP_, op->dtTable_.at(myDtKey).cIDXs, coreID);
            if (op->outSP_.at(outSPIDx1).bigDimToSize_ !=
                op->outSP_.at(outSPIDx2).bigDimToSize_)
              bigDimMatch = false;
          } else {
            if (op->inpSP_[inpSPIDx].bigDimToSize_ !=
                op->inpSP_[myDtKey].bigDimToSize_)
              bigDimMatch = false;
          }

          if (bigDimMatch) {
            if (!checkMCMode) {
              canMerge = true;
            } else {
              if (op->dtTable_.at(inpSPIDx).selectedMCMode ==
                  op->dtTable_.at(myDtKey).selectedMCMode)
                canMerge = true;
            }
          }
        }
      }
    }

    if (canMerge && !foundILPairFirst) {
      if (dtKeysPerInnerLOs.back().second.size() >= maxValOuterLoopStcdp) {
        canMerge = false;  // force
      }
    }

    // check for interleaving
    if (checkInterLeaving) {
      // reset old pair info
      if (foundILPairSecond) {
        canMerge = false;
        foundILPairFirst = false;
        foundILPairSecond = false;
      }

      if (((STCDPOpLx*)op)->interLeaveDtTableIdx.count(myDtKey)) {
        DT_CHECK(((STCDPOpLx*)op)->interLeaveDtTableIdx.at(myDtKey).size() ==
                 2);
        if (idx + 1 < dtKeys.size()) {
          if (((STCDPOpLx*)op)->interLeaveDtTableIdx.at(myDtKey).at(1) ==
              dtKeys.at(idx + 1)) {
            canMerge = false;  // first entry should be first entry
            foundILPairFirst = true;
          }
        }
      }
    }

    if (canMerge) {
      dtKeysPerInnerLOs.back().second.push_back(myDtKey);
    } else {
      // create new entry
      dtKeysPerInnerLOs.resize(dtKeysPerInnerLOs.size() + 1);
      dtKeysPerInnerLOs.back().first = op->dtTable_.at(myDtKey).loopOrder;
      dtKeysPerInnerLOs.back().second.push_back(myDtKey);
    }

    // check for interleaving
    if (checkInterLeaving) {
      if (((STCDPOpLx*)op)->interLeaveDtTableIdx.count(myDtKey) == 0) {
        if (foundILPairFirst) {
          DT_CHECK(
              ((STCDPOpLx*)op)->interLeaveDtTableIdx.count(dtKeys.at(idx - 1)));
          DT_CHECK(((STCDPOpLx*)op)
                       ->interLeaveDtTableIdx.at(dtKeys.at(idx - 1))
                       .size() == 2);

          DT_CHECK(((STCDPOpLx*)op)
                       ->interLeaveDtTableIdx.at(dtKeys.at(idx - 1))
                       .at(1) == myDtKey);
          foundILPairSecond = true;
          DT_CHECK(canMerge);
        }
      }
    }
  }
  return dtKeysPerInnerLOs;
}

void DcgFE::computeInnerLoopCollapseFactor(baseSTCDPOp* op) {
  for (auto& dtEntry : op->dtTable_) {
    int minCollapseFac = 1;
    bool doesLoopOrderChanges = false;
    // check producer
    int inpSPIdx = dtEntry.first;
    double currDimProduct = 1;
    DT_CHECK(dtEntry.first == dtEntry.second.pIDX);
    int currProd = 1;
    if (op->inpLds->pieces_.size() &&
        dtEntry.second.loopOrder != op->inpLds->layoutDimOrder_)
      doesLoopOrderChanges = true;

    if (op->outLds->pieces_.size() &&
        dtEntry.second.loopOrder != op->outLds->layoutDimOrder_)
      doesLoopOrderChanges = true;

    for (int idx = 0; idx < dtEntry.second.loopOrder.size() - 1; idx++) {
      auto dimName = dtEntry.second.loopOrder[idx];
      if (op->inpSP_[inpSPIdx].dimToSize_.at(dimName) ==
          op->inpSP_[inpSPIdx].bigDimToSize_.at(dimName)) {
        minCollapseFac++;
        int dimSizeCurrRef =
            DCGUtils::isValPresent(op->inpLds->stickDimOrder_, dimName)
                ? op->inpLds->dimToStickSize_.at(dimName)
                : 1;
        currProd *=
            (op->inpSP_[inpSPIdx].dimToSize_.at(dimName) / dimSizeCurrRef);
      } else if (op->name == OpFuncs::STCDPOpLx) {
        // if all outerdims are one then we can collapse
        for (int idx2 = idx + 1; idx2 < dtEntry.second.loopOrder.size();
             idx2++) {
          auto dimNameOuter = dtEntry.second.loopOrder[idx2];
          int stickDim =
              DCGUtils::isValPresent(op->inpLds->stickDimOrder_, dimNameOuter)
                  ? op->inpLds->dimToStickSize_.at(dimNameOuter)
                  : 1;
          DT_CHECK(op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) >=
                   stickDim);
          if (op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) == stickDim)
            minCollapseFac++;
          else
            break;
        }
        DT_CHECK(minCollapseFac <= dtEntry.second.loopOrder.size());
        break;

      } else if (idx + 1 < dtEntry.second.loopOrder.size() - 1) {
        // peek into outer dim
        auto dimNameOuter = dtEntry.second.loopOrder[idx + 1];
        int dimSizeCurrRef =
            DCGUtils::isValPresent(op->inpLds->stickDimOrder_, dimName)
                ? op->inpLds->dimToStickSize_.at(dimName)
                : 1;
        int dimSizeNextRef =
            DCGUtils::isValPresent(op->inpLds->stickDimOrder_, dimNameOuter)
                ? op->inpLds->dimToStickSize_.at(dimNameOuter)
                : 1;

        if (op->inpSP_[inpSPIdx].bigDimToSize_.at(dimNameOuter) ==
                dimSizeNextRef &&
            currProd == 1 &&
            op->inpSP_[inpSPIdx].dimToSize_.at(dimName) == dimSizeCurrRef)
          minCollapseFac++;
        else
          break;
      } else {
        break;
      }
    }

    // check consumer
    for (const auto& outSPIdx : dtEntry.second.cIDXs) {
      int currCollapseFac = 1;
      int currProd = 1;
      for (int idx = 0; idx < dtEntry.second.loopOrder.size() - 1; idx++) {
        auto dimName = dtEntry.second.loopOrder[idx];
        if (op->outSP_.at(outSPIdx).dimToSize_.at(dimName) ==
            op->outSP_.at(outSPIdx).bigDimToSize_.at(dimName)) {
          currCollapseFac++;
          int dimSizeCurrRef =
              DCGUtils::isValPresent(op->outLds->stickDimOrder_, dimName)
                  ? op->outLds->dimToStickSize_.at(dimName)
                  : 1;
          currProd *=
              op->outSP_.at(outSPIdx).dimToSize_.at(dimName) / dimSizeCurrRef;

        } else if (op->name == OpFuncs::STCDPOpLx) {
          // if all outerdims are one then we can collapse
          for (int idx2 = idx + 1; idx2 < dtEntry.second.loopOrder.size();
               idx2++) {
            auto dimNameOuter = dtEntry.second.loopOrder[idx2];
            int stickDim =
                DCGUtils::isValPresent(op->outLds->stickDimOrder_, dimNameOuter)
                    ? op->outLds->dimToStickSize_.at(dimNameOuter)
                    : 1;
            DT_CHECK(op->outSP_.at(outSPIdx).dimToSize_.at(dimNameOuter) >=
                     stickDim);
            if (op->outSP_.at(outSPIdx).dimToSize_.at(dimNameOuter) == stickDim)
              currCollapseFac++;
            else
              break;
          }
          DT_CHECK(currCollapseFac <= dtEntry.second.loopOrder.size());
          break;
        } else if (idx + 1 < dtEntry.second.loopOrder.size() - 1) {
          // peek into outer dim
          auto dimNameOuter = dtEntry.second.loopOrder[idx + 1];
          int dimSizeCurrRef =
              DCGUtils::isValPresent(op->outLds->stickDimOrder_, dimName)
                  ? op->outLds->dimToStickSize_.at(dimName)
                  : 1;
          int dimSizeNextRef =
              DCGUtils::isValPresent(op->outLds->stickDimOrder_, dimNameOuter)
                  ? op->outLds->dimToStickSize_.at(dimNameOuter)
                  : 1;

          if (op->outSP_.at(outSPIdx).bigDimToSize_.at(dimNameOuter) ==
                  dimSizeNextRef &&
              currProd == 1 &&
              op->outSP_.at(outSPIdx).dimToSize_.at(dimName) == dimSizeCurrRef)
            currCollapseFac++;
          else
            break;
        } else {
          break;
        }
      }

      if (currCollapseFac < minCollapseFac) minCollapseFac = currCollapseFac;
    }
    if (doesLoopOrderChanges) {
      dtEntry.second.collapseFactor = 1;
    } else {
      dtEntry.second.collapseFactor = minCollapseFac;
    }
  }
}

void DcgFE::computeMulticastOptMetadata(STCDPOpLx* op,
                                        bool canReOrder /*= true*/) {
  bool istieBreakCCW = true;
  for (auto& dtEntry : op->dtTable_) {
    int inpSPIdx = dtEntry.first;
    dtEntry.second.CCWHopCWHop = std::make_pair(0, 0);
    dtEntry.second.hop_Mode3 = 0;
    dtEntry.second.hopSplit_Mode3 = {0, 0};
    int pCoreID = dtEntry.second.pMemID;
    std::vector<int> cCoreIDs;

    for (const auto outSPIdx : dtEntry.second.cIDXs) {
      DT_CHECK(op->outSP_.at(outSPIdx).placement.getMemId().size() == 1);
      int cCoreID = op->outSP_.at(outSPIdx).placement.getMemId().back();
      cCoreIDs.push_back(cCoreID);
      if (pCoreID == cCoreID) continue;
      if (cCoreID < pCoreID) cCoreID += maxNumCores;

      dtEntry.second.CCWHopCWHop.first =
          std::max(dtEntry.second.CCWHopCWHop.first, cCoreID - pCoreID);
      dtEntry.second.CCWHopCWHop.second = std::max(
          dtEntry.second.CCWHopCWHop.second, maxNumCores + pCoreID - cCoreID);

      // mode =3
      dtEntry.second.hop_Mode3 = std::max(
          dtEntry.second.hop_Mode3,
          std::min(cCoreID - pCoreID, maxNumCores + pCoreID - cCoreID));

      if (cCoreID - pCoreID < maxNumCores + pCoreID - cCoreID)
        dtEntry.second.hopSplit_Mode3.first =
            std::max(dtEntry.second.hopSplit_Mode3.first, cCoreID - pCoreID);
      else
        dtEntry.second.hopSplit_Mode3.second =
            std::max(dtEntry.second.hopSplit_Mode3.second,
                     maxNumCores + pCoreID - cCoreID);
    }

    if (dtEntry.second.CCWHopCWHop.first == dtEntry.second.CCWHopCWHop.second) {
      if (istieBreakCCW)
        dtEntry.second.bestMCMode = 1;
      else
        dtEntry.second.bestMCMode = 2;

      istieBreakCCW = (istieBreakCCW == true ? false : true);
    } else if (dtEntry.second.CCWHopCWHop.first <
               dtEntry.second.CCWHopCWHop.second) {
      dtEntry.second.bestMCMode = 1;
    } else {
      dtEntry.second.bestMCMode = 2;
    }

    dtEntry.second.cMemIDs = cCoreIDs;
    dtEntry.second.minHops = std::min(dtEntry.second.CCWHopCWHop.first,
                                      dtEntry.second.CCWHopCWHop.second);
  }

  // set start window based on producer order
  for (const auto& mapkv : op->coreIDtoDtKey_L3SU) {
    for (int idx = 0; idx < mapkv.second.size(); idx++) {
      int inpSPIdx = mapkv.second.at(idx);
      DT_CHECK(op->dtTable_.count(inpSPIdx));
      op->dtTable_.at(inpSPIdx).idealStWindow = idx;
    }
  }

  // reorder sub-pieces within each consumer
  // can only move subpiece with procuder time-window =0
  if (canReOrder) {
    for (auto& mapkv : op->coreIDtoDtKey_L3LU) {
      DT_CHECK(mapkv.second.size());
      std::vector<int> reOrderList_t0;
      std::vector<int> reOrderList_rest;
      for (int idx = 0; idx < mapkv.second.size(); idx++) {
        int inpSPIdx = mapkv.second.at(idx);
        if (op->dtTable_.at(inpSPIdx).idealStWindow == 0) {
          reOrderList_t0.push_back(inpSPIdx);
        } else {
          reOrderList_rest.push_back(inpSPIdx);
        }
      }

      for (auto inpSPIdx : reOrderList_rest) reOrderList_t0.push_back(inpSPIdx);

      mapkv.second = reOrderList_t0;  // re-order list
    }
  }

  // re-adjust start window based on producer order
  int numAdjusts = 0;
  for (int i = 0; i < 4; i++) {
    for (const auto& mapkv : op->coreIDtoDtKey_L3LU) {
      DT_CHECK(mapkv.second.size());
      int reftime = op->dtTable_.at(mapkv.second.at(0)).idealStWindow;
      for (int idx = 1; idx < mapkv.second.size(); idx++) {
        reftime++;
        int inpSPIdx = mapkv.second.at(idx);
        if (op->dtTable_.at(inpSPIdx).idealStWindow < reftime) {
          numAdjusts++;
          op->dtTable_.at(inpSPIdx).idealStWindow = reftime;
          // change the producer
          int pCoreID = op->inpSP_.at(inpSPIdx).placement.getMemId().back();
          int foundCount = 0;
          for (const auto& pIdx : op->coreIDtoDtKey_L3SU.at(pCoreID)) {
            if (inpSPIdx == pIdx) {
              foundCount = 1;
            } else if (foundCount > 0) {
              if (op->dtTable_.at(pIdx).idealStWindow <= reftime + foundCount) {
                op->dtTable_.at(pIdx).idealStWindow = reftime + foundCount;
                foundCount++;
              } else {
                break;
              }
            }
          }
        } else if (op->dtTable_.at(inpSPIdx).idealStWindow > reftime) {
          reftime = op->dtTable_.at(inpSPIdx).idealStWindow;
        }
      }
    }
    if (numAdjusts == 0) break;
    numAdjusts = 0;
  }

  DT_CHECK(numAdjusts == 0);

  // populate idealStWindowToDtKey, maxBurst, trVolume
  op->idealStWindowToDtKey.clear();
  op->maxBeforeTransactions_ = 0;
  for (const auto& mapkv : op->coreIDtoDtKey_L3SU) {
    double numBeforeTransactions = 0;
    for (int idx = 0; idx < mapkv.second.size(); idx++) {
      int inpSPIdx = mapkv.second.at(idx);
      auto& dtEntry = op->dtTable_.at(inpSPIdx);
      op->idealStWindowToDtKey[dtEntry.idealStWindow].push_back(inpSPIdx);

      dtEntry.maxBurst = 1;
      dtEntry.trVolume = 1;
      dtEntry.numTransactions_ = 1;
      for (int idx = 0; idx < dtEntry.loopOrder.size(); idx++) {
        auto dimName = dtEntry.loopOrder.at(idx);
        dtEntry.trVolume *= op->inpSP_.at(inpSPIdx).dimToSize_.at(dimName);
        dtEntry.numTransactions_ *=
            op->inpSP_.at(inpSPIdx).dimToSize_.at(dimName) /
            op->inpLds->getNumElemsInStick(dimName);
        if (idx < dtEntry.collapseFactor) {
          dtEntry.maxBurst *= op->inpSP_.at(inpSPIdx).dimToSize_.at(dimName);
          if (op->inpLds->dimToStickSize_.count(dimName))
            dtEntry.maxBurst /= op->inpLds->dimToStickSize_.at(dimName);
        }
      }

      dtEntry.trCost = (double)dtEntry.minHops / dtEntry.maxBurst;

      if (dtEntry.maxBurst > sysDef.l3BurstSize)
        dtEntry.maxBurst = sysDef.l3BurstSize;

      dtEntry.numTransactions_ /= dtEntry.maxBurst;
      dtEntry.numBeforeTransactions_ = numBeforeTransactions;
      op->maxBeforeTransactions_ =
          std::max(op->maxBeforeTransactions_, (int64_t)numBeforeTransactions);

      numBeforeTransactions += dtEntry.numTransactions_;

      if (numBeforeTransactions >= SRQ_HW_BUG_THRES) {
        op->is_SRQ_HW_Bug_prone_ = true;
      }
    }
  }

  // sort  idealStWindowToDtKey
  for (auto& mapkv : op->idealStWindowToDtKey) {
    for (int idx2 = 0; idx2 < mapkv.second.size(); idx2++) {
      for (int idx3 = 1; idx3 < mapkv.second.size(); idx3++) {
        int inpSPIdx1 = mapkv.second.at(idx3);
        int inpSPIdxCurr = mapkv.second.at(idx2);
        if (op->dtTable_.at(inpSPIdxCurr).trCost <
            op->dtTable_.at(inpSPIdx1).trCost) {
          mapkv.second.at(idx2) = inpSPIdx1;
          mapkv.second.at(idx3) = inpSPIdxCurr;
        }
      }
    }
  }

  // allocate mode
  int globalContentionCCW = 0;
  int globalContentionCW = 0;
  int maxTimeStep = 0;
  for (auto& mapkv : op->idealStWindowToDtKey) {
    std::vector<std::pair<int, int>> ccwCoresScrDest;  // list of [scr,dest)
    std::vector<std::pair<int, int>> ccCoresScrDest;   // list of [scr,dest)
    int contentionCCW = 0;
    int contentionCW = 0;

    for (int idx2 = 0; idx2 < mapkv.second.size(); idx2++) {
      int inpSPIdxCurr = mapkv.second.at(idx2);
      auto& dtEntry = op->dtTable_.at(inpSPIdxCurr);

      if (dtEntry.bestMCMode == 1) {
        std::pair<int, int> srcHop =
            std::make_pair(dtEntry.pMemID, dtEntry.CCWHopCWHop.first);
        bool hasContentionCCW = doesSegmentOverLap(ccwCoresScrDest, srcHop, 1);

        if (hasContentionCCW) {
          // CW contention
          srcHop.second = dtEntry.CCWHopCWHop.second;
          bool hasContentionCW = doesSegmentOverLap(ccCoresScrDest, srcHop, 2);

          if (hasContentionCW) {
            if (globalContentionCCW > globalContentionCW) {
              dtEntry.selectedMCMode = 2;  // contention --> cw to balance
              contentionCW++;
              globalContentionCW++;
            } else {
              dtEntry.selectedMCMode = 1;  // contention --> ccw to balance
              contentionCCW++;
              globalContentionCCW++;
            }
          } else {
            dtEntry.selectedMCMode = 2;  // noContention --> CW
          }
        } else {
          dtEntry.selectedMCMode = dtEntry.bestMCMode;
        }
      } else if (dtEntry.bestMCMode == 2) {
        std::pair<int, int> srcHop =
            std::make_pair(dtEntry.pMemID, dtEntry.CCWHopCWHop.second);
        bool hasContentionCW = doesSegmentOverLap(ccCoresScrDest, srcHop, 2);
        if (hasContentionCW) {
          // CCW contention
          srcHop.second = dtEntry.CCWHopCWHop.first;
          bool hasContentionCCW =
              doesSegmentOverLap(ccwCoresScrDest, srcHop, 1);
          if (hasContentionCCW) {
            if (globalContentionCCW > globalContentionCW) {
              dtEntry.selectedMCMode = 2;  // contention --> cw to balance
              contentionCW++;
              globalContentionCW++;
            } else {
              dtEntry.selectedMCMode = 1;  // contention --> ccw to balance
              contentionCCW++;
              globalContentionCCW++;
            }
          } else {
            dtEntry.selectedMCMode = 1;  // noContention --> CCW
          }

        } else {
          dtEntry.selectedMCMode = dtEntry.bestMCMode;
        }
      } else {
        DT_CHECK(0);  // need to add support
      }

      if (dtEntry.selectedMCMode == 1) {
        int endCore = dtEntry.CCWHopCWHop.first + dtEntry.pMemID;
        if (endCore >= maxNumCores) {
          // break into two
          ccwCoresScrDest.push_back({dtEntry.pMemID, 32});  // need to include
                                                            // 31
          ccwCoresScrDest.push_back({0, endCore % maxNumCores});
        } else {
          ccwCoresScrDest.push_back({dtEntry.pMemID, endCore});
        }
      } else if (dtEntry.selectedMCMode == 2) {
        int endCore = dtEntry.pMemID - dtEntry.CCWHopCWHop.second;
        if (endCore < 0) {
          // break into two
          ccCoresScrDest.push_back({dtEntry.pMemID, -1});  // need to include 0
          ccCoresScrDest.push_back({31, endCore + maxNumCores});
        } else {
          ccCoresScrDest.push_back({dtEntry.pMemID, endCore});
        }
      } else {
        DT_CHECK(0);  // need to add support
      }
    }

    // record contention
    DT_CHECK(op->idealStWindowToContention.count(mapkv.first) == 0);
    op->idealStWindowToContention[mapkv.first] = {contentionCCW, contentionCW};

    if (maxTimeStep < mapkv.first) maxTimeStep = mapkv.first;
  }

  // creating opportunities where two producers can simultaneously send data
  // on different rings to a common consumer

  if (op->idealStWindowToDtKey.size() && !isInpFetchNeigh_) {
    DT_CHECK(op->idealStWindowToDtKey.count(0));
    DT_CHECK(op->idealStWindowToContention.count(0));

    for (int ts = 1; ts <= maxTimeStep; ts++) {
      DT_CHECK(op->idealStWindowToDtKey.count(ts));
      DT_CHECK(op->idealStWindowToContention.count(ts));
      const auto& contention = op->idealStWindowToContention.at(ts);
      auto& entries = op->idealStWindowToDtKey.at(ts);
      if (contention.first == 0 && contention.second == 0 &&
          entries.size() <= 2) {
        if (entries.size() == 2) {
          // check if swapping modes is superior
          auto& dtEntry1 = op->dtTable_.at(entries.at(0));
          auto& dtEntry2 = op->dtTable_.at(entries.at(1));

          // check prior time step
          DT_CHECK(op->idealStWindowToDtKey.count(ts - 1));
          auto& priorEntries = op->idealStWindowToDtKey.at(ts - 1);
          bool foundNFlipEnt1 = false;
          bool foundNFlipEnt2 = false;

          for (auto priorInpIdx : priorEntries) {
            auto& priorDtEntry = op->dtTable_.at(priorInpIdx);
            if (priorDtEntry.cMemIDs == dtEntry1.cMemIDs) {
              if (priorDtEntry.selectedMCMode == dtEntry1.selectedMCMode)
                foundNFlipEnt1 = true;
            } else if (priorDtEntry.cMemIDs == dtEntry2.cMemIDs) {
              if (priorDtEntry.selectedMCMode == dtEntry2.selectedMCMode)
                foundNFlipEnt2 = true;
            }
          }

          if (foundNFlipEnt1 && foundNFlipEnt2) {
            int temp = dtEntry1.selectedMCMode;
            dtEntry1.selectedMCMode = dtEntry2.selectedMCMode;
            dtEntry2.selectedMCMode = temp;
          }
        } else if (entries.size() == 1) {
          // check if swapping modes is superior
          DT_CHECK(op->idealStWindowToDtKey.count(ts - 1));
          auto& dtEntry1 = op->dtTable_.at(entries.at(0));
          auto& priorEntries = op->idealStWindowToDtKey.at(ts - 1);
          for (auto priorInpIdx : priorEntries) {
            auto& priorDtEntry = op->dtTable_.at(priorInpIdx);
            if (priorDtEntry.cMemIDs == dtEntry1.cMemIDs) {
              if (priorDtEntry.selectedMCMode == dtEntry1.selectedMCMode) {
                if (dtEntry1.selectedMCMode == 1)
                  dtEntry1.selectedMCMode = 2;
                else if (dtEntry1.selectedMCMode == 2)
                  dtEntry1.selectedMCMode = 1;
                else
                  DT_CHECK(0);  // unexpected
              }
            }
          }
        }
      }
    }
  }
}

void DcgFE::promoteToMode3(STCDPOpLx* op) {
  // if (op->inferredSegGroups.size() >= 2) {
  for (auto& dtEntry : op->dtTable_) {
    if (dtEntry.second.hop_Mode3 != -1 &&
        dtEntry.second.hop_Mode3 <= (maxNumCores / 8 * 3)) {
      if (dtEntry.second.selectedMCMode == 1) {
        // CCW
        if (dtEntry.second.hop_Mode3 < dtEntry.second.CCWHopCWHop.first)
          dtEntry.second.selectedMCMode = 3;
      } else if (dtEntry.second.selectedMCMode == 2) {
        if (dtEntry.second.hop_Mode3 < dtEntry.second.CCWHopCWHop.second)
          dtEntry.second.selectedMCMode = 3;
      }
    }

    if (dtEntry.second.selectedMCMode == 3 &&
        dtEntry.second.hop_Mode3 == dtEntry.second.minHops) {
      dtEntry.second.selectedMCMode = dtEntry.second.bestMCMode;
    }
  }
  //}
}

bool DcgFE::doesSegmentOverLap(std::vector<std::pair<int, int>> list,
                               std::pair<int, int> srcHop, int mcMode) {
  bool hasContention = false;
  int startCore = srcHop.first;
  int endCore;
  std::vector<std::pair<int, int>> srcDest;
  if (mcMode == 1) {  // CCW
    endCore = srcHop.second + startCore;
    if (endCore >= maxNumCores) {
      // break into two
      srcDest.push_back({startCore, 32});
      srcDest.push_back({-1, endCore % maxNumCores});
    } else {
      srcDest.push_back({startCore, endCore});
    }

    for (const auto& entryToCheck : srcDest) {
      if (hasContention) break;
      for (const auto& tbEntry : list) {
        if (entryToCheck.second <= tbEntry.first) {
          // no contention
        } else if (entryToCheck.first >= tbEntry.second) {
          // no contention
        } else {
          hasContention = true;
          break;
        }
      }
    }
  } else if (mcMode == 2) {
    endCore = startCore - srcHop.second;
    if (endCore < 0) {
      // break into two
      srcDest.push_back({startCore, -1});
      srcDest.push_back({32, endCore + maxNumCores});
    } else {
      srcDest.push_back({startCore, endCore});
    }

    for (const auto& entryToCheck : srcDest) {
      if (hasContention) break;
      for (const auto& tbEntry : list) {
        if (entryToCheck.first <= tbEntry.second) {
          // start < endOfTbEntry --> no contention
        } else if (entryToCheck.second >= tbEntry.first) {
          // end >= startOfTbEntry -->no contention
        } else {
          hasContention = true;
          break;
        }
      }
    }
  } else {
    DT_CHECK(0);
  }
  return hasContention;
}

void DcgFE::dumpMulticastOptMetadata(std::string dscName, STCDPOpLx* op,
                                     std::string fileName) {
  std::ofstream outFile;
  char buf[20];
  outFile.open(fileName, std::ofstream::app);
  if (!outFile.is_open())
    DCGUtils::message("Error: Unable to open file=" + fileName);

  outFile << "---------------------------------------------------------------"
          << std::endl;
  outFile << "Node Name : " << dscName << "\n" << std::endl;

  outFile << std::setw(12) << " TimeWindow," << std::setw(18)
          << "#numParallelTrs," << std::setw(18) << "Cont.(CCW,CW),"
          << std::setw(4) << "  "
          << "List : TrID1 (preferred mode | selected | maxBurst | minHop | "
             "myCoreID), "
             "TrID2, .."
          << std::endl;
  for (const auto& mapkv : op->idealStWindowToDtKey) {
    std::string text = std::to_string(mapkv.first) + ",";
    std::string text3 = std::to_string(mapkv.second.size()) + ",";
    DT_CHECK(op->idealStWindowToContention.count(mapkv.first));
    auto& contention = op->idealStWindowToContention.at(mapkv.first);
    std::string text4 = "[" + std::to_string(contention.first) + "," +
                        std::to_string(contention.second) + "],";
    outFile << std::setw(12) << text << std::setw(18) << text3 << std::setw(18)
            << text4 << std::setw(4) << "  ";
    for (int idx2 = 0; idx2 < mapkv.second.size(); idx2++) {
      int inpSPIdxCurr = mapkv.second.at(idx2);
      const auto& dtEntry = op->dtTable_.at(inpSPIdxCurr);
      std::string mode = dtEntry.bestMCMode == 1   ? "CCW"
                         : dtEntry.bestMCMode == 2 ? "CW"
                                                   : "R";
      std::string selected = dtEntry.selectedMCMode == 1   ? "CCW"
                             : dtEntry.selectedMCMode == 2 ? "CW"
                                                           : "R";
      std::string text2 = std::to_string(inpSPIdxCurr) + " (" + mode + " | " +
                          selected + " | " + std::to_string(dtEntry.maxBurst) +
                          " | " + std::to_string(dtEntry.minHops) + " | " +
                          std::to_string(dtEntry.pMemID) + ")";

      if (idx2 != mapkv.second.size() - 1) text2 += ", ";
      outFile << text2;
    }
    outFile << std::endl;
  }
  outFile << "---------------------------------------------------------------"
          << std::endl;
  outFile.close();
}

void DcgFE::transformToPcfgSfp(SenPcfg& newPcfg, baseSTCDPOp* op, int coreID,
                               SenComponents pcfgType) {
  DT_CHECK(op->name == OpFuncs::STCDPOpLx || op->name == OpFuncs::STCDPOpHBM ||
           op->name == OpFuncs::ResizeNNHBM || op->name == OpFuncs::ResizeNNLX);

  DT_CHECK(op->coreIDtoDtKey_LX.count(coreID));
  const std::vector<int>& dtKeys = op->coreIDtoDtKey_LX.at(coreID);

  std::vector<int> lpCount;
  double totalSticks = 0;
  int mainProduct = -1;
  bool allSamePieces = true;
  for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
    int inpSPIDx = dtKeys.at(idx3);
    auto dimSize = op->inpSP_[inpSPIDx].dimToSize_;

    makeStickLevelAdjustments(dimSize, op->inpLds);
    double product = 1;
    for (const auto& mapkv : dimSize) product *= mapkv.second;

    totalSticks += product;
    if (idx3 == 0)
      mainProduct = product;
    else if (mainProduct != product)
      allSamePieces = false;
  }

  if (totalSticks > 16384) {
    DT_CHECK(allSamePieces);
    DT_CHECK(dtKeys.size() < 16384);
    lpCount.push_back(dtKeys.size());
    if (mainProduct >= 16384) {
      int inpSPIDx = dtKeys.at(0);
      auto dimSize = op->inpSP_[inpSPIDx].dimToSize_;
      for (const auto& mapkv : dimSize) lpCount.push_back(mapkv.second);
    } else {
      lpCount.push_back(mainProduct);
    }
  } else {
    lpCount.push_back(totalSticks);
  }

  std::map<std::string, int> upSizeFactor;
  if (op->name == OpFuncs::ResizeNNHBM) {
    ResizeNNHBM* myOptemp = (ResizeNNHBM*)op;
    upSizeFactor = myOptemp->upSizeFactor;
  } else if (op->name == OpFuncs::ResizeNNLX) {
    ResizeNNLX* myOptemp = (ResizeNNLX*)op;
    upSizeFactor = myOptemp->upSizeFactor;
  }
  int totalUpSizeFactor = 1;
  for (auto& mapkv : upSizeFactor) {
    totalUpSizeFactor *= mapkv.second;
  }

  if (totalUpSizeFactor > 1) {
    DT_CHECK(allSamePieces);
    lpCount.push_back(totalUpSizeFactor);
  }

  SenPcfgNode* lastTopNodeOuter = nullptr;
  SenPcfgNode* firstBotNodeOuter = nullptr;

  // outer loop
  std::vector<std::string> loopName;
  for (int idx = 0; idx < lpCount.size(); idx++) {
    SenPcfgMvloopNode* newOuterLoopNode =
        (SenPcfgMvloopNode*)newPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);

    newOuterLoopNode->name = "c" + std::to_string(coreID) + "-" +
                             newPcfg.senComponentsToString.at(pcfgType) +
                             "-Outer-Loop-" + std::to_string(idx) + "-" +
                             std::to_string(op->uniqueID);
    newOuterLoopNode->loopName = newOuterLoopNode->name;
    loopName.push_back(newOuterLoopNode->loopName);
    newOuterLoopNode->loopCount = lpCount.at(idx);

    // outer loop end
    SenPcfgMvloopBranchNode* newOuterLoopBranchNode =
        (SenPcfgMvloopBranchNode*)newPcfg.createPcfgNode(
            SenPcfgNode::Type::MVLOOPBRANCH);

    newOuterLoopBranchNode->name = "c" + std::to_string(coreID) + "-" +
                                   newPcfg.senComponentsToString.at(pcfgType) +
                                   "-Outer-LoopBranch-" + std::to_string(idx) +
                                   "-" + std::to_string(op->uniqueID);
    newOuterLoopBranchNode->loopNode = newOuterLoopNode;
    newOuterLoopBranchNode->next.push_back(newOuterLoopNode);

    if (lastTopNodeOuter == nullptr) {
      newPcfg.srcNode = newOuterLoopNode;
      lastTopNodeOuter = newOuterLoopNode;
      DT_CHECK(firstBotNodeOuter == nullptr);
      firstBotNodeOuter = newOuterLoopBranchNode;
    } else {
      newOuterLoopNode->prev.push_back(lastTopNodeOuter);
      lastTopNodeOuter->next.push_back(newOuterLoopNode);

      firstBotNodeOuter->prev.push_back(newOuterLoopBranchNode);
      newOuterLoopBranchNode->next.push_back(firstBotNodeOuter);
      lastTopNodeOuter = newOuterLoopNode;
      firstBotNodeOuter = newOuterLoopBranchNode;
    }
  }
  // Data Transfer : figure out bigDimToSize
  SenPcfgPtSFPDtNode* newDtNode = (SenPcfgPtSFPDtNode*)newPcfg.createPcfgNode(
      SenPcfgNode::Type::PTSFPDATATRANSFER);
  DtPair newDtPair;
  newDtPair.src_ = SenComponents::LX;
  newDtPair.dst_ = SenComponents::LX;

  newDtNode->setDestStAddr(-1);
  newDtNode->name = "c" + std::to_string(coreID) + "-" +
                    newPcfg.senComponentsToString.at(pcfgType) + "-FIFO-" +
                    newPcfg.senComponentsToString.at(newDtPair.src_) + "-" +
                    newPcfg.senComponentsToString.at(newDtPair.dst_) + "-" +
                    std::to_string(op->uniqueID);
  newDtNode->coreletId = 0;  // always use zero
  newDtNode->dtInfo = nullptr;
  newDtNode->srcDest = newDtPair;
  newDtNode->dsInfo = nullptr;
  newDtNode->useBurst = false;
  newDtNode->self = pcfgType;
  newDtNode->xrfRowIdx = -1;
  newDtNode->useImaOrFma = (op->inpLds->df == DataFormats::SENINT8 ||
                            op->inpLds->df == DataFormats::SENINT4);

  // set bigStAddrOffsets for each inner loop
  PcfgDtOffsets newDtOffset;
  init(newDtOffset);
  for (auto lpName : loopName)
    newDtNode->bigStAddrOffsets[lpName] = newDtOffset;

  // hook in NodeGraph
  if (lastTopNodeOuter != nullptr) {
    lastTopNodeOuter->next.push_back(newDtNode);
    newDtNode->prev.push_back(lastTopNodeOuter);
  }

  // hook in NodeGraph
  newDtNode->next.push_back(firstBotNodeOuter);
  firstBotNodeOuter->prev.push_back(newDtNode);
}

void DcgFE::findCoreRank(STCDPOpLx* op) {
  op->coreIDtoTrRank.clear();
  std::vector<int> pScore;
  std::vector<int> cScore;
  pScore.resize(maxNumCores, 0);
  cScore.resize(maxNumCores, 0);
  int maxConsumer = 0, maxProducer = 0;
  int maxCid = 0;
  int maxPId = 0;
  for (auto& dtTableEntry : op->dtTable_) {
    int pSpIdx = dtTableEntry.first;
    int pMemID = op->inpSP_[pSpIdx].placement.getMemId().at(0);
    bool extConsumerPresent = false;
    for (auto& cSpIdx : dtTableEntry.second.cIDXs) {
      const std::vector<int>& cMemIdVec =
          op->outSP_[cSpIdx].placement.getMemId();
      SenComponents cMemType = op->outSP_[cSpIdx].placement.Type();
      for (int idx = 0; idx < cMemIdVec.size(); idx++) {
        int cMemID = cMemIdVec[idx];
        if (pMemID != cMemID) {
          cScore.at(cMemID)++;
          extConsumerPresent = true;
          if (maxConsumer < cScore.at(cMemID)) {
            maxConsumer = cScore.at(cMemID);
            maxCid = cMemID;
          }
        }
      }
    }

    if (extConsumerPresent) {
      pScore.at(pMemID)++;
      if (maxProducer < pScore.at(pMemID)) {
        maxProducer = pScore.at(pMemID);
        maxPId = pMemID;
      }
    }
  }

  // initialize the rank

  // method 2:
  // std::vector<int> fScore = (maxConsumer > maxProducer) ? cScore : pScore;
  // int maxScore = (maxConsumer > maxProducer) ? maxConsumer : maxProducer;

  // method 3:
  // for (int i =0; i < maxNumCores; i++ )
  //	op->coreIDtoTrRank[i] = inserted++;

  // int maxID = (maxConsumer>maxProducer) ? maxCid : maxPId;
  // for (int i =maxID; i <maxNumCores; i++ )
  //	op->coreIDtoTrRank[i] = inserted++;
  // for (int i =0; i <maxID; i++ )
  //	op->coreIDtoTrRank[i] = inserted++;

  // method4
  // int maxScore = 0;
  // std::vector<int> fScore;
  // for (int i = 0; i < maxNumCores; i++) {
  //  fScore.push_back(cScore.at(i) + pScore.at(i));
  //  if (maxScore < fScore.back()) maxScore = fScore.back();
  //}

  // method5
  int maxScore = 0;
  int mulFactor = (maxConsumer < maxProducer) ? maxConsumer : maxProducer;
  bool isConsumerMain = (maxConsumer < maxProducer) ? false : true;
  std::vector<int> fScore;
  for (int i = 0; i < maxNumCores; i++) {
    fScore.push_back(cScore.at(i) + pScore.at(i) * maxConsumer);
    // if (isConsumerMain)
    //  fScore.push_back((cScore.at(i) + mulFactor) * mulFactor +
    //  pScore.at(i));
    // else
    //  fScore.push_back(cScore.at(i) + (pScore.at(i) + mulFactor) *
    //  mulFactor);
    if (maxScore < fScore.back()) maxScore = fScore.back();
  }

  std::map<int, std::set<int>> coreIdtoInferSegGp;
  for (const auto& segGropus : op->inferredSegGroups) {
    for (auto coreID : segGropus) coreIdtoInferSegGp[coreID] = segGropus;
  }

  bool trueSegmentNode = true;
  for (const auto& shareGropus : op->prodConsList) {
    DT_CHECK(coreIdtoInferSegGp.count(shareGropus.first));
    if (coreIdtoInferSegGp.at(shareGropus.first) != shareGropus.second)
      trueSegmentNode = false;
  }

  int inserted = 0;
  while (inserted < maxNumCores && maxScore >= 0) {
    std::deque<int> tassignList;
    for (int i = 0; i < maxNumCores; i++) {
      // if (maxScore == fScore.at(i)) op->coreIDtoTrRank[i] = inserted++;
      if (maxScore == fScore.at(i)) tassignList.push_back(i);
    }

    if (op->inferredSegGroups.size() == 1 || !trueSegmentNode) {
      int pingPong = 0;

      bool shouldpingPong = false;
      if (tassignList.size() > 2) {
        if (op->prodConsList.count(tassignList.front()) &&
            op->prodConsList.count(tassignList.back())) {
          shouldpingPong = true;
        }
      }

      while (tassignList.size()) {
        int coreID = pingPong == 0 ? tassignList.front() : tassignList.back();
        if (pingPong == 0)
          tassignList.pop_front();
        else
          tassignList.pop_back();

        op->coreIDtoTrRank[coreID] = inserted++;
        if (shouldpingPong) pingPong = (pingPong + 1) % 2;
      }
    } else {
      std::deque<int> tassignListResidue;
      std::map<std::set<int>, std::deque<int>> tassignListSorted;
      while (tassignList.size()) {
        int coreID = tassignList.front();
        if (coreIdtoInferSegGp.count(coreID)) {
          tassignListSorted[coreIdtoInferSegGp.at(coreID)].push_back(coreID);
        } else {
          tassignListResidue.push_back(coreID);
        }
        tassignList.pop_front();
      }

      int globalPingPong = 0;
      for (auto& mapkv : tassignListSorted) {
        int pingPong = globalPingPong;

        bool shouldpingPong = false;
        if (mapkv.second.size() > 2) {
          if (op->prodConsList.count(mapkv.second.front()) &&
              op->prodConsList.count(mapkv.second.back())) {
            shouldpingPong = true;
          }
        }

        while (mapkv.second.size()) {
          int coreID =
              pingPong == 0 ? mapkv.second.front() : mapkv.second.back();
          if (pingPong == 0)
            mapkv.second.pop_front();
          else
            mapkv.second.pop_back();

          op->coreIDtoTrRank[coreID] = inserted++;
          if (shouldpingPong) pingPong = (pingPong + 1) % 2;
        }
        if (shouldpingPong) globalPingPong = (globalPingPong + 1) % 2;
      }
      while (tassignListResidue.size()) {
        int coreID = tassignListResidue.front();
        tassignListResidue.pop_front();
        op->coreIDtoTrRank[coreID] = inserted++;
      }
    }
    maxScore--;
  }

  // take in account segCoreGroups
  for (auto& segCoreGp : op->segCoreGroups) {
    int maxCoreID = -1;
    int minCoreID = 100;
    for (auto& coreID : segCoreGp) {
      if (maxCoreID < coreID) maxCoreID = coreID;
      if (minCoreID > coreID) minCoreID = coreID;
    }
    int maxDist = maxCoreID - minCoreID;

    if (maxDist > segCoreGp.size()) {
      // DT_CHECK(maxDist == maxNumCores);
      //  add segment length : to find correct distance on the ring
      for (auto& coreID : segCoreGp)
        coreID = (coreID + segCoreGp.size()) % maxNumCores;
      std::sort(segCoreGp.begin(), segCoreGp.end());
      for (auto& coreID : segCoreGp) {
        coreID -= segCoreGp.size();
        if (coreID < 0) coreID += maxNumCores;
      }
    } else {
      std::sort(segCoreGp.begin(), segCoreGp.end());
    }

    std::vector<int> rankList;
    for (const auto coreID : segCoreGp)
      rankList.push_back(op->coreIDtoTrRank.at(coreID));
    std::sort(rankList.begin(), rankList.end());
    std::deque<int> clist;
    for (const auto coreID : segCoreGp) clist.push_back(coreID);

    int pingPong = 0;
    int count = 0;
    while (clist.size() && count < rankList.size()) {
      int coreID = pingPong == 0 ? clist.front() : clist.back();
      if (pingPong == 0)
        clist.pop_front();
      else
        clist.pop_back();

      op->coreIDtoTrRank.at(coreID) = rankList.at(count);
      count++;
      pingPong = (pingPong + 1) % 2;
    }
  }
}

void DcgFE::checkSegCoreProperties(STCDPOpLx* op) {
  std::map<std::set<int>, std::map<std::string, double>> segCoreGrpToDimSize;
  for (const auto& mapkv : op->prodConsList) {
    int pCoreID = mapkv.first;
    auto cCoreIDs = mapkv.second;

    // handle within LX checks
    if (cCoreIDs.size() == 1 && cCoreIDs.count(pCoreID))
      continue;  // within LX transfers

    bool found = false;
    for (const auto& segCoreGp : op->segCoreGroups) {
      if (DCGUtils::isValPresent(segCoreGp, pCoreID)) {
        DT_CHECK(!found);  // only part of one segment
        found = true;
        // make sure each consumer is also present
        for (auto cCoreID : cCoreIDs) {
          DT_CHECK(DCGUtils::isValPresent(segCoreGp, cCoreID));
        }
      }
    }

    if (found) {
      // each producer can only send one piece
      DT_CHECK(op->coreIDtoDtKey_L3SU.count(pCoreID));
      DT_CHECK(op->coreIDtoDtKey_L3SU.at(pCoreID).size() == 1);

      int inpSPIdx = op->coreIDtoDtKey_L3SU.at(pCoreID).back();
      if (segCoreGrpToDimSize.count(cCoreIDs))
        DT_CHECK(segCoreGrpToDimSize.at(cCoreIDs) ==
                 op->inpSP_.at(inpSPIdx).dimToSize_);
      else
        segCoreGrpToDimSize[cCoreIDs] = op->inpSP_.at(inpSPIdx).dimToSize_;
    }
  }

  // make sure in each segment first and last coreID are producer
  for (auto& segCoreGp : op->segCoreGroups) {
    DT_CHECK(op->prodConsList.count(segCoreGp.front()));
    DT_CHECK(op->prodConsList.count(segCoreGp.back()));
  }

  /*
  for (const auto& segCoreGp : op->segCoreGroups ) {
          DT_CHECK( segCoreGp.size());
          int minCoreID=100;
          int maxCoreID=-1;

          for (auto ccid : segCoreGp) {
                  if ( minCoreID > ccid)
                          minCoreID = ccid;
                  if ( maxCoreID < ccid)
                          maxCoreID = ccid;
          }

          DT_CHECK( minCoreID!= 100);
          DT_CHECK( maxCoreID!= -1);
          DT_CHECK( op->prodConsList.count(minCoreID));
          DT_CHECK( op->prodConsList.count(maxCoreID));
  }
  */
}

void DcgFE::splitDtTableEntriesForSegCoreGrps(STCDPOpLx* op) {
  for (auto& segCoreGp : op->segCoreGroups) {
    for (auto pCoreID : segCoreGp) {
      int segLength = segCoreGp.size();
      if (pCoreID == segCoreGp.front() || pCoreID == segCoreGp.back())
        continue;  // no replication

      // replicate data
      if (op->prodConsList.count(pCoreID)) {
        // handle within LX checks
        if (op->prodConsList.at(pCoreID).size() == 1 &&
            op->prodConsList.at(pCoreID).count(pCoreID))
          continue;  // within LX transfers

        // from CCW and CW consumer groups
        std::vector<int> ccwcIDXs;  // GT
        std::vector<int> cwcIDXs;   // LT
        int minIDCCW = -1;
        int minIDCW = -1;
        // shift the pCoreID to middle '16 in 32 cores'
        int idShift = (maxNumCores / 2) - pCoreID;
        if (idShift < 0) idShift += 32;

        int vpCoreID = (pCoreID + idShift) % maxNumCores;
        DT_CHECK(op->coreIDtoDtKey_L3SU.count(pCoreID));
        int inpSPIdx = op->coreIDtoDtKey_L3SU.at(pCoreID).back();

        for (auto outSPIdx : op->dtTable_.at(inpSPIdx).cIDXs) {
          int cCoreID = op->outSP_.at(outSPIdx).placement.getMemId().back();
          int vcCoreID = (cCoreID + idShift) % maxNumCores;
          if ((vcCoreID - vpCoreID) > 0) {
            ccwcIDXs.push_back(outSPIdx);
            if (minIDCCW == -1)
              minIDCCW = cCoreID;
            else if (minIDCCW > cCoreID)
              minIDCCW = cCoreID;
          } else {
            cwcIDXs.push_back(outSPIdx);
            if (minIDCW == -1)
              minIDCW = cCoreID;
            else if (minIDCW > cCoreID)
              minIDCW = cCoreID;
          }
        }

        if (ccwcIDXs.size() && cwcIDXs.size()) {
          // create new dtTableEntry
          op->inpSP_.push_back(op->inpSP_.at(inpSPIdx));
          int newInpSpIdx = op->inpSP_.size() - 1;
          op->dtTable_[newInpSpIdx] = op->dtTable_.at(inpSPIdx);
          op->dtTable_.at(newInpSpIdx).pIDX = newInpSpIdx;
          op->dtTable_.at(newInpSpIdx).cIDXs = cwcIDXs;
          op->dtTable_.at(newInpSpIdx).minCMemID = minIDCW;
          op->dtTable_.at(newInpSpIdx).preSegTrMode = 2;  // CW
          op->dtTable_.at(inpSPIdx).cIDXs = ccwcIDXs;
          op->dtTable_.at(inpSPIdx).minCMemID = minIDCCW;
          op->dtTable_.at(inpSPIdx).preSegTrMode = 1;  // CCW
        }
      }
    }
  }
}

void DcgFE::reoderSubPieceSegCores(STCDPOpLx* op) {
  for (auto& segCoreGp : op->segCoreGroups) {
    for (int idx = 0; idx < segCoreGp.size(); idx++) {
      auto pCoreID = segCoreGp.at(idx);
      int segLength = segCoreGp.back() - segCoreGp.front();
      if (pCoreID == segCoreGp.front() || pCoreID == segCoreGp.back())
        continue;  // no replication

      if (op->prodConsList.count(pCoreID) == 0) continue;
      DT_CHECK(op->coreIDtoDtKey_L3SU.at(pCoreID).size() <= 2);
      if (op->coreIDtoDtKey_L3SU.at(pCoreID).size() == 2) {
        int inpSPIdx1 = op->coreIDtoDtKey_L3SU.at(pCoreID).front();
        int inpSPIdx2 = op->coreIDtoDtKey_L3SU.at(pCoreID).back();
        int ccwSPIDX = (op->dtTable_.at(inpSPIdx1).preSegTrMode == 1)
                           ? inpSPIdx1
                           : inpSPIdx2;
        int cwSPIDX = (op->dtTable_.at(inpSPIdx1).preSegTrMode == 2)
                          ? inpSPIdx1
                          : inpSPIdx2;
        DT_CHECK(ccwSPIDX != cwSPIDX);
        op->coreIDtoDtKey_L3SU.at(pCoreID).clear();
        // re-order producer subpieces
        if (idx >= segCoreGp.size() / 2) {
          // CW preferred core
          op->coreIDtoDtKey_L3SU.at(pCoreID).push_back(cwSPIDX);
          op->coreIDtoDtKey_L3SU.at(pCoreID).push_back(ccwSPIDX);
        } else {
          // CCW preferred core
          op->coreIDtoDtKey_L3SU.at(pCoreID).push_back(ccwSPIDX);
          op->coreIDtoDtKey_L3SU.at(pCoreID).push_back(cwSPIDX);
        }
      }
    }

    // tabulate interleaving pieces
    int dtidx1 = op->coreIDtoDtKey_L3SU.at(segCoreGp.front()).front();
    int dtidx2 = op->coreIDtoDtKey_L3SU.at(segCoreGp.back()).front();
    DT_CHECK(op->interLeaveDtTableIdx.count(dtidx1) == 0);
    op->interLeaveDtTableIdx[dtidx1] = {dtidx1, dtidx2};
  }
}

void DcgFE::findInterleavingOpportunities(STCDPOpLx* op) {
  if (op->interLeaveDtTableIdx.size() == 0) {
    std::set<int> pIDXBooked;

    /*
    std::map<int,int> rankTocoreID;
    for (auto mapkv: op->coreIDtoTrRank )
            rankTocoreID[mapkv.second] =mapkv.first;

    std::vector<int> coreList;
    int firstPiecePCoreID =
    op->dtTable_.at(op->idealStWindowToDtKey.at(0).front()).pMemID; for ( int
    i=0; i < maxNumCores; i++) { if ( rankTocoreID.count(i) &&
    rankTocoreID.at(i)!=firstPiecePCoreID) {
                    coreList.push_back(rankTocoreID.at(i));
            }
    }
    coreList.push_back(firstPiecePCoreID);
    */

    // for ( int j=0; j < coreList.size(); j++ ) {
    //	int coreID = coreList.at(j);
    //	if (  op->coreIDtoDtKey_L3LU.count(coreID)==0)
    //		continue;
    //  auto& inpSPIdxVec = op->coreIDtoDtKey_L3LU.at(coreID);

    // still try to interleave
    std::set<int> cannot_interleave;
    bool can_interleave = false;
    for (int i = 0; i < 2;
         i++) {  // first iteration is preparing all cannot_interleave
      for (const auto& mapkv : op->coreIDtoDtKey_L3LU) {
        auto& inpSPIdxVec = mapkv.second;
        auto& coreID = mapkv.first;
        int idx = 0;
        while (idx < inpSPIdxVec.size()) {
          if (pIDXBooked.count(inpSPIdxVec.at(idx))) {
            idx++;
            continue;
          }
          if (idx + 1 < inpSPIdxVec.size()) {
            int firstSPIdx = inpSPIdxVec.at(idx);
            int secondSPIdx = inpSPIdxVec.at(idx + 1);
            bool canBeInterleaved = false;
            if (op->dtTable_.at(firstSPIdx).cMemIDs ==
                    op->dtTable_.at(secondSPIdx).cMemIDs &&
                op->dtTable_.at(firstSPIdx).selectedMCMode !=
                    op->dtTable_.at(secondSPIdx).selectedMCMode &&
                pIDXBooked.count(secondSPIdx) == 0 &&
                cannot_interleave.count(firstSPIdx) == 0 &&
                cannot_interleave.count(secondSPIdx) == 0) {
              // need to check if they are the first enrty in their respective
              // producer list
              int firstpMemID = op->dtTable_.at(firstSPIdx).pMemID;
              int secondpMemID = op->dtTable_.at(secondSPIdx).pMemID;
              DT_CHECK(op->coreIDtoDtKey_L3SU.at(firstpMemID).size());
              DT_CHECK(op->coreIDtoDtKey_L3SU.at(secondpMemID).size());
              if (op->coreIDtoDtKey_L3SU.at(firstpMemID)[0] == firstSPIdx &&
                  op->coreIDtoDtKey_L3SU.at(secondpMemID)[0] == secondSPIdx) {
                if (op->inpSP_.at(firstSPIdx).dimToSize_ ==
                        op->inpSP_.at(secondSPIdx).dimToSize_ &&
                    op->dtTable_.at(firstSPIdx).collapseFactor ==
                        op->dtTable_.at(secondSPIdx).collapseFactor &&
                    op->dtTable_.at(firstSPIdx).idealStWindow ==
                        op->dtTable_.at(secondSPIdx).idealStWindow - 1) {
                  canBeInterleaved = true;
                }
              }
            }

            if (canBeInterleaved) {
              can_interleave = true;
              idx = idx + 2;
              if (i == 1) {
                DT_CHECK(op->interLeaveDtTableIdx.count(firstSPIdx) == 0);
                op->interLeaveDtTableIdx[firstSPIdx] = {firstSPIdx,
                                                        secondSPIdx};
                pIDXBooked.insert(firstSPIdx);
                pIDXBooked.insert(secondSPIdx);
              }
            } else {
              cannot_interleave.insert(firstSPIdx);
              cannot_interleave.insert(secondSPIdx);
              idx++;
            }
          } else {
            idx++;
            break;
          }
        }
      }
      if (!can_interleave) break;
    }
  }
}

void DcgFE::computeProdCoreGpList(STCDPOpLx* op) {
  for (auto& dtTableEntry : op->dtTable_) {
    int pSpIdx = dtTableEntry.first;
    int pCoreID = dtTableEntry.second.pMemID;
    // create consumer
    std::set<int> cMemIds;
    for (auto& cIDX : dtTableEntry.second.cIDXs) {
      auto coreid = op->outSP_[cIDX].placement.getMemId().at(0);
      if (coreid != pCoreID) cMemIds.insert(coreid);
    }

    if (cMemIds.size() == 0) continue;

    if (op->prodCoreGpList.count(pCoreID) == 0) {
      op->prodCoreGpList[pCoreID].push_back(cMemIds);
    } else {
      // find the match
      bool found = false;
      for (int idx = 0; idx < op->prodCoreGpList.at(pCoreID).size(); idx++) {
        auto& cSets = op->prodCoreGpList.at(pCoreID).at(idx);
        for (auto& coreid : cMemIds)
          if (cSets.count(coreid)) {
            found = true;
            break;
          }
        if (found) {
          // insert
          for (auto& coreid : cMemIds) cSets.insert(coreid);
          break;
        }
      }
      if (!found) op->prodCoreGpList[pCoreID].push_back(cMemIds);
    }
  }

  // make sure Gps are exclusive
  for (auto& kv : op->prodCoreGpList) {
    std::set<int> eraseList;
    for (int idx1 = 1; idx1 < kv.second.size(); idx1++) {
      for (int idx2 = 0; idx2 < idx1; idx2++) {
        bool found = false;
        for (auto& coreid : kv.second.at(idx2)) {
          if (kv.second.at(idx1).count(coreid)) {
            found = true;
            break;
          }
        }

        if (found) {
          eraseList.insert(idx2);
          for (auto& coreid : kv.second.at(idx2))
            kv.second.at(idx1).insert(coreid);
          break;
        }
      }
    }

    // erase
    int idx = 0;
    for (auto it = kv.second.begin(); it != kv.second.end();) {
      if (eraseList.count(idx))
        it = kv.second.erase(it);
      else
        ++it;
    }
  }
}

void DcgFE::printProducerConsumerInfo(SuperDsc& sdsc, std::string fileName) {
  // find average & max consumer/producer
  int max_consumer = -1;  // for each producer
  int max_prod = -1;      // for each consumer

  int avg_consumer = 0;  // for each producer
  int avg_prod = 0;      // for each consumer

  int numProd = 0;
  int numCons = 0;

  int tRequestPerCons_max_old = -1;
  int tRequestPerCons_max_new = -1;
  int tRequestPerCons_avg_old = 0;
  int tRequestPerCons_avg_new = 0;
  int tResquest_div = 0;

  auto computeProdCon = [&](auto& prodConsList) {
    std::map<int, int> prodCount;
    for (auto& kv : prodConsList) {
      int consumerCount = kv.second.size();
      if (kv.second.count(kv.first)) consumerCount--;

      if (consumerCount > 0) {
        numProd++;
        avg_consumer += consumerCount;
      }

      max_consumer = std::max(max_consumer, consumerCount);
      // count producers
      for (auto cId : kv.second) {
        if (cId == kv.first) continue;

        if (prodCount.count(cId) == 0)
          prodCount[cId] = 1;
        else
          prodCount.at(cId)++;
      }
    }

    // compute max producer
    for (auto& kv : prodCount) {
      max_prod = std::max(kv.second, max_prod);

      if (kv.second > 0) {
        numCons++;
        avg_prod += kv.second;
      }
    }
  };

  if (sdsc.prodConsList.size()) {
    computeProdCon(sdsc.prodConsList);
  } else {
    // go through each data dscs
    for (auto& ddsc : sdsc.dataOpdscs_) {
      if (ddsc.opName == OpFuncs::STCDPOpLx) {
        auto op = (STCDPOpLx*)ddsc.op;
        computeProdCon(op->prodConsList);
      }
    }
  }

  // find chunk-level information for input neighbor fetch

  int max_prod_ch = -1;  // for across chunks
  int avg_prod_ch = 0;   // for across chunks
  int numChunks = 0;

  int maxMulticastD = -1;
  int avg_MulticastD = 0;
  int avg_MulticastD_c = 0;

  if (isInpFetchNeigh_) {
    DT_CHECK(sdsc.dataOpdscs_.size() > myIFNInfo_.dataDsc_Idx);
    DT_CHECK(myIFNInfo_.dataDsc_Idx >= 0);
    auto& ddsc = sdsc.dataOpdscs_.at(myIFNInfo_.dataDsc_Idx);
    DT_CHECK(ddsc.opName == OpFuncs::STCDPOpLx);
    auto op = (STCDPOpLx*)ddsc.op;

    for (auto& kv : op->coreIDtoDtKey_L3LU) {
      int coreID = kv.first;
      std::vector<std::vector<int>> chunkPieces;
      chunkPieces.push_back({});
      if (myIFNInfo_.cIDtoL3toLXSyncDtIdx.count(coreID) == 0) continue;
      int count = 0;
      for (auto& dtIdx : kv.second) {
        count++;
        chunkPieces.back().push_back(dtIdx);
        // check for sync boundary
        if (myIFNInfo_.cIDtoL3toLXSyncDtIdx.at(coreID).count(dtIdx)) {
          // last entry
          if (count < kv.second.size()) chunkPieces.push_back({});
        }
      }

      // go through each set and find max producer and average producer
      for (auto& vec : chunkPieces) {
        std::set<int> producers;
        for (auto& dtIdx : vec) {
          auto prodId = op->dtTable_.at(dtIdx).pMemID;
          producers.insert(prodId);
        }
        if (producers.size()) {
          numChunks++;
          avg_prod_ch += producers.size();
          max_prod_ch = std::max(max_prod_ch, (int)producers.size());
        }
      }

      // compute total request
      int trequest_new = 0;
      int trequest_old = 0;
      for (auto& dtIdx : kv.second) {
        DT_CHECK(op->dtTable_.at(dtIdx).collapseFactor > 0);
        auto dimToSize = op->inpSP_.at(dtIdx).dimToSize_;
        makeStickLevelAdjustments(dimToSize, op->inpLds);
        int numrequest_new = 1;
        int numrequest_old = 1;
        if (op->dtTable_.at(dtIdx).useBurst) {
          int totalSticks = 1;
          for (int i = 0; i < op->dtTable_.at(dtIdx).collapseFactor; i++) {
            auto dim = op->dtTable_.at(dtIdx).loopOrder.at(i);
            totalSticks *= dimToSize.at(dim);
          }

          if (totalSticks >= (int)sysDef.l3BurstSize) {
            // cannot use stride
            numrequest_new = 1;
            numrequest_old = ceil(totalSticks / sysDef.l3BurstSize);
            for (int i = op->dtTable_.at(dtIdx).collapseFactor;
                 i < op->dtTable_.at(dtIdx).loopOrder.size(); i++) {
              auto dim = op->dtTable_.at(dtIdx).loopOrder.at(i);
              numrequest_new *= dimToSize.at(dim);
              numrequest_old *= dimToSize.at(dim);
            }
          } else {
            // can use stride --> so skip
            // idx==op->dtTable_.at(dtIdx).collapseFactor
            for (int i = op->dtTable_.at(dtIdx).collapseFactor + 1;
                 i < op->dtTable_.at(dtIdx).loopOrder.size(); i++) {
              auto dim = op->dtTable_.at(dtIdx).loopOrder.at(i);
              numrequest_new *= dimToSize.at(dim);
              numrequest_old *= dimToSize.at(dim);
            }

            // stride dim will result in additional requests
            if (op->dtTable_.at(dtIdx).collapseFactor <
                op->dtTable_.at(dtIdx).loopOrder.size()) {
              auto dim = op->dtTable_.at(dtIdx).loopOrder.at(
                  op->dtTable_.at(dtIdx).collapseFactor);
              numrequest_old *= dimToSize.at(dim);
            }
          }
        } else {
          for (int i = 1; i < op->dtTable_.at(dtIdx).loopOrder.size(); i++) {
            auto dim = op->dtTable_.at(dtIdx).loopOrder.at(i);
            numrequest_new *= dimToSize.at(dim);
            numrequest_old *= dimToSize.at(dim);
          }
          auto firstDim = op->dtTable_.at(dtIdx).loopOrder.at(0);
          numrequest_old *= dimToSize.at(firstDim);
        }

        trequest_new += numrequest_new;
        trequest_old += numrequest_old;
      }
      DT_CHECK(trequest_new <= trequest_old);
      if (trequest_old > 0) {
        tRequestPerCons_max_old =
            std::max(tRequestPerCons_max_old, trequest_old);
        tRequestPerCons_max_new =
            std::max(tRequestPerCons_max_new, trequest_new);
        tRequestPerCons_avg_old += trequest_old;
        tRequestPerCons_avg_new += trequest_new;
        tResquest_div++;
      }
    }

    // find max multicast degree
    for (auto& kv : op->coreIDtoDtKey_L3SU) {
      for (auto& dtIdx : kv.second) {
        auto numSharers = op->dtTable_.at(dtIdx).myGTR.numSharers;
        if (numSharers > 0) {
          avg_MulticastD += numSharers;
          avg_MulticastD_c++;
          maxMulticastD = std::max(maxMulticastD, numSharers);
        }
      }
    }
  }

  if (numProd > 0 && numCons > 0) {
    avg_consumer = avg_consumer / numProd;
    avg_prod = avg_prod / numCons;

    std::ofstream outFile;
    char buf[20];
    outFile.open(fileName, std::ofstream::app);
    if (!outFile.is_open())
      DCGUtils::message("Error: Unable to open file=" + fileName);

    outFile << sdsc.name_ << ",";

    outFile << std::setw(10) << max_consumer << "," << std::setw(10)
            << avg_consumer << "," << std::setw(10) << max_prod << ","
            << std::setw(10) << avg_prod;

    if (isInpFetchNeigh_ && numChunks > 0) {
      avg_prod_ch /= numChunks;
      outFile << "," << std::setw(10) << max_prod_ch << std::setw(10) << ","
              << avg_prod_ch << ",";
      DT_CHECK(avg_MulticastD_c);
      avg_MulticastD /= avg_MulticastD_c;
      outFile << std::setw(10) << maxMulticastD << "," << std::setw(10)
              << avg_MulticastD << ",";

      DT_CHECK(tResquest_div > 0);
      tRequestPerCons_avg_old /= tResquest_div;
      tRequestPerCons_avg_new /= tResquest_div;
      // request counting
      outFile << std::setw(10) << tRequestPerCons_max_old << ","
              << std::setw(10) << tRequestPerCons_avg_old << ",";
      outFile << std::setw(10) << tRequestPerCons_max_new << ","
              << std::setw(10) << tRequestPerCons_avg_new;
    }
    outFile << std::endl;

    outFile.close();
  }
}

void DcgFE::printTrafficPerCore(SuperDsc& sdsc, std::string fileName) {
  std::map<int, int> coreIdToInpVol;
  std::map<int, int> coreIdToOutVol;
  std::map<int, int> coreIdToThroughVol;

  int totalProdSticks = 0;
  auto initCore = [&](int coreID) {
    if (coreIdToOutVol.count(coreID) == 0) coreIdToOutVol[coreID] = 0;
    if (coreIdToInpVol.count(coreID) == 0) coreIdToInpVol[coreID] = 0;
    if (coreIdToThroughVol.count(coreID) == 0) coreIdToThroughVol[coreID] = 0;
  };

  for (auto& ddsc : sdsc.dataOpdscs_) {
    if (ddsc.opName == OpFuncs::STCDPOpHBM) {
      for (int c = 0; c < 32; c++) initCore(c);
    }
    if (ddsc.opName == OpFuncs::STCDPOpLx ||
        ddsc.opName == OpFuncs::STCDPOpHBM) {
      auto op = (baseSTCDPOp*)ddsc.op;
      auto stickSize = 128.0 / op->inpLds->wordLength;
      // go through the list of coreIDs
      for (auto& kv : op->coreIDtoDtKey_L3SU) {
        // producer
        int coreID = kv.first;
        initCore(coreID);

        for (auto& dtIdx : kv.second) {
          int tot = 1;
          for (auto dim : op->inpSP_.at(dtIdx).dimToSize_) tot *= dim.second;
          coreIdToOutVol.at(coreID) += tot / stickSize;
          totalProdSticks += tot / stickSize;

          // find though traffic
          int CCW_hop = op->dtTable_.at(dtIdx).CCWHopCWHop.first;
          int CW_hop = op->dtTable_.at(dtIdx).CCWHopCWHop.second;

          if (op->dtTable_.at(dtIdx).selectedMCMode == 1 ||
              (op->dtTable_.at(dtIdx).selectedMCMode == 3 &&
               CCW_hop < 32 / 2)) {
            for (int c = 1; c < CCW_hop; c++) {
              int ccoreId = (c + coreID) / 32;
              initCore(ccoreId);
              coreIdToThroughVol.at(ccoreId) += tot / stickSize;
            }
          } else if (op->dtTable_.at(dtIdx).selectedMCMode == 2 ||
                     (op->dtTable_.at(dtIdx).selectedMCMode == 3 &&
                      CW_hop < 32 / 2)) {
            for (int c = 1; c < CW_hop; c++) {
              int ccoreId = (coreID - c);
              if (ccoreId < 0) ccoreId += 32;
              initCore(ccoreId);
              coreIdToThroughVol.at(ccoreId) += tot / stickSize;
            }
          } else {
            for (int c = 1; c < CCW_hop; c++) {
              int ccoreId = (c + coreID) / 32;
              initCore(ccoreId);
              coreIdToThroughVol.at(ccoreId) += (tot / 2.0) / stickSize;
            }
            for (int c = 1; c < CW_hop; c++) {
              int ccoreId = (coreID - c);
              if (ccoreId < 0) ccoreId += 32;
              initCore(ccoreId);
              coreIdToThroughVol.at(ccoreId) += (tot / 2.0) / stickSize;
            }
          }
        }
      }

      for (auto& kv : op->coreIDtoDtKey_L3LU) {
        // consumer
        int coreID = kv.first;
        initCore(coreID);
        for (auto& dtIdx : kv.second) {
          int tot = 1;
          for (auto dim : op->inpSP_.at(dtIdx).dimToSize_) tot *= dim.second;
          coreIdToInpVol.at(coreID) += tot / stickSize;
        }
      }

      // for hbm transfers
      for (auto& kv : op->dtTable_) {
        if (kv.second.pMemID == -1) {
          int tot = 1;
          for (auto dim : op->inpSP_.at(kv.first).dimToSize_) tot *= dim.second;
          for (int c = 0; c < 32; c++) {
            initCore(c);
            coreIdToThroughVol.at(c) += (tot / 2.0) / stickSize;
          }
        }
      }
    }
  }

  std::ofstream outFile;
  char buf[20];
  outFile.open(fileName);
  if (!outFile.is_open())
    DCGUtils::message("Error: Unable to open file=" + fileName);

  outFile << "---------------------------------------------------------------"
          << std::endl;
  outFile << std::setw(12) << "Core Id" << std::setw(20) << "Incoming Sticks"
          << std::setw(20) << "Outgoing Sticks" << std::setw(20)
          << "In/out Sticks" << std::setw(20) << "Through Sticks" << std::endl;
  outFile << std::setw(12) << std::setw(20) << std::setw(20) << std::setw(20)
          << std::endl;

  for (auto& kv : coreIdToInpVol) {
    DT_CHECK(coreIdToOutVol.count(kv.first));
    outFile << std::setw(12) << kv.first << std::setw(20) << kv.second
            << std::setw(20) << coreIdToOutVol.at(kv.first) << std::setw(20)
            << (coreIdToOutVol.at(kv.first) + kv.second) << std::setw(20)
            << (coreIdToThroughVol.at(kv.first)) << std::endl;
  }

  outFile << "totalProdSticks=" << totalProdSticks << "\n";
  outFile.close();
}

void DcgFE::computeSegCoreGroups(STCDPOpLx* op) {
  if (op->segCoreGroups.size()) return;
  if (sysDef.coreArch >= IsaCoreGen::RCUDD1A_ISA &&
      dscGlobal->psumRing == "sfpring")
    return;

  bool enSegCoreCopy = true;
  std::vector<std::vector<int>> segCoreGroups;
  // check if segments are contiguous
  for (auto& coreGrp : op->inferredSegGroups) {
    // check size: do it only for more than 2 segments
    if (coreGrp.size() >= maxNumCores / 2 || coreGrp.size() <= 2) {
      enSegCoreCopy = false;
      break;
    }

    bool enWrap = false;
    if (coreGrp.count(0) && coreGrp.count(maxNumCores - 1)) enWrap = true;
    std::unordered_set<int> coreGrpFast;
    int min = -1;
    int max = -1;
    for (auto& coreId : coreGrp) {
      int cid = coreId;
      if (enWrap && coreId > maxNumCores / 2) cid -= maxNumCores;

      if (min == -1) {
        min = cid;
        max = cid;
      } else {
        min = std::min(min, cid);
        max = std::max(max, cid);
      }
      coreGrpFast.insert(cid);
    }

    if (min + coreGrp.size() - 1 != max) {
      enSegCoreCopy = false;
      break;
    }

    // copy into segCoreGroups
    segCoreGroups.push_back({});
    segCoreGroups.back().push_back(min < 0 ? min + maxNumCores : min);
    for (int i = min + 1; i < max; i++) {
      if (coreGrpFast.count(i) == 0) {
        enSegCoreCopy = false;
        break;
      }
      segCoreGroups.back().push_back(i < 0 ? i + maxNumCores : i);
    }
    segCoreGroups.back().push_back(min < 0 ? max + maxNumCores : max);

    // make sure in each segment first and last coreID are producer
    if (enWrap) {
      if (min < 0) min += maxNumCores;
      if (max < 0) max += maxNumCores;
    }

    if (op->prodConsList.count(min) == 0) {
      enSegCoreCopy = false;
    } else {
      if (op->prodConsList.at(min).size() == 1 &&
          op->prodConsList.at(min).count(min)) {
        // only within LX transfer
        enSegCoreCopy = false;
      }
    }

    if (op->prodConsList.count(max) == 0) {
      enSegCoreCopy = false;
    } else {
      if (op->prodConsList.at(max).size() == 1 &&
          op->prodConsList.at(max).count(max)) {
        // only within LX transfer
        enSegCoreCopy = false;
      }
    }

    if (!enSegCoreCopy) break;
  }

  if (!enSegCoreCopy) return;
  // legality checks..
  std::map<std::set<int>, std::map<std::string, double>> segCoreGrpToDimSize;

  std::unordered_set<int> pcores;
  for (auto& dtTableEntry : op->dtTable_) {
    int inpSPIdx = dtTableEntry.first;
    int pMemID = op->inpSP_[inpSPIdx].placement.getMemId().at(0);
    if (pcores.count(pMemID)) {
      enSegCoreCopy = false;
      break;
    }
    pcores.insert(pMemID);

    auto& cCoreIDs = op->prodConsList.at(pMemID);
    if (segCoreGrpToDimSize.count(cCoreIDs)) {
      if (segCoreGrpToDimSize.at(cCoreIDs) !=
          op->inpSP_.at(inpSPIdx).dimToSize_) {
        enSegCoreCopy = false;
        break;
      }
    } else {
      segCoreGrpToDimSize[cCoreIDs] = op->inpSP_.at(inpSPIdx).dimToSize_;
    }
  }

  if (!enSegCoreCopy) return;

  // all conditional met
  op->segCoreGroups = segCoreGroups;
}

void DcgFE::transformToPcfgNBufferLXUnit(SenPcfg& myPcfg, STCDPOpHBM* op,
                                         int coreID, SenComponents pcfgType) {
  DT_CHECK(is_any_of(pcfgType, SenComponents::LXSU0, SenComponents::LXLU0));

  bool needConstRcvSync = false;
  if (op->name == OpFuncs::ResizeNNHBM) {
    const auto& genConstIntr = ((ResizeNNHBM*)op)->genConstIntr;
    if (genConstIntr) {
      DT_CHECK(pcfgType == SenComponents::LXSU0);
    }
  }

  std::vector<std::pair<std::vector<std::string>, std::vector<int>>>
      dtKeysPerInnerLOs = clusterDtKeysUsingInnerLOs(
          op->coreIDtoDtKey_LX.at(coreID), op, coreID,
          (pcfgType == SenComponents::LXLU0));

  SenPcfgNode* lastnode_top_sgraph = nullptr;
  SenPcfgNode* firstnode_bottom_sgraph = nullptr;
  SenPcfgNode* lastnode_bottom_sgraph = nullptr;

  DT_CHECK(dtKeysPerInnerLOs.size() == 1);
  const std::vector<std::string>& loopOrder = dtKeysPerInnerLOs.at(0).first;
  const std::vector<int>& dtKeys = dtKeysPerInnerLOs.at(0).second;

  const auto& amode = op->coreIDtoANInfo.at(coreID);

  std::vector<std::string>& layoutDimOrder_ = op->inpLds->isHbmPinned()
                                                  ? op->inpLds->layoutDimOrder_
                                                  : op->outLds->layoutDimOrder_;
  DT_CHECK(amode.loopCount.size() == layoutDimOrder_.size());
  std::map<std::string, double> stride_oloop;
  int t_loop_count = 1;
  for (int lpIdx = layoutDimOrder_.size() - 1; lpIdx >= 0; lpIdx--) {
    const auto& loopDimName = layoutDimOrder_.at(lpIdx);
    auto loop_name = "c" + std::to_string(coreID) + "-" +
                     myPcfg.senComponentsToString.at(pcfgType) + "-OL-" +
                     loopDimName + "-" + std::to_string(op->uniqueID);

    stride_oloop[loop_name] =
        amode.getAddrInfo().at(pcfgType)->getOffset(loopDimName);
    auto loop_be = createLoopAndBranchEnd(myPcfg, loop_name,
                                          amode.loopCount.at(loopDimName));

    t_loop_count *= amode.loopCount.at(loopDimName);
    if (lpIdx == layoutDimOrder_.size() - 1) {
      if (lastnode_top_sgraph == nullptr) {
        myPcfg.srcNode = loop_be.first;
      }
      lastnode_bottom_sgraph = loop_be.second;
    } else {
      lastnode_top_sgraph->next.push_back(loop_be.first);
      loop_be.first->prev.push_back(lastnode_top_sgraph);

      loop_be.second->next.push_back(firstnode_bottom_sgraph);
      firstnode_bottom_sgraph->prev.push_back(loop_be.second);
    }
    lastnode_top_sgraph = loop_be.first;
    firstnode_bottom_sgraph = loop_be.second;
  }

  DT_CHECK(t_loop_count == dtKeys.size());

  std::map<std::string, double> loop_count_il =
      op->inpSP_.at(dtKeys.at(0)).dimToSize_;
  for (int idx3 = 1; idx3 < dtKeys.size(); idx3++) {
    DT_CHECK(loop_count_il == op->inpSP_.at(dtKeys.at(0)).dimToSize_);
  }

  makeStickLevelAdjustments(loop_count_il, (pcfgType == SenComponents::LXLU0)
                                               ? op->inpLds
                                               : op->outLds);

  std::map<std::string, std::string> dimToLoopName;
  for (int lpIdx = loopOrder.size() - 1; lpIdx >= 0; lpIdx--) {
    const auto& loopDimName = loopOrder.at(lpIdx);
    auto loop_name = "c" + std::to_string(coreID) + "-" +
                     myPcfg.senComponentsToString.at(pcfgType) + "-IL-" +
                     loopDimName + "-" + std::to_string(op->uniqueID);

    dimToLoopName[loopDimName] = loop_name;
    auto loop_be = createLoopAndBranchEnd(myPcfg, loop_name,
                                          loop_count_il.at(loopDimName));

    lastnode_top_sgraph->next.push_back(loop_be.first);
    loop_be.first->prev.push_back(lastnode_top_sgraph);

    loop_be.second->next.push_back(firstnode_bottom_sgraph);
    firstnode_bottom_sgraph->prev.push_back(loop_be.second);
    lastnode_top_sgraph = loop_be.first;
    firstnode_bottom_sgraph = loop_be.second;
  }

  // take care of ResizeNNHBM op
  if (op->name == OpFuncs::ResizeNNHBM) {
    baseSTCDPOp* myOpResize = (baseSTCDPOp*)op;
    std::map<std::string, int> upSizeFactor;
    int dimReqUpsize = 0;
    if (op->name == OpFuncs::ResizeNNHBM) {
      ResizeNNHBM* myOptemp = (ResizeNNHBM*)op;
      upSizeFactor = myOptemp->upSizeFactor;
    } else {
      ResizeNNLX* myOptemp = (ResizeNNLX*)op;
      upSizeFactor = myOptemp->upSizeFactor;
    }
    int totalUpSizeFactor = 1;
    for (auto& mapkv : upSizeFactor) {
      if (mapkv.second > 1) dimReqUpsize++;
      totalUpSizeFactor *= mapkv.second;
    }

    if (pcfgType == SenComponents::LXLU0) {
      SenPcfgMvloopNode* newMvLoopNode =
          (SenPcfgMvloopNode*)myPcfg.createPcfgNode(SenPcfgNode::Type::MVLOOP);
      std::string resizeLoopName = "c" + std::to_string(coreID) + "-" +
                                   myPcfg.senComponentsToString.at(pcfgType) +
                                   "-IL-resizeCollapsed-" +
                                   std::to_string(op->uniqueID);
      newMvLoopNode->name = resizeLoopName;
      newMvLoopNode->loopName = resizeLoopName;
      newMvLoopNode->loopCount = totalUpSizeFactor;

      // inner loop end
      SenPcfgMvloopBranchNode* newLoopBranchNode =
          (SenPcfgMvloopBranchNode*)myPcfg.createPcfgNode(
              SenPcfgNode::Type::MVLOOPBRANCH);

      newLoopBranchNode->name = "c" + std::to_string(coreID) + "-" +
                                myPcfg.senComponentsToString.at(pcfgType) +
                                "-ILBranch-resizeCollapsed-" +
                                std::to_string(op->uniqueID);
      newLoopBranchNode->loopNode = newMvLoopNode;
      newLoopBranchNode->next.push_back(newMvLoopNode);

      // hook in NodeGraph
      newMvLoopNode->prev.push_back(lastnode_top_sgraph);
      lastnode_top_sgraph->next.push_back(newMvLoopNode);
      firstnode_bottom_sgraph->prev.push_back(newLoopBranchNode);
      newLoopBranchNode->next.push_back(firstnode_bottom_sgraph);

      lastnode_top_sgraph = newMvLoopNode;
      firstnode_bottom_sgraph = newLoopBranchNode;
    } else if (pcfgType == SenComponents::LXSU0) {
      for (int idx_r = loopOrder.size() - 1; idx_r >= 0; idx_r--) {
        const auto& loopDimName = loopOrder[idx_r];
        DT_CHECK(upSizeFactor.count(loopDimName));

        if (upSizeFactor.at(loopDimName) == 1) continue;

        SenPcfgMvloopNode* newMvLoopNode =
            (SenPcfgMvloopNode*)myPcfg.createPcfgNode(
                SenPcfgNode::Type::MVLOOP);
        std::string tag = loopDimName;
        std::string resizeLoopName = "c" + std::to_string(coreID) + "-" +
                                     myPcfg.senComponentsToString.at(pcfgType) +
                                     "-IL-resizeL-" + tag + "-" +
                                     std::to_string(op->uniqueID);
        newMvLoopNode->name = resizeLoopName;
        newMvLoopNode->loopName = resizeLoopName;
        newMvLoopNode->loopCount = upSizeFactor.at(loopDimName);

        // inner loop end
        SenPcfgMvloopBranchNode* newLoopBranchNode =
            (SenPcfgMvloopBranchNode*)myPcfg.createPcfgNode(
                SenPcfgNode::Type::MVLOOPBRANCH);

        newLoopBranchNode->name = "c" + std::to_string(coreID) + "-" +
                                  myPcfg.senComponentsToString.at(pcfgType) +
                                  "-ILBranch-resize-" + tag + "-" +
                                  std::to_string(op->uniqueID);
        newLoopBranchNode->loopNode = newMvLoopNode;
        newLoopBranchNode->next.push_back(newMvLoopNode);

        // hook in NodeGraph
        newMvLoopNode->prev.push_back(lastnode_top_sgraph);
        lastnode_top_sgraph->next.push_back(newMvLoopNode);
        firstnode_bottom_sgraph->prev.push_back(newLoopBranchNode);
        newLoopBranchNode->next.push_back(firstnode_bottom_sgraph);

        lastnode_top_sgraph = newMvLoopNode;
        firstnode_bottom_sgraph = newLoopBranchNode;
      }
    }
  }

  // Data Transfer
  std::map<std::string, double> bigDimToSize;
  if (pcfgType == SenComponents::LXSU0) {
    int outSPIDx = getIdxForMatchingCMenID(
        op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
    bigDimToSize = op->outSP_.at(outSPIDx).bigDimToSize_;
  } else {
    bigDimToSize = op->inpSP_.at(dtKeys.at(0)).bigDimToSize_;
  }

  // check if all biDimToSize are same..
  for (int idx3 = 1; idx3 < dtKeys.size(); idx3++) {
    int inpSPIDx = dtKeys.at(idx3);
    int outSPIDx;
    if (pcfgType == SenComponents::LXSU0) {
      outSPIDx = getIdxForMatchingCMenID(
          op->outSP_, op->dtTable_.at(inpSPIDx).cIDXs, coreID);
      DT_CHECK(bigDimToSize == op->outSP_.at(outSPIDx).bigDimToSize_);
    } else {
      DT_CHECK(op->inpSP_.size() > inpSPIDx);
      if (!(op->name == OpFuncs::ResizeNNHBM ||
            op->name == OpFuncs::ResizeNNLX))
        DT_CHECK(bigDimToSize == op->inpSP_.at(inpSPIDx).bigDimToSize_);
    }
  }

  SenPcfgDtNode* newDtNode =
      (SenPcfgDtNode*)myPcfg.createPcfgNode(SenPcfgNode::Type::DATATRANSFER);
  DtPair newDtPair;
  if (pcfgType == SenComponents::LXSU0) {
    // LXLUSUFIFO transfer: to LX
    newDtPair.src_ = op->useLXSFPLXTransfers ? SenComponents::PE0
                                             : SenComponents::LXLUSUFIFO;
    newDtPair.dst_ = SenComponents::LX;
  } else {
    // LXLUSUFIFO transfer: from LX
    newDtPair.src_ = SenComponents::LX;
    newDtPair.dst_ = op->useLXSFPLXTransfers ? SenComponents::PE0
                                             : SenComponents::LXLUSUFIFO;
  }

  newDtNode->name = "c" + std::to_string(coreID) + "-" +
                    myPcfg.senComponentsToString.at(pcfgType) + "-dt-" +
                    myPcfg.senComponentsToString.at(newDtPair.src_) + "-" +
                    myPcfg.senComponentsToString.at(newDtPair.dst_) + "-" +
                    std::to_string(op->uniqueID);
  newDtNode->coreletId = -1;  // corelet independent
  newDtNode->dtInfo = nullptr;
  newDtNode->srcDest = newDtPair;
  newDtNode->dsInfo = nullptr;
  newDtNode->dimLayoutOrder = (pcfgType == SenComponents::LXLU0)
                                  ? op->inpLds->layoutDimOrder_
                                  : op->outLds->layoutDimOrder_;
  newDtNode->myBigDimSize = bigDimToSize;
  makeStickLevelAdjustments(
      newDtNode->myBigDimSize,
      (pcfgType == SenComponents::LXLU0) ? op->outLds : op->inpLds);

  // set start Address
  if (pcfgType == SenComponents::LXLU0) {
    fillAddr(newDtNode->SrcStartAddr(),
             op->inpSP_.at(dtKeys.at(0)).placement.StartAddr(), 0);
  } else {
    int outSPIDx = getIdxForMatchingCMenID(
        op->outSP_, op->dtTable_.at(dtKeys.at(0)).cIDXs, coreID);
    fillAddr(newDtNode->DestStartAddr(),
             op->outSP_.at(outSPIDx).placement.StartAddr(), 0);
  }

  // add burst Info, used in SenProg
  newDtNode->useBurst = true;
  if (op->name == OpFuncs::ResizeNNHBM) {
    newDtNode->useBurst = false;
  }

  for (int idx3 = 0; idx3 < dtKeys.size(); idx3++) {
    int inpSPIDx = dtKeys.at(idx3);
    if (!op->dtTable_.at(inpSPIDx).useBurst) newDtNode->useBurst = false;
  }

  // no burst for now, add later..
  for (auto& myDim : newDtNode->myBigDimSize) {
    newDtNode->myLitDimSize[myDim.first] = 1;
  }

  // set bigStAddrOffsets for each inner loop
  for (int lpIdx = loopOrder.size() - 1; lpIdx >= 0; lpIdx--) {
    const auto& loopDimName = loopOrder.at(lpIdx);
    PcfgDtOffsets newDtOffset;
    init(newDtOffset);

    // find location in dimLayoutOrder
    for (auto& dimName : newDtNode->dimLayoutOrder) {
      if (dimName == loopDimName) {
        break;
      } else {
        newDtOffset.dimOffset *= newDtNode->myBigDimSize.at(dimName);
      }
    }

    // special case: ResizeNNHBM Op: Insert another conditional
    if (pcfgType == SenComponents::LXSU0 && op->name == OpFuncs::ResizeNNHBM) {
      std::map<std::string, int> upSizeFactor;
      ResizeNNHBM* myOptemp = (ResizeNNHBM*)op;
      upSizeFactor = myOptemp->upSizeFactor;

      baseSTCDPOp* myOpResize = (baseSTCDPOp*)op;
      const auto& loopDimName = loopOrder[lpIdx];
      DT_CHECK(upSizeFactor.count(loopDimName));
      std::string tag = loopDimName;
      if (upSizeFactor.at(loopDimName) > 1) {
        std::string resizeLoopName = "c" + std::to_string(coreID) + "-" +
                                     myPcfg.senComponentsToString.at(pcfgType) +
                                     "-IL-resizeL-" + tag + +"-" +
                                     std::to_string(op->uniqueID);
        newDtNode->bigStAddrOffsets[resizeLoopName] = newDtOffset;
        // modify offset for regular case
        newDtOffset.dimOffset *= upSizeFactor.at(loopDimName);
      }
    }

    newDtNode->bigStAddrOffsets[dimToLoopName[loopDimName]] = newDtOffset;
  }

  PcfgDtOffsets newDtOffset;
  init(newDtOffset);
  for (auto kv : stride_oloop) {
    newDtOffset.dimOffset = kv.second;
    newDtNode->bigStAddrOffsets[kv.first] = newDtOffset;
  }

  // hook in NodeGraph
  lastnode_top_sgraph->next.push_back(newDtNode);
  newDtNode->prev.push_back(lastnode_top_sgraph);
  lastnode_top_sgraph = newDtNode;

  // hook in NodeGraph
  lastnode_top_sgraph->next.push_back(firstnode_bottom_sgraph);
  firstnode_bottom_sgraph->prev.push_back(lastnode_top_sgraph);

  lastnode_top_sgraph = lastnode_bottom_sgraph;
}

template <typename Dtype>
void DcgFE::fillMemIdAddr(FoldManager<Dtype>& lhs, FoldManager<Dtype>& rhs,
                          int idx) {
  if (rhs.hasZeroFoldDim()) {
    lhs.insertData({rhs.getData().at(idx)});
  } else {
    auto foldDimProp = rhs.getDimProp();
    for (auto& [dim_ptr, foldFunc] : foldDimProp) {
      DT_CHECK(is_any_of(foldFunc, BaseFuncType::Constant, BaseFuncType::Map));
    }

    lhs.buildFoldSpace(foldDimProp);
    lhs.apply(rhs, {},
              [&](auto&& l, auto&& r) -> Dtype { return {r.at(idx)}; });
  }
}

void DcgFE::fillAddr(FoldManager<int64_t>& lhs,
                     const FoldManager<std::vector<int64_t>>& rhs, int idx) {
  if (rhs.hasZeroFoldDim()) {
    auto data = rhs.getData();
    DT_CHECK(idx < data.size());
    lhs.insertData(data.at(idx));
  } else {
    auto foldDimProp = rhs.getDimProp();
    for (auto& [dim_ptr, foldFunc] : foldDimProp) {
      DT_CHECK(is_any_of(foldFunc, BaseFuncType::Constant, BaseFuncType::Map));
    }

    lhs.buildFoldSpace(foldDimProp);
    lhs.apply(rhs, {}, [&](auto&& l, auto&& r) {
      DT_CHECK(idx < r.size());
      return r[idx];
    });
  }
}
