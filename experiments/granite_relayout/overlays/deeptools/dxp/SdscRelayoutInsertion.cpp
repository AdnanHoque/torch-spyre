/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2026
 ************************************************************/

#include <dsc/dataOpDsc.h>
#include <dsc/superdsc.h>
#include <sharedtools/mem_track_bundle.h>

#include <cstdlib>
#include <set>
#include <string>

#include "dxp.h"

#define DEBUG_RELAYOUT 1

static double getRelayoutPieceSize(
    double form_size,
    const std::map<int, std::map<PrimaryDimTypes, int>>& coreIdToWkSlice,
    const std::set<PrimaryDimTypes>& tensorDims) {
  double num_pieces = 1;
  std::map<PrimaryDimTypes, std::set<int>> wkslices_in_dim;
  for (const auto& core_to_slice : coreIdToWkSlice) {
    for (const auto& dim_to_slice : core_to_slice.second) {
      if (tensorDims.count(dim_to_slice.first) == 0) continue;
      wkslices_in_dim[dim_to_slice.first].insert(dim_to_slice.second);
    }
  }
  for (const auto& dim_to_slices : wkslices_in_dim) {
    num_pieces *= dim_to_slices.second.size();
  }
  return std::ceil(form_size / num_pieces);
}

static bool hasSameWorkSlicePartition(
    const std::map<int, std::map<PrimaryDimTypes, int>>& lhs,
    const std::map<int, std::map<PrimaryDimTypes, int>>& rhs) {
  std::multiset<std::map<PrimaryDimTypes, int>> lhsSlices;
  std::multiset<std::map<PrimaryDimTypes, int>> rhsSlices;
  for (const auto& [coreId, workSlice] : lhs) {
    lhsSlices.insert(workSlice);
  }
  for (const auto& [coreId, workSlice] : rhs) {
    rhsSlices.insert(workSlice);
  }
  return lhsSlices == rhsSlices;
}

void Dxp::insertRelayoutSdsc(SuperDsc* sdsc, int ps,
                             MemTrackBundle* memTrackers,
                             std::vector<SuperDsc*>& relayout_sdscs,
                             LdsSegment dxpSeg) {
  auto& lxTrackPerCore = memTrackers->lxTrackPerCore;
  auto& hbmTrack = memTrackers->hbmTrack.at(dxpSeg);
  auto& dldsc = sdsc->dscs_.at(0);
  bool isDirectOutCopyEligible = false;
  bool directOutCopyWasReplaced = false;
  if (dldsc.computeOp_.size() > 0) {
    // Direct out-copy is only meaningful for a single-input/single-output
    // IDENTITY/SHUFFLE (2 labeledDs). Ops such as scatter can also be tagged
    // IDENTITY/SHUFFLE but carry additional inputs; those fall through to the
    // regular relayout path rather than the direct-copy optimization.
    if ((dldsc.computeOp_.front().opFuncName == OpFuncs::IDENTITY ||
         dldsc.computeOp_.front().opFuncName == OpFuncs::SHUFFLE) &&
        dldsc.labeledDs_.size() == 2) {
      auto& inpDs = dldsc.labeledDs_.front();
      auto& outDs = dldsc.labeledDs_.back();
      if (inpDs.dsType_ == outDs.dsType_ && inpDs.scale_ == outDs.scale_ &&
          inpDs.density_ == outDs.density_) {
        auto isInpLx =
            (inpDs.memOrg_.count(SenComponents::LX) &&
             inpDs.memOrg_.at(SenComponents::LX).allocateNode_ != nullptr);
        auto isOutLx =
            (outDs.memOrg_.count(SenComponents::LX) &&
             outDs.memOrg_.at(SenComponents::LX).allocateNode_ != nullptr);
        if (isInpLx && isOutLx) {
          isDirectOutCopyEligible = true;
        }
      }
    }
  }
  for (int iter = 0; iter < dldsc.labeledDs_.size() - 1; iter++) {
    auto& lds = dldsc.labeledDs_.at(iter);
    // Step 1: For each input lds, check if relayout is needed. Relayout
    // is needed if (1) lds is lx pinned, and (2) lds layout does not match
    // compute distribution
    if (lds.pinnedComponent() == HBM || lds.pinnedComponent() == NO_COMPONENT)
      continue;

    auto pre_relayout_allocNode = lds.memOrg_.at(LX).allocateNode_;
    auto& allocCoords = lds.memOrg_.at(LX).allocateNode_->allocateCoordinates_;
    if (allocCoords.coreIdToWkSlice_.empty() ||
        allocCoords.coreIdToWkSlice_ == sdsc->coreIdToWkSlice_)
      continue;

#ifdef DEBUG_RELAYOUT
    std::cout << "Inserting relayout for SDSC " << sdsc->name_ << ": "
              << lds.dsName_ << std::endl;
#endif

    // Step 2a: Create new sdsc for the relayout, and copy relevant properties
    // from the consumer sdsc
    SuperDsc* relayout_sdsc = new SuperDsc;
    relayout_sdsc->name_ = sdsc->name_ + "-Relayout";
    relayout_sdsc->sdscFoldProps_.emplace_back(std::make_unique<FoldDimProp>(
        sdsc->sdscFoldProps_.at(0)->getSize(), "time"));
    relayout_sdsc->sdscFolds_.clone(sdsc->sdscFolds_);
    relayout_sdsc->coreFoldProp_ =
        std::make_unique<FoldDimProp>(sdsc->coreFoldProp_->getSize(), "core");
    relayout_sdsc->coreletFoldProp_ = std::make_unique<FoldDimProp>(
        sdsc->coreletFoldProp_->getSize(), "corelet");
    relayout_sdsc->unpadN_ = sdsc->unpadN_;
    relayout_sdsc->N_ = sdsc->N_;
    relayout_sdsc->target_ = sdsc->target_;

    // Step 2b: Insert the relayout program step and check if there is
    // sufficient contiguous space in LX for the post-relayout form. This must
    // use the same tracker state as the eventual allocation below; a pure free
    // capacity check before the inserted step can pass even when no common
    // contiguous block is available at the real relayout step.
    memTrackers->insertPsBeforeAndRename(ps);
    auto out_form_size = lds.wordLength;
    auto N_ = sdsc->dscs_.front().N_;
    auto dim_count = 0;
    std::set<PrimaryDimTypes> relayout_tensor_dims;
    for (auto dimid :
         sdsc->dscs_.front().primaryDsInfo_.at(lds.dsType_).layoutDimOrder_) {
      relayout_tensor_dims.insert(dimid);
      if (lds.scale_[dim_count] == -2)
        out_form_size *= 64;
      else if (lds.scale_[dim_count] == -1)
        out_form_size *= 1;
      else
        out_form_size *= N_.primaryDimToVal_st(dimid);
      dim_count++;
    }
    auto out_piece_size = getRelayoutPieceSize(
        out_form_size, sdsc->coreIdToWkSlice_, relayout_tensor_dims);

    bool lx_space_found = true;
    std::map<int, int64_t> coreIdToAllocAddr;
    if (!isDirectOutCopyEligible) {
      std::vector<int> exPhases = {ps, ps + 1};
      for (const auto& mapkv : sdsc->coreIdToWkSlice_) {
        auto& memTracker = lxTrackPerCore.at(mapkv.first);
        auto addr = memTracker.checkAndAddDs(lds.dsName_, "dynamic",
                                             out_piece_size, exPhases);
        DT_CHECK(addr != EXISTS);
        if (addr != DOESNT_FIT) {
          coreIdToAllocAddr[mapkv.first] = addr;
        } else {
          lx_space_found = false;
          break;
        }
      }
      if (!lx_space_found) {
        for (auto& mapkv : coreIdToAllocAddr) {
          auto& memTracker = lxTrackPerCore.at(mapkv.first);
          memTracker.removeDs(lds.dsName_, "dynamic", exPhases);
        }
      }
    } else {
      DT_CHECK(iter == 0);
      auto& outDs = dldsc.labeledDs_.back();
      auto& outAllocNode = outDs.memOrg_.at(SenComponents::LX).allocateNode_;
      for (const auto& mapkv : sdsc->coreIdToWkSlice_) {
        coreIdToAllocAddr[mapkv.first] =
            outAllocNode->startAddressCoreCorelet_.getSingleData(
                {{0, mapkv.first}});
      }
    }

    if (lx_space_found) {
#ifdef DEBUG_RELAYOUT
      std::cout << "Lx space found, inserting stcdpLx" << std::endl;
#endif
      // Step 3-lx: Lx space has been found for the post-relayout form, so
      // relayout will be achieved through stcdpLx. Create a new datadsc for the
      // relayout node
      auto& ddsc = relayout_sdsc->dataOpdscs_.emplace_back();
      ddsc.dscName_ = sdsc->name_ + "-LxRelayout";
      ddsc.opName = OpFuncs::STCDPOpLx;
      ddsc.op = new STCDPOpLx;
      ddsc.op->name = OpFuncs::STCDPOpLx;
      ddsc.primaryDs_["activation"].name_ = "activation";
      ddsc.target_ = sdsc->target_;
      FoldDimProp* x =
          new FoldDimProp(sdsc->sdscFoldProps_.at(0)->getSize(), "time");
      ddsc.foldProps_.push_back(x);

      // Step 3a-lx: Create new input and output lds for the relayout node, and
      // fill their properties
      ddsc.labeledDs_.resize(2);
      auto& inpLds = ddsc.labeledDs_.at(0);
      auto& outLds = ddsc.labeledDs_.at(1);
      outLds.ldsName_ = lds.dsName_ + "-LxRelayout-out";
      inpLds.ldsName_ = lds.dsName_ + "-LxRelayout-inp";

      // fill coreIDsUsed for the datadsc
      std::set<int> cores_seen;
      for (const auto& mapkv : sdsc->coreIdToWkSlice_) {
        if (cores_seen.count(mapkv.first) == 0) {
          ddsc.coreIdsUsed_.push_back(mapkv.first);
          cores_seen.insert(mapkv.first);
        }
      }
      for (const auto& mapkv : allocCoords.coreIdToWkSlice_) {
        if (cores_seen.count(mapkv.first) == 0) {
          ddsc.coreIdsUsed_.push_back(mapkv.first);
          cores_seen.insert(mapkv.first);
        }
      }
      relayout_sdsc->numCoresUsed_ = cores_seen.size();

      // fill coreIdToDscSchedule
      for (auto coreid : cores_seen) {
        DscScheduleStep schedule(0, -1, 0, 0);
        relayout_sdsc->coreIdToDscSchedule[coreid].push_back(schedule);
      }

      // fill ldsInfo for the input and output lds
      for (int i = 0; i < 2; i++) {
        auto& myLds = (i == 0) ? outLds : inpLds;
        myLds.pdsName_ = "activation";
        myLds.wordLength = lds.wordLength;
        myLds.df = lds.dataFormat_;
        auto& refDsInfo = dldsc.primaryDsInfo_.at(lds.dsType_);

        // Fill dimNames, layoutDimOrder_, dimToLayoutSize_ and validGap_
        auto& N_ = dldsc.N_;
        auto refLdo = pre_relayout_allocNode->layoutDimOrder_;
        if (isDirectOutCopyEligible) {
          auto& outDs = dldsc.labeledDs_.back();
          auto& outAllocNode =
              outDs.memOrg_.at(SenComponents::LX).allocateNode_;
          refLdo = outAllocNode->layoutDimOrder_;
        }
        for (auto& pDim : refLdo) {
          DT_CHECK(pDim != PrimaryDimTypes::IJ && pDim != PrimaryDimTypes::KIJ);
          std::string dimName = dldsc.primaryDimToString.at(pDim);
          myLds.layoutDimOrder_.push_back(dimName);
          if (i == 0) {
            ddsc.primaryDs_.at("activation").dimNames.insert(dimName);
          } else {
            DT_CHECK(ddsc.primaryDs_.at("activation").dimNames.count(dimName));
          }
          myLds.dimToLayoutSize_[dimName] =
              N_.paramNameToVal(dldsc.primaryDimToString.at(pDim));
          myLds.validGap_[dimName];
          myLds.validGap_.at(dimName).push_back(
              std::make_pair(myLds.dimToLayoutSize_.at(dimName), 0));
        }

        // Fill stickDimOrder_ and stickSize_
        DT_CHECK(refDsInfo.stickDimOrder_.size() ==
                 refDsInfo.stickSize_.size());
        DT_CHECK(refDsInfo.stickDimOrder_.size() ==
                 refDsInfo.stickRepl_.size());
        for (int idx = 0; idx < refDsInfo.stickDimOrder_.size(); idx++) {
          auto& pDim = refDsInfo.stickDimOrder_.at(idx);
          DT_CHECK(refDsInfo.stickRepl_.at(idx) == 1);  // no repeats allowed
          std::string dimName;
          if (pDim == PrimaryDimTypes::IJ) {
            DT_CHECK(0);
          } else if (pDim == PrimaryDimTypes::KIJ) {
            DT_CHECK(0);
          } else {
            dimName = dldsc.primaryDimToString.at(pDim);
          }

          if (std::find(myLds.stickDimOrder_.begin(),
                        myLds.stickDimOrder_.end(),
                        dimName) == myLds.stickDimOrder_.end()) {
            myLds.stickDimOrder_.push_back(dimName);
            myLds.dimToStickSize_[dimName] = refDsInfo.stickSize_.at(idx);
          } else {
            myLds.dimToStickSize_.at(dimName) *= refDsInfo.stickSize_.at(idx);
          }
          DT_CHECK(ddsc.primaryDs_.at("activation").dimNames.count(dimName));
        }
      }

      // Fill totalElements_
      for (int i = 0; i < 2; i++) {
        auto& myLds = (i == 0) ? outLds : inpLds;
        double totElements = 1;
        for (auto kv : myLds.dimToLayoutSize_) {
          totElements *= kv.second;
        }
        myLds.totElements = totElements;
      }
      ddsc.dimPool_ = ddsc.primaryDs_.at("activation").dimNames;
      DT_CHECK(outLds.stickDimOrder_ == inpLds.stickDimOrder_);
      DT_CHECK(outLds.dimToStickSize_ == inpLds.dimToStickSize_);

      // Step 3b-lx: Create input and output pieces
      for (int i = 0; i < 2; i++) {
        auto& myLds = (i == 0) ? outLds : inpLds;
        int piece_count = 0;
        // Find the CoreD for each dim. This is given by N_ / total number of
        // work slices created
        std::map<PrimaryDimTypes, std::set<int>> wkslices_in_dim;
        for (auto kv :
             (i == 0) ? sdsc->coreIdToWkSlice_ : allocCoords.coreIdToWkSlice_) {
          for (auto kv2 : kv.second) {
            if (relayout_tensor_dims.count(kv2.first) == 0) continue;
            wkslices_in_dim[kv2.first].insert(kv2.second);
          }
        }

        for (auto kv :
             (i == 0) ? sdsc->coreIdToWkSlice_ : allocCoords.coreIdToWkSlice_) {
          PieceInfo new_piece;
          for (auto kv2 : kv.second) {
            if (relayout_tensor_dims.count(kv2.first) == 0) continue;
            // Fill dimToSize_
            std::string dimName = dldsc.primaryDimToString.at(kv2.first);
            DT_CHECK(wkslices_in_dim.at(kv2.first).size() >= 1);
            new_piece.dimToSize_[dimName] =
                dldsc.N_.paramNameToVal(dimName) /
                wkslices_in_dim.at(kv2.first).size();

            // Piece coordinates are element offsets, not shard ordinals.
            new_piece.dimToStartCordinate[dimName] =
                kv2.second * new_piece.dimToSize_[dimName];

            // Fill validGap_
            new_piece.validGap_[dimName] = {
                std::make_pair(new_piece.dimToSize_[dimName],
                               0)};  // Need to change if gap != 0
          }
          // Fill PlacementInfo
          new_piece.placement[SenComponents::LX];
          new_piece.placement.at(SenComponents::LX).setType(SenComponents::LX);
          new_piece.placement.at(SenComponents::LX).insertMemId(kv.first);
          auto& AddrHelper =
              new_piece.placement.at(SenComponents::LX).StartAddr();
          FoldDimProp* addr_fold_prop = new FoldDimProp{1, "time"};
          AddrHelper.buildMapDim(addr_fold_prop, 0);
          int64_t stAddr = -1;
          if (i == 1) {
            // input of relayout
            auto& ldsLxAlloc = lds.memOrg_.at(LX).allocateNode_;
            stAddr =
                ldsLxAlloc->startAddressCoreCorelet_.getData(kv.first, 0, 0);
          } else {
            // output of relayout
            DT_CHECK(coreIdToAllocAddr.count(kv.first));
            stAddr = coreIdToAllocAddr.at(kv.first);
          }
          new_piece.placement.at(SenComponents::LX).insertStAddr(stAddr, {0});
          myLds.coreIdTolxStartAddress_[kv.first] = stAddr;

          // Insert the piece into the lds
          new_piece.key_ = "p" + std::to_string(piece_count);
          myLds.pieces_["p" + std::to_string(piece_count)] = new_piece;
          piece_count++;
        }
      }

      // Step 3c-lx: Modify the coreId_ to workslice map and addresses of the
      // lds that required relayout in the orig sdsc to match with post-relayout
      // form
      // update input lds order if direct copy
      if (isDirectOutCopyEligible) {
        auto& outDs = dldsc.labeledDs_.back();
        auto& outAllocNode = outDs.memOrg_.at(SenComponents::LX).allocateNode_;
        pre_relayout_allocNode->layoutDimOrder_ = outAllocNode->layoutDimOrder_;

        // The output allocation is already expressed in the destination form,
        // so keep its destination coordinate folds.  The explicit custom map
        // is redundant with the consumer SDSC, though, and prevents DDC from
        // propagating a corelet-split dimension.  Let DDC use the SDSC's map.
        outAllocNode->allocateCoordinates_.coreIdToWkSlice_.clear();

        // The relaid input and the direct output now name the same physical
        // destination form.  Reuse the already-constructed destination folds
        // instead of asking DDC to infer an equivalent factorization.
        allocCoords = outAllocNode->allocateCoordinates_;

        // STCDP has already implemented the direct SHUFFLE by moving the input
        // into the destination allocation.  The remaining compute SDSC reads
        // and writes that same destination form, so lower it to the supported
        // identity op instead of asking DDC for a nonexistent SHUFFLE DDL.
        dldsc.computeOp_.front().opFuncName = OpFuncs::IDENTITY;
        directOutCopyWasReplaced = true;
      }
      if (!isDirectOutCopyEligible) {
        if (hasSameWorkSlicePartition(allocCoords.coreIdToWkSlice_,
                                      sdsc->coreIdToWkSlice_)) {
          // A pure ownership remap keeps the same logical work-slice
          // partition on different physical cores.  Its folds remain valid;
          // only the producer core-to-slice association is stale.
          allocCoords.coreIdToWkSlice_.clear();
        } else {
          // A true repartition (for example 4x8 -> 8x4) changes the logical
          // slice geometry.  Drop the producer coordinate description so DDC
          // rebuilds it from the consumer schedule.  Physical placement is
          // retained below through the rewritten start addresses.
          allocCoords.clear();
        }
      }
      auto& stAddr = pre_relayout_allocNode->startAddressCoreCorelet_;
      for (int coreid = 0; coreid < stAddr.getFoldSpaceSize().at(0); coreid++) {
        if (sdsc->coreIdToWkSlice_.count(coreid) == 0) {
          stAddr.insertData(-1, coreid, 0, 0);
        } else {
          for (auto piece : outLds.pieces_) {
            if (coreid ==
                piece.second.placement.at(SenComponents::LX).getMemId().at(0)) {
              auto newStAddr = piece.second.placement.at(SenComponents::LX)
                                   .getStartAddr({0})
                                   .at(0);  // TODO: check if same piece can
                                            // exist in multiple cores
              stAddr.insertData(newStAddr, coreid, 0, 0);
            }
          }
        }
      }
    } else {
#ifdef DEBUG_RELAYOUT
      std::cout << "Lx space not found, inserting stcdpHBM" << std::endl;
#endif
      // Step 3-hbm: Lx space is not available for the post-relayout form, so
      // relayout will be achieved through stcdpHBM. Create a new dldsc for
      // the relayout node
      DesignSpaceConfig new_dldsc;
      relayout_sdsc->dscs_.push_back(new_dldsc);

      auto& relayout_dldsc = relayout_sdsc->dscs_.at(0);
      relayout_dldsc.name_ = sdsc->name_ + "-HBMRelayout";
      relayout_dldsc.target_ = dldsc.target_;

      const auto stickSizes = dldsc.getCumulativeStickSizes(lds.dsType_);
      for (const auto& [dim, size] : pre_relayout_allocNode->getPageSize()) {
        auto& dsDim = relayout_dldsc.N_.primaryDimToValHandler_st(dim);
        DT_CHECK(std::fmod(dsDim, size) == 0);
        dsDim /= size;
        if (stickSizes.count(dim)) {
          dsDim = std::ceil(dsDim / stickSizes.at(dim)) * stickSizes.at(dim);
        }
      }

      relayout_dldsc.dataStageParam_[0].ss_ = relayout_dldsc.N_;  // single core
      relayout_dldsc.dataStageParam_[0].ss_.name_ = "core";
      relayout_dldsc.dataStageParam_[0].el_ =
          relayout_dldsc.dataStageParam_[0].ss_;
      relayout_sdsc->numWkSlicesPerDim_ = sdsc->numWkSlicesPerDim_;

      // Step 3a-hbm: Fill compute op properties
      ComputeOpInfo relayout_op;
      relayout_op.opFuncName = OpFuncs::IDENTITY;
      relayout_op.exUnit = SenComponents::PE;
      relayout_op.attributes_ = dldsc.computeOp_.at(0).attributes_;
      relayout_dldsc.computeOp_.push_back(relayout_op);

      // Step 3b-hbm: Create new input and output lds for the relayout node,
      // and fill their properties
      relayout_dldsc.labeledDs_.resize(2);
      auto& inpLds = relayout_dldsc.labeledDs_.at(0);
      auto& outLds = relayout_dldsc.labeledDs_.at(1);
      for (int i = 0; i < 2; i++) {
        auto& myLds = (i == 0) ? outLds : inpLds;
        myLds.dsName_ =
            lds.dsName_ + "-HBMRelayout-" + ((i == 0) ? "out" : "in");
        myLds.ldsIdx_ = (i == 0) ? 1 : 0;
        myLds.dsType_ = lds.dsType_;
        myLds.segment_ = lds.segment_;
        myLds.isStatic_ = lds.isStatic_;
        myLds.scale_ = lds.scale_;
        myLds.density_ = lds.density_;
        myLds.wordLength = lds.wordLength;
        myLds.dataFormat_ = lds.dataFormat_;
        LabeledDsInfo* ldsptr = &myLds;
        if (i == 1) {
          myLds.lxSize_ = lds.lxSize_;
          myLds.lxBufferSize_ = lds.lxBufferSize_;
          myLds.lxStartAddress_ = lds.lxStartAddress_;
          relayout_dldsc.computeOp_.at(0).inputLabeledDs.push_back(ldsptr);
        } else {
          myLds.hbmSize_ = lds.lxSize_;
          relayout_dldsc.computeOp_.at(0).outputLabeledDs.push_back(ldsptr);
        }
      }

      // Step 3c-hbm: fill coreIDsUsed for the datadsc
      std::set<int> cores_seen;
      /*for (const auto& mapkv : sdsc->coreIdToWkSlice_) {
        if (cores_seen.count(mapkv.first) == 0) {
          dldsc.coreIdsUsed_.push_back(mapkv.first);
          relayout_sdsc->coreIdToWkSlice_[mapkv.first] = mapkv.second;
          cores_seen.insert(mapkv.first);
        }
      }*/
      for (const auto& mapkv : allocCoords.coreIdToWkSlice_) {
        if (cores_seen.count(mapkv.first) == 0) {
          // dldsc.coreIdsUsed_.push_back(mapkv.first);
          relayout_sdsc->coreIdToWkSlice_[mapkv.first] = mapkv.second;
          cores_seen.insert(mapkv.first);
        }
      }
      relayout_sdsc->numCoresUsed_ = cores_seen.size();
      relayout_dldsc.numCoresUsed_ = cores_seen.size();
      relayout_dldsc.numCoreletsUsed_ = 1;
      DscScheduleStep schedule(-1, 0, 0, 0);
      for (auto coreid : cores_seen) {
        relayout_dldsc.coreIdsUsed_.push_back(coreid);
        relayout_sdsc->coreIdToDscSchedule[coreid].push_back(schedule);
        relayout_sdsc->coreIdToDsc_[coreid] = &relayout_dldsc;
      }

      // Step 3d-hbm: Create new allocNodes for the input and output
      for (auto& kv : lds.memOrg_) {
        auto allocNode = kv.second.allocateNode_;
        if (allocNode == nullptr) continue;
        if (kv.first != SenComponents::LX) continue;

        // input of relayout
        dsc2::AllocateNode* inp_allocNode = new dsc2::AllocateNode(*allocNode);
        relayout_dldsc.scheduleTree_.getHeadMutable()->addChildNode(
            inp_allocNode);
        inpLds.memOrg_[SenComponents::LX] = lds.memOrg_.at(SenComponents::LX);
        inpLds.memOrg_.at(SenComponents::LX).allocateNode_ = inp_allocNode;

        // output of relayout
        dsc2::AllocateNode* out_allocNode = new dsc2::AllocateNode(*allocNode);
        out_allocNode->name_ = out_allocNode->name_ + "_HBMRelayout-out";
        out_allocNode->component_ = SenComponents::HBM;
        /*out_allocNode->allocateCoordinates_.coreIdToWkSlice_ =
            sdsc->coreIdToWkSlice_;*/
        out_allocNode->allocateCoordinates_.coreIdToWkSlice_.clear();

        // find space in HBM for the output of relayout
        auto hbm_stAddr = hbmTrack.checkAndAddDs(outLds.dsName_, "dynamic",
                                                 out_form_size, {ps, ps + 1});
        DT_CHECK_MSG(hbm_stAddr >= 0, "Must find space in HBM\n");

        // find stride for each dim for computing hbm start address in each
        // core
        std::map<PrimaryDimTypes, int> strides;
        auto ref_dsinfo = dldsc.primaryDsInfo_.at(lds.dsType_);
        for (int iter = 0; iter < allocNode->layoutDimOrder_.size(); iter++) {
          auto curr_dim = allocNode->layoutDimOrder_.at(iter);
          strides[curr_dim] = outLds.wordLength;
          std::map<PrimaryDimTypes, int> stick_dim_sizes;
          for (auto iter2 = 0; iter2 < ref_dsinfo.stickDimOrder_.size();
               iter2++) {
            strides[curr_dim] *= ref_dsinfo.stickSize_.at(iter2);
            stick_dim_sizes[ref_dsinfo.stickDimOrder_.at(iter2)] =
                ref_dsinfo.stickSize_.at(iter2);
          }
          for (int iter2 = 0; iter2 < iter; iter2++) {
            auto dim2 = allocNode->layoutDimOrder_.at(iter2);
            strides[curr_dim] *= N_.primaryDimToVal_st(dim2);
            if (std::find(ref_dsinfo.stickDimOrder_.begin(),
                          ref_dsinfo.stickDimOrder_.end(),
                          dim2) != ref_dsinfo.stickDimOrder_.end()) {
              strides[curr_dim] /= stick_dim_sizes.at(dim2);
            }
          }
        }

        // update start addresses for output of relayout
        auto& stAddr = out_allocNode->startAddressCoreCorelet_;
        for (int coreid = 0; coreid < stAddr.getFoldSpaceSize().at(0);
             coreid++) {
          if (sdsc->coreIdToWkSlice_.count(coreid) == 0) {
            stAddr.insertData(-1, coreid, 0, 0);
          } else {
            auto offset_addr = hbm_stAddr;
            for (auto kv : sdsc->coreIdToWkSlice_.at(coreid)) {
              if (sdsc->numWkSlicesPerDim_.at(kv.first) == 1) continue;
              if (strides.count(kv.first) == 0) continue;
              offset_addr += (kv.second *
                              (N_.primaryDimToVal_st(kv.first) /
                               sdsc->numWkSlicesPerDim_.at(kv.first)) *
                              strides.at(kv.first));
            }
            stAddr.insertData(offset_addr, coreid, 0, 0);
          }
        }
        out_allocNode->ldsIdx_ = outLds.ldsIdx_;
        inp_allocNode->ldsIdx_ = inpLds.ldsIdx_;
        relayout_dldsc.scheduleTree_.getHeadMutable()->addChildNode(
            out_allocNode);
        outLds.memOrg_[SenComponents::HBM] = lds.memOrg_.at(SenComponents::LX);
        outLds.memOrg_.at(SenComponents::HBM).allocateNode_ = out_allocNode;
        outLds.memOrg_[SenComponents::LX] = lds.memOrg_.at(SenComponents::LX);
        outLds.memOrg_.at(SenComponents::LX).allocateNode_ = nullptr;
        relayout_dldsc.primaryDsInfo_[outLds.dsType_] =
            dldsc.primaryDsInfo_.at(outLds.dsType_);

        // Step 3e-hbm: Modify the lds that required relayout in the orig sdsc
        // to match with post-relayout form
        allocCoords.coreIdToWkSlice_.clear();
        pre_relayout_allocNode->startAddressCoreCorelet_ =
            out_allocNode->startAddressCoreCorelet_;
        pre_relayout_allocNode->name_ = out_allocNode->name_;
        pre_relayout_allocNode->component_ = SenComponents::HBM;
        lds.memOrg_[SenComponents::HBM] = outLds.memOrg_.at(SenComponents::HBM);
        lds.memOrg_.at(SenComponents::HBM).allocateNode_ =
            pre_relayout_allocNode;
        lds.memOrg_.at(SenComponents::LX).allocateNode_ = nullptr;
      }
    }
    relayout_sdscs.push_back(relayout_sdsc);
#ifdef DEBUG_RELAYOUT
    std::string filename = "relayout_debug_" + sdsc->name_ + "_input";
    filename = filename + std::to_string(iter);
    filename = filename + ".json";
    relayout_sdsc->exportJson(filename);
#endif
  }

  // Diagnostic isolation for a direct-copy SHUFFLE. STCDP already writes the
  // destination allocation, so the old SHUFFLE/IDENTITY program is logically
  // redundant. Replace only the explicitly named SDSC with the same minimal
  // NOP form used by ProgramCorrection. This lets us distinguish transport
  // corruption from an unsafe in-place identity program without changing the
  // rest of the bundle.
  const char* directCopyNopTarget = std::getenv("DXP_RELAYOUT_DIRECT_COPY_NOP");
  if (directOutCopyWasReplaced && directCopyNopTarget != nullptr &&
      sdsc->name_ == std::string(directCopyNopTarget)) {
    std::cout << "Replacing redundant post-relayout SDSC " << sdsc->name_
              << " with NOP" << std::endl;
    sdsc->dscs_.clear();
    sdsc->coreIdToDsc_.clear();
    sdsc->coreIdToDscSchedule.clear();
    sdsc->coreIdToWkSlice_.clear();
    sdsc->numWkSlicesPerDim_.clear();
    sdsc->opFuncsUsed_.clear();
    sdsc->numCoresUsed_ = 0;
    sdsc->dataOpdscs_.clear();
    sdsc->dataOpdscs_.emplace_back().opName = OpFuncs::Nop;
  }
#ifdef DEBUG_RELAYOUT
  std::string filename = "origsdsc_debug_" + sdsc->name_ + ".json";
  sdsc->exportJson(filename);
#endif
}
