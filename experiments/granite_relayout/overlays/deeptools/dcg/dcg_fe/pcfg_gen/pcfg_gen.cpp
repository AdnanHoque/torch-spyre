/************************************************************
* IBM Confidential
* (C) Copyright IBM Corp. 2020, 2025
************************************************************/

/*
 * Description:
 *
 */

#include <dcg/dcg_fe/dcg_frontend.h>
#include <dsc/dsc2Pcfg.h>

#include <cstdlib>

void DcgFE::generatePcfgIRForDataOp(SuperDsc& mySDsc, DataOpDsc& myDataOpDsc,
                                    int c /*=0*/) {
  if (verbose_ > 0) std::cout << "Creating PCFG for DataDsc.." << std::endl;

  auto createNoOp = [&]() {
    myDataOpDsc.pcfg_.clear();
    myDataOpDsc.pcfg_.resize(myDataOpDsc.coreIdsUsed_.size());
    for (int i = 0; i < myDataOpDsc.coreIdsUsed_.size(); i++) {
      auto& coreid = myDataOpDsc.coreIdsUsed_.at(i);
      std::pair<SenPcfg, SenComponents> newPairL3LU;
      newPairL3LU.second = SenComponents::L3LU;
      newPairL3LU.first.dtFormat = DataFormats::SEN169_FP16;
      SenPcfgNopNode* newNoNode =
          (SenPcfgNopNode*)newPairL3LU.first.createPcfgNode(
              SenPcfgNode::Type::NOP);
      newNoNode->name =
          "c" + std::to_string(coreid) + "-L3LU-NOP-" + std::to_string(c);
      newPairL3LU.first.srcNode = newNoNode;
      myDataOpDsc.pcfg_.at(i).push_back(newPairL3LU);
    }
  };

  if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx ||
      myDataOpDsc.op->name == OpFuncs::STCDPOpHBM ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNLX ||
      myDataOpDsc.op->name == OpFuncs::ResizeNNHBM) {
    bool skip = false;
    if (myDataOpDsc.op->name == OpFuncs::STCDPOpLx) {
      STCDPOpLx* myOp = (STCDPOpLx*)myDataOpDsc.op;
      if (myOp->dtTable_.empty()) {
        createNoOp();
        skip = true;
      }
    }

    bool forceNoOpt = false;
    if (!skip && myDataOpDsc.op->name == OpFuncs::STCDPOpLx &&
        std::getenv("P06_STCDP_FORCE_NOOPT") != nullptr &&
        !myDataOpDsc.labeledDs_.empty()) {
      const auto& dims = myDataOpDsc.labeledDs_.front().dimToLayoutSize_;
      forceNoOpt = dims.count("x") && dims.count("y") && dims.count("mb") &&
                   dims.count("in") && dims.at("x") == 8 &&
                   dims.at("y") == 512 && dims.at("mb") == 4 &&
                   dims.at("in") == 128;
      if (forceNoOpt) {
        std::cout << "P06_STCDP_FORCE_NOOPT active for " << mySDsc.name_
                  << std::endl;
      }
    }

    if (!skip) createPcfgsSTCDPOp(&myDataOpDsc, forceNoOpt);
  } else if (myDataOpDsc.op->name == OpFuncs::GatherOpHBM) {
    GatherOpHBM* myOp = (GatherOpHBM*)myDataOpDsc.op;

    createPcfgsGatherOpHBM(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::APEOpLX ||
             myDataOpDsc.op->name == OpFuncs::APEOpHBM) {
    createPcfgsAPEOp(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::ReStickifyOpLx ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpHBM) {
    createPcfgsReStickyOp(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::XRFWriteHBM ||
             myDataOpDsc.op->name == OpFuncs::XRFWriteLX) {
    createPcfgsXRFWrite(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx ||
             myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM) {
    createPcfgsReStickyOp(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::Nop) {
    createNoOp();
  } else if (myDataOpDsc.op->name == OpFuncs::StickifyOpHBM) {
    createPcfgsStickifyOp(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::ConstPadOpLX ||
             myDataOpDsc.op->name == OpFuncs::ConstPadOpHBM) {
    createPcfgsConstPadOp(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::ScatterOpHBM) {
    ScatterOpHBM* myOp = (ScatterOpHBM*)myDataOpDsc.op;
    createPcfgsScatterOpHBM(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::ITOF) {
    baseITOFOp* myOp = (baseITOFOp*)myDataOpDsc.op;
    createPcfgsITOFOp(&myDataOpDsc);
  } else if (myDataOpDsc.op->name == OpFuncs::ITOFHBM) {
    baseITOFOp* myOp = (baseITOFOp*)myDataOpDsc.op;
    createPcfgsITOFOp(&myDataOpDsc);
  } else {
    // add new ops here..
    DT_CHECK(0);
  }

  // Set SDSC fold props
  std::deque<const FoldDimProp*> foldProps;
  for (auto& foldProp : mySDsc.sdscFoldProps_) {
    foldProps.push_back(foldProp.get());
  }

  for (auto& corePcfgs : myDataOpDsc.pcfg_) {
    for (auto& [pcfg, _] : corePcfgs) {
      pcfg.sdscFoldProps = foldProps;
    }
  }
}

void DcgFE::generatePcfgIRForDLOp(SuperDsc& mySDsc,
                                  std::vector<SenPcfg>& pcfgL3lu,
                                  std::vector<SenPcfg>& pcfgL3su) {
  // create PCFG for l3-lu, l3-su
  int maxCoreId = 0;
  for (auto& entry : mySDsc.coreIdToDsc_) {
    if (maxCoreId < entry.first) {
      maxCoreId = entry.first;
    }
  }
  pcfgL3lu.resize(maxCoreId + 1);  // core idx..
  pcfgL3su.resize(maxCoreId + 1);
  int maxGrpIDinL3 = 0;
  for (auto& kv : mySDsc.coreIdToDsc_) {
    auto coreId = kv.first;
    DT_CHECK(pcfgL3lu.size() > coreId);
    DT_CHECK(pcfgL3su.size() > coreId);
    const auto dsc = kv.second;
    const bool hasDSC2 = dsc->isDSC2();
    int maxGrpIDLU;
    int maxGrpIDSU;

    if (hasDSC2) {
      if (enableL3DlScheduler_) {
        DscPcfgTranslator::PcfgInfo infoL3lu = {pcfgL3lu.at(coreId), coreId, 0,
                                                SenComponents::L3LU,
                                                SenComponents::L3LU};
        DscPcfgTranslator::PcfgInfo infoL3su = {pcfgL3su.at(coreId), coreId, 0,
                                                SenComponents::L3SU,
                                                SenComponents::L3SU};
        DscPcfgTranslator::transformDscCompToPcfg(mySDsc, *dscGlobal, *dsc,
                                                  infoL3lu);
        DscPcfgTranslator::transformDscCompToPcfg(mySDsc, *dscGlobal, *dsc,
                                                  infoL3su);
        maxGrpIDLU = pcfgL3lu.at(coreId).getMaxGTRGroupId();
        maxGrpIDSU = pcfgL3su.at(coreId).getMaxGTRGroupId();
      } else {
        DT_ERROR("maxGrpIDLU and maxGrpIDSU will be used but not set");
      }
    } else {
      maxGrpIDLU = createPcfgForUnitPerCore(mySDsc, pcfgL3lu.at(coreId),
                                            SenComponents::L3LU, coreId);
      maxGrpIDSU = createPcfgForUnitPerCore(mySDsc, pcfgL3su.at(coreId),
                                            SenComponents::L3SU, coreId);
    }
    if (verbose_ > 0 && debugEn_) {
      std::cout << "--- PCFG: L3LU for Core: " << coreId << " ---" << std::endl;
      pcfgL3lu.at(coreId).exportJson(std::cout);
      std::cout << "--- PCFG: L3SU for Core: " << coreId << " ---" << std::endl;
      pcfgL3su.at(coreId).exportJson(std::cout);
    }

    maxGrpIDinL3 = maxGrpIDinL3 < maxGrpIDLU ? maxGrpIDLU : maxGrpIDinL3;
    maxGrpIDinL3 = maxGrpIDinL3 < maxGrpIDSU ? maxGrpIDSU : maxGrpIDinL3;
  }

  // DT_CHECK(maxGrpIDinL3 <= maxGrpIDL3);
  firstAvailGlobalGrpId_ += maxGrpIDinL3;
  firstAvailGlobalGrpId_ = firstAvailGlobalGrpId_ % sysDef.maxGroupID;
}

DataOpDsc* DcgFE::generatePcfgIRForDataOpInpFetch(
    SuperDsc& mySDscMain, SuperDsc* mySDscPre, int datadscIdx /*= -1*/,
    std::string jsonFileName /*= ""*/) {
  isInpFetchNeigh_ = true;
  // ACT1 create dataDSC and fill ldsInfo

  DataOpDsc* masterDDsc = nullptr;
  if (mySDscPre == nullptr) {
    DT_CHECK(datadscIdx >= 0 && datadscIdx < mySDscMain.dataOpdscs_.size());
    masterDDsc = &mySDscMain.dataOpdscs_.at(0);
    checkDLDSCForInputFetchNeighborOp(
        mySDscMain);  // verification check DL-DSC data..
    createPcfgForInputFetchNeighbor(mySDscMain, mySDscMain.dataOpdscs_.at(0),
                                    datadscIdx, jsonFileName);
  } else {
    masterDDsc = new DataOpDsc;
    checkDLDSCForInputFetchNeighborOp(
        mySDscMain, *mySDscPre);  // verification check DL-DSC data..
    fillDataDSCForInputFetchNeighbor(mySDscMain, *mySDscPre, *masterDDsc);
    createPcfgForInputFetchNeighbor(mySDscMain, *masterDDsc, datadscIdx,
                                    jsonFileName);
  }

  isInpFetchNeigh_ = false;
  return masterDDsc;
}
