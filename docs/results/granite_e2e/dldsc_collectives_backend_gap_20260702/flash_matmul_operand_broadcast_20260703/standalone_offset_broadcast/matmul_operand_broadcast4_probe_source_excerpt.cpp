  myDataOpDsc.labeledDs_.push_back(outDs0);

  myDataOpDsc.op = new STCDPOpLx;
  myDataOpDsc.op->name = OpFuncs::STCDPOpLx;
  STCDPOpLx* myOp = (STCDPOpLx*)myDataOpDsc.op;
}

void populateDataDSCwithMatmulOperandBroadcast4(DataOpDsc& myDataOpDsc) {
  myDataOpDsc.dscName_ = "MatmulOperandBroadcast4";
  myDataOpDsc.opName = OpFuncs::STCDPOpLx;
  myDataOpDsc.coreIdsUsed_ = {0, 1, 2, 3};
  myDataOpDsc.primaryDs_["activation"].dimNames = {"in", "out"};
  myDataOpDsc.primaryDs_["activation"].name_ = "activation";
  for (const auto& dim : myDataOpDsc.primaryDs_.at("activation").dimNames) {
    myDataOpDsc.dimPool_.insert(dim);
  }

  auto initLds = [](LdsInfo& lds, const std::string& name) {
    const std::vector<std::string> dims = {"in", "out"};
    lds.ldsName_ = name;
    lds.wordLength = 2;
    lds.pdsName_ = "activation";
    lds.layoutDimOrder_ = dims;
    lds.stickDimOrder_ = {"out"};
    lds.df = DataFormats::SEN169_FP16;
    lds.dimToLayoutSize_ =
        DCGUtils::createNFillMap(dims, std::vector<double>{128, 512});
    lds.dimToStickSize_ = DCGUtils::createNFillMap(
        lds.stickDimOrder_, std::vector<double>{64});
    lds.validGap_ = DCGUtils::createNFillMap(
        dims, std::vector<std::vector<std::pair<double, double>>>{
                  {{128, 0}}, {{512, 0}}});
    lds.totElements = 128 * 512;
  };

  auto addPiece = [](LdsInfo& lds, const std::string& key,
                     const std::vector<int>& cores, int64_t address,
                     int outStart) {
    const std::vector<std::string> dims = {"in", "out"};
    auto& piece = lds.pieces_[key];
    piece.key_ = key;
    piece.dimToSize_ =
        DCGUtils::createNFillMap(dims, std::vector<double>{128, 128});
    piece.dimToStartCordinate =
        DCGUtils::createNFillMap(dims, std::vector<double>{0, double(outStart)});
    piece.validGap_ = DCGUtils::createNFillMap(
        dims, std::vector<std::vector<std::pair<double, double>>>{
                  {{128, 0}}, {{128, 0}}});
    piece.placement[SenComponents::LX];
    auto senType = SenComponents::LX;
    piece.placement[senType].setType(SenComponents::LX);
    for (const auto core : cores) {
      piece.placement[senType].insertMemId(core);
      piece.placement[senType].insertStAddr(address);
    }
  };

  LdsInfo inpDs;
  initLds(inpDs, "Tensor1-LxInputNeighborFetch-inp");
  for (int chunk = 0; chunk < 4; ++chunk) {
    addPiece(inpDs, "src" + std::to_string(chunk), {chunk}, 0, chunk * 128);
  }
  myDataOpDsc.labeledDs_.push_back(inpDs);

  LdsInfo outDs;
  initLds(outDs, "Tensor1");
  for (int chunk = 0; chunk < 4; ++chunk) {
    addPiece(outDs, "dst_from_src" + std::to_string(chunk), {0, 1, 2, 3},
             1048576 + chunk * 128 * 128 * 2, chunk * 128);
  }
  myDataOpDsc.labeledDs_.push_back(outDs);

  myDataOpDsc.op = new STCDPOpLx;
  myDataOpDsc.op->name = OpFuncs::STCDPOpLx;
}

}  // namespace DCGDataDSCGenerator
