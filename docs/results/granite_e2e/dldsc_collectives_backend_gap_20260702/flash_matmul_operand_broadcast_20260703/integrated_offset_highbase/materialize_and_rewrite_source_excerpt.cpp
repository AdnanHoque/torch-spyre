        if (forcedDstBaseEnv != nullptr) {
          stAddr = std::atoll(forcedDstBaseEnv);
        }
        if (stAddr < 0) {
          DT_CHECK_MSG(false,
                       "matmul_operand_broadcast could not allocate " +
                           std::to_string(static_cast<long long>(
                               matmulDstBytes)) +
                           " bytes in LX for consumer core " +
                           std::to_string(coreAndSlice.first));
        }
        destinationStartAddressByCore[coreAndSlice.first] = stAddr;
        if (std::getenv("DEEPTOOLS_MATMUL_OPERAND_BROADCAST_DEBUG") != nullptr) {
          std::cerr << "matmul_operand_broadcast dst core="
                    << coreAndSlice.first << " addr=" << stAddr
                    << " bytes=" << matmulDstBytes << std::endl;
        }
      }

      if (fissionRows > 0) {
        DT_CHECK(fissionRows > 0);
        int relayoutIndex = 0;
        const int sourceCoreCount =
            static_cast<int>(matmulBroadcastHook.plan.sourceCoreCount);
        for (int sourceStart = 0; sourceStart < sourceCoreCount;
             sourceStart += fissionRows) {
          memTrackers->insertPsBefore(ps);
          SuperDsc* relayout_sdsc = makeMatmulOperandBroadcastRelayoutSdsc(
              "-Fission" + std::to_string(relayoutIndex++));
          const int sourceLimit =
              std::min(sourceCoreCount, sourceStart + fissionRows);
          std::string materializationReason;
          if (!attachMatmulOperandBroadcastInputFetch(
                  sdsc, relayout_sdsc, lds, matmulBroadcastHook,
                  &materializationReason, &destinationStartAddressByCore,
                  sourceStart, sourceLimit)) {
            delete relayout_sdsc;
            emitMatmulOperandBroadcastPlanArtifact(sdsc, lds, ps,
                                                   matmulBroadcastHook);
            DT_CHECK_MSG(
                false,
                "matmul_operand_broadcast fission materialization failed: " +
                    materializationReason);
          }
          emitMatmulOperandBroadcastSdscArtifact(
              sdsc, lds, matmulBroadcastHook, relayout_sdsc);
          relayout_sdscs.push_back(relayout_sdsc);
        }
        auto allocNode = lds.memOrg_.count(SenComponents::LX)
                             ? lds.memOrg_.at(SenComponents::LX).allocateNode_
                             : nullptr;
        if (allocNode != nullptr) {
          allocNode->allocateCoordinates_.coreIdToWkSlice_.clear();
          auto& stAddr = allocNode->startAddressCoreCorelet_;
          for (int coreId = 0; coreId < stAddr.getFoldSpaceSize().at(0);
               ++coreId) {
            const auto dstIt = destinationStartAddressByCore.find(coreId);
            const auto newAddr = dstIt == destinationStartAddressByCore.end()
                                     ? -1
                                     : dstIt->second;
            stAddr.insertData(newAddr, coreId, 0, 0);
            if (std::getenv("DEEPTOOLS_MATMUL_OPERAND_BROADCAST_DEBUG") != nullptr) {
              std::cerr << "matmul_operand_broadcast consumer base core="
                        << coreId << " addr=" << newAddr << std::endl;
            }
          }
          if (!lds.coreStateInit_.empty()) {
            for (int coreId = 0; coreId < stAddr.getFoldSpaceSize().at(0);
                 ++coreId) {
              if (static_cast<size_t>(coreId) >= lds.coreStateInit_.size())
                continue;
