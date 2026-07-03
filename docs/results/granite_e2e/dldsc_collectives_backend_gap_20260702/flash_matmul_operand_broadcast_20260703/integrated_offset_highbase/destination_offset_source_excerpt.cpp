                                 ? coreAndSlice.second.at(pDim)
                                 : 0;
        dimToSize[dimName] = dimExtent / split;
        dimToStart[dimName] = sliceIdx * dimToSize.at(dimName);
      }
      const auto srcAddr = lxStartAddressForCore(lds, coreAndSlice.first);
      if (srcAddr < 0) {
        *reason = "missing source LX base for producer core " +
                  std::to_string(coreAndSlice.first);
        return false;
      }
      addLxPiece(groupedInpLds,
                 "src" + std::to_string(coreAndSlice.first), coreAndSlice.first,
                 srcAddr, dimToSize, dimToStart);
      if (useMultiPlacementOutput) {
        sourcePieceShapes.push_back({dimToSize, dimToStart});
      }
    }

    if (useMultiPlacementOutput) {
      for (const auto& dstCoreAndAddress : destinationStartAddressByCore) {
        if (coresSeen.insert(dstCoreAndAddress.first).second) {
          groupedDdsc.coreIdsUsed_.push_back(dstCoreAndAddress.first);
        }
      }
      int dstPieceIdx = 0;
      for (const auto& sourceShape : sourcePieceShapes) {
        const int64_t destinationOffset = byteOffsetWithinPiece(
            groupedOutLds, fullDimToSize, fullDimToStart,
            sourceShape.second);
        std::map<int, int64_t> destinationAddressByCoreWithOffset;
        for (const auto& dstCoreAndAddress : destinationStartAddressByCore) {
          destinationAddressByCoreWithOffset[dstCoreAndAddress.first] =
              dstCoreAndAddress.second + destinationOffset;
        }
        addReplicatedLxPieceWithAddresses(
            groupedOutLds, "dst_from_src" + std::to_string(dstPieceIdx++),
            destinationAddressByCoreWithOffset, sourceShape.first,
            sourceShape.second);
      }
    } else {
      for (const auto& dstCoreAndAddress : destinationStartAddressByCore) {
        if (coresSeen.insert(dstCoreAndAddress.first).second) {
          groupedDdsc.coreIdsUsed_.push_back(dstCoreAndAddress.first);
        }
        addLxPiece(groupedOutLds,
                   "dst" + std::to_string(dstCoreAndAddress.first),
                   dstCoreAndAddress.first, dstCoreAndAddress.second,
                   fullDimToSize, fullDimToStart);
      }
    }
