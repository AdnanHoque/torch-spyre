// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Probe-only wrapper around Deeptools' InputFetchNeighbor path.
//
// dcg_inpfetch_standalone can build the InputFetchNeighbor PCFG for the
// Torch-Spyre restickify fixture, but its -s path rejects a trivial factor-1
// SuperDSC fold attached by the Torch-Spyre bundle. This wrapper leaves the
// JSON import contract intact, clears the no-op fold metadata in memory, and
// then asks the normal DCG backend to print senprog text. For the runtime
// packaging prototype it can also materialize the same program frame binary
// shape that DXP writes into loadprogram_to_device/init.txt.

#include <dcg/dcg_manager/dcg_manager.h>
#include <dsc/superdsc.h>
#include <sharedtools/progtailor.h>

#if __has_include(<dip/dip.h>)
#include <dip/dip.h>
#define TORCH_SPYRE_HAS_DIP_HEADER 1
#else
#define TORCH_SPYRE_HAS_DIP_HEADER 0
#endif

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

// Some installed Deeptools images ship a progtailor header whose declaration
// takes ``const DesignSpaceConfigGlobal*`` while libsharedtools exports the
// older ABI with a mutable pointer. Use the exported ABI directly so the probe
// stays buildable against that image.
void create_program_frame_ptr_abi(
    const SuperDsc& source_sdsc, SuperDsc& dest_sdsc,
    const std::map<SenComponents, Isa>& isa_per_unit,
    std::deque<int64_t> fold_coord, const std::string& tag,
    DesignSpaceConfigGlobal* dsc_global, bool sen_pcfg)
    __asm__(
        "_ZN10ProgTailor28createAndFillProgramFramePtrERK8SuperDscRS0_RKSt3mapI13SenComponents3IsaSt4lessIS5_ESaISt4pairIKS5_S6_EEESt5dequeIlSaIlEERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEP23DesignSpaceConfigGlobalb");

const ProgramFrame& get_program_frame(const SuperDsc& sdsc) {
  auto sentient = sdsc.prog_frame_ptr_.find(SenTargets::SENTIENT);
  if (sentient != sdsc.prog_frame_ptr_.end() && sentient->second.ptr_ &&
      sentient->second.size_ > 0) {
    return sentient->second;
  }
  for (const auto& item : sdsc.prog_frame_ptr_) {
    if (item.second.ptr_ && item.second.size_ > 0) return item.second;
  }
  std::cerr << "no usable program frame generated; available frames="
            << sdsc.prog_frame_ptr_.size() << "\n";
  for (const auto& item : sdsc.prog_frame_ptr_) {
    std::cerr << "  target=" << static_cast<int>(item.first)
              << " size=" << item.second.size_
              << " ptr=" << item.second.ptr_.get() << "\n";
  }
  std::exit(3);
}

void write_init_text(const SuperDsc& sdsc, const std::string& path) {
  const auto& frame = get_program_frame(sdsc);
  std::cerr << "writing init text from frame size=" << frame.size_ << "\n";
  const auto* bytes = reinterpret_cast<const uint8_t*>(frame.ptr_.get());
  const size_t flits = frame.size_ / 128;

  std::ofstream stream(path);
  stream.fill('0');
  for (size_t flit = 0; flit < flits; ++flit) {
    for (int byte = 127; byte >= 0; --byte) {
      const size_t idx = flit * 128 + static_cast<size_t>(byte);
      const uint16_t value = static_cast<uint8_t>(bytes[idx]) & 0xFF;
      stream << std::hex << std::setw(2) << value;
    }
    stream << '\n';
  }
}

void write_init_binary(const SuperDsc& sdsc, const std::string& path) {
  const auto& frame = get_program_frame(sdsc);
  std::cerr << "writing init binary from frame size=" << frame.size_ << "\n";
  const auto* bytes = reinterpret_cast<const char*>(frame.ptr_.get());
  std::ofstream stream(path, std::ios::binary);
  stream.write(bytes, static_cast<std::streamsize>(frame.size_));
}

std::vector<LdsSegment> dxp_lds_segments() {
  return {
      LdsSegment::OUTPUT,   LdsSegment::INPUT, LdsSegment::MODEL,
      LdsSegment::STACK,    LdsSegment::HEAP,  LdsSegment::RESERVE1,
      LdsSegment::RESERVE2, LdsSegment::CONST};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <consumer-main-sdsc.json> <producer-pre-sdsc.json> <out-dir>\n";
    return 2;
  }

  const std::string consumer_path = argv[1];
  const std::string producer_path = argv[2];
  const std::string out_dir = argv[3];
  std::filesystem::create_directories(out_dir);

  SuperDsc consumer_sdsc;
  SuperDsc producer_sdsc;
  consumer_sdsc.importJson(consumer_path);
  producer_sdsc.importJson(producer_path);
  consumer_sdsc.target_ = SenTargets::SENTIENT;
  producer_sdsc.target_ = SenTargets::SENTIENT;

  // The staged Torch-Spyre SDSCs carry a single factor-1 time fold. Importing
  // needs the fold shell, but the senprog path only checks sdscFoldProps_.
  consumer_sdsc.sdscFoldProps_.clear();

  DesignSpaceConfigGlobal dsc_global(SenTargets::SENTIENT);
  dsc_global.setDtVersion(1);
  std::map<SenComponents, Isa> isa_per_unit;
  Isa::generateIsaPerUnit(isa_per_unit, dsc_global.sysDef);

  DcgManager data_op_gen(&dsc_global, &isa_per_unit);
  data_op_gen.verbose = 1;
  data_op_gen.toggleOptLvl = 1;
  data_op_gen.createSenProg = true;
  data_op_gen.debugEn = true;
  data_op_gen.enable_prog_verification_ = true;
  data_op_gen.runDcgForInputFetchNeighbor(consumer_sdsc, &producer_sdsc);

  if (dsc_global.doProgirOpt) {
    ProgTailor pt(isa_per_unit);
    pt.removeRedundantInstrs(&consumer_sdsc, dsc_global.doPatchProg);
  }

  const std::string senprog = out_dir + "/senprog.txt";
  data_op_gen.printSenProgram(&consumer_sdsc, senprog);

#if TORCH_SPYRE_HAS_DIP_HEADER
  consumer_sdsc.prog_frame_ptr_.erase(SenTargets::SENTIENT);
  auto lds_segments = dxp_lds_segments();
  Dip dip(&dsc_global);
  dip.bDumpUnusedCore = false;
  dip.isaPerUnit = &isa_per_unit;
  dip.targetCores = 1;
  dip.patchInit = true;
  dip.isSentinel = true;
  dip.ldsSegs = &lds_segments;
  dip.LSM_pause = false;
  dip.initGeneration = true;
  dip.sdsc = &consumer_sdsc;
  dip.runDip();
#else
  create_program_frame_ptr_abi(consumer_sdsc, consumer_sdsc, isa_per_unit, {0},
                               "", &dsc_global, false);
#endif
  const std::string init_txt = out_dir + "/init.txt";
  const std::string init_bin = out_dir + "/init_binary.bin";
  write_init_text(consumer_sdsc, init_txt);
  write_init_binary(consumer_sdsc, init_bin);

  std::cout << "wrote " << senprog << "\n";
  std::cout << "wrote " << init_txt << "\n";
  std::cout << "wrote " << init_bin << " ("
            << get_program_frame(consumer_sdsc).size_ << " bytes)\n";
  std::cout.flush();
  // Probe-only: some installed Deeptools builds tear down or export the
  // imported InputFetchNeighbor SuperDsc state incorrectly after senprog
  // emission. The artifact we need has already been written, so exit without
  // running destructors.
  std::_Exit(0);
}
