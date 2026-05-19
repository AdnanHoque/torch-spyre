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
// then asks the normal DCG backend to print senprog text.

#include <dcg/dcg_manager/dcg_manager.h>
#include <dsc/superdsc.h>
#include <sharedtools/progtailor.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <string>

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

  // The staged Torch-Spyre SDSCs carry a single factor-1 time fold. Importing
  // needs the fold shell, but the senprog path only checks sdscFoldProps_.
  consumer_sdsc.sdscFoldProps_.clear();

  DesignSpaceConfigGlobal dsc_global;
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
  consumer_sdsc.exportJson(out_dir + "/after_inpfetch.json");

  std::cout << "wrote " << senprog << "\n";
  std::cout << "wrote " << out_dir << "/after_inpfetch.json\n";
  return 0;
}
