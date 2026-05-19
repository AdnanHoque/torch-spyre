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

// Probe-only Deeprt export harness for standalone data-op SDSCs.
//
// This is intentionally not part of production Torch-Spyre lowering. It injects
// one SuperDsc into a one-node DscSenGraph, then calls Deeprt's vertical
// scheduler/codegen/export path. It is useful for data-op restickify experiments
// because raw dxp_standalone rejects SuperDsc objects with datadscs_.

#include <deeprt/deeprt.h>
#include <dsc/dscsengraph.h>
#include <dsc/superdsc.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

// The installed Deeptools SDK exposes DeepRt::memTrackers but does not install
// sharedtools/mem_track_bundle.h. The full class lives in libsharedtools; this
// narrow declaration is enough for this probe to call the initializer before
// the Deeprt scheduling pipeline touches the trackers.
class MemTrackBundle {
 public:
  void initializeMemoryTrackers(const DesignSpaceConfigGlobal& dscGlobal);
};

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <sdsc.json> <out-dir> <backend:senulator|sentient|senpcfg>\n";
    return 2;
  }

  const std::string sdsc_path = argv[1];
  const std::string out_dir = argv[2];
  const std::string backend_name = argv[3];

  SenTargets backend = SenTargets::SENULATOR;
  if (backend_name == "sentient") {
    backend = SenTargets::SENTIENT;
  } else if (backend_name == "senpcfg") {
    backend = SenTargets::SENPCFG;
  } else if (backend_name != "senulator") {
    std::cerr << "unknown backend: " << backend_name << "\n";
    return 2;
  }

  std::filesystem::create_directories(out_dir);

  auto sdsc = std::make_shared<SuperDsc>();
  sdsc->importJson(sdsc_path);
  sdsc->target_ = backend;

  DesignSpaceConfigGlobal dsc_global(backend, true, 32);
  dsc_global.dtVersion = 2;
  dsc_global.doTraining = false;
  dsc_global.numDevices = 1;
  dsc_global.sysDef.numCoreletsPerCore = 2;
  dsc_global.dataDebugMode = true;
  dsc_global.parallelThreads = 1;

  DeepRt deep_rt(dsc_global);
  deep_rt.exportDsc = true;
  deep_rt.exportPcfg = true;
  deep_rt.pruneDataDsc = false;
  deep_rt.noCodeGen = false;
  deep_rt.outputDir = out_dir;
  deep_rt.verbose = 1;
  deep_rt.verticalStackVerbosity = 1;
  deep_rt.memTrackers->initializeMemoryTrackers(dsc_global);

  auto* graph = new sengraph::DscSenGraph(1);
  auto* node = graph->insertNode(sdsc->name_, "SenPreparedOp");
  graph->finalize();
  deep_rt.dSenGraph = graph;
  deep_rt.dsgNodeToSdsc_[node] = sdsc;
  deep_rt.dsgNodeFold0ToAllFoldedNodeExphase_[node].emplace_back(node, 0);
  deep_rt.dsgnode_to_senprog_map_.try_emplace(node);
  deep_rt.dsgnode_to_smc_map_.try_emplace(node);
  deep_rt.dsgnode_to_systemc_map_.try_emplace(node);
  deep_rt.be_usage_.dtversion = 2;
  deep_rt.be_usage_.addInfo(sdsc.get(), DeepRt::CodeGenTools::DCC);

  deep_rt.staticDsg_ = new sengraph::DscSenGraph(1);
  deep_rt.staticDsg_->finalize();
  deep_rt.precompDsg_ = new sengraph::DscSenGraph(1);
  deep_rt.precompDsg_->finalize();
  deep_rt.dynamicDsg_ = new sengraph::DscSenGraph(1);
  deep_rt.dynamicDsg_->insertNode(sdsc->name_, "SenPreparedOp");
  deep_rt.dynamicDsg_->finalize();

  std::cout << "input_dataops=" << sdsc->dataOpdscs_.size()
            << " input_dldscs=" << sdsc->dscs_.size()
            << " cores=" << sdsc->numCoresUsed_ << "\n";

  auto timing = deep_rt.runSchedulerCodeGenInitPipelinePerSdsc(node);
  timing.print();
  deep_rt.printAndExport(4);

  sdsc->exportJson(out_dir + "/after_pipeline.json");

  std::cout << "after_progstateinfo=" << sdsc->progstateinfo_.size() << "\n";
  std::cout << "after_spb=" << sdsc->spb_.size() << "\n";
  for (const auto& kv : sdsc->prog_frame_ptr_) {
    std::cout << "prog_frame target=" << static_cast<int>(kv.first)
              << " size=" << kv.second.size_
              << " st_address=" << kv.second.st_address << "\n";
  }

  return 0;
}
