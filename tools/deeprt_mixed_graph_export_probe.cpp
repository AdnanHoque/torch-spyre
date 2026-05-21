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

// Probe-only Deeprt export harness for a prepared three-node graph:
//   producer compute SDSC -> restickify data-op SDSC -> consumer compute SDSC.
//
// This is not production lowering. It answers one narrow integration question:
// can Deeprt's graph-level scheduler/codegen/export path accept the LX-restickify
// data-op artifact between two real Torch-Spyre compute SDSCs?

#include <deeprt/deeprt.h>
#include <dsc/dscsengraph.h>
#include <dsc/superdsc.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// The installed Deeptools SDK exposes DeepRt::memTrackers but does not install
// sharedtools/mem_track_bundle.h. The full class lives in libsharedtools; this
// narrow declaration is enough for this probe to call the initializer before
// the Deeprt scheduling pipeline touches the trackers.
class MemTrackBundle {
 public:
  void initializeMemoryTrackers(const DesignSpaceConfigGlobal& dscGlobal);
};

namespace {

SenTargets parse_backend(const std::string& backend_name) {
  if (backend_name == "sentient") {
    return SenTargets::SENTIENT;
  }
  if (backend_name == "senpcfg") {
    return SenTargets::SENPCFG;
  }
  if (backend_name == "senulator") {
    return SenTargets::SENULATOR;
  }
  std::cerr << "unknown backend: " << backend_name << "\n";
  std::exit(2);
}

std::shared_ptr<SuperDsc> load_sdsc(const std::string& path,
                                    SenTargets backend) {
  auto sdsc = std::make_shared<SuperDsc>();
  sdsc->importJson(path);
  sdsc->target_ = backend;
  return sdsc;
}

std::vector<sengraph::Node*> populate_chain(sengraph::DscSenGraph* graph,
                                            const std::string& producer_name,
                                            const std::string& restickify_name,
                                            const std::string& consumer_name,
                                            int producer_output_index,
                                            int restickify_input_index,
                                            int restickify_output_index,
                                            int consumer_input_index) {
  auto* producer = graph->insertNode(producer_name, "SenPreparedOp");
  auto* restickify = graph->insertNode(restickify_name, "SenPreparedOp");
  auto* consumer = graph->insertNode(consumer_name, "SenPreparedOp");

  graph->insertDataEdge(
      producer, producer_output_index, restickify, restickify_input_index);
  graph->insertDataEdge(
      restickify, restickify_output_index, consumer, consumer_input_index);
  graph->insertCtrlEdge(producer, restickify);
  graph->insertCtrlEdge(restickify, consumer);
  graph->insertInput(producer);
  if (consumer_input_index > 0) {
    // The consumer may have another graph/runtime input before the internal
    // restickify edge.  Marking the consumer as an input node keeps Deeprt's
    // graph bookkeeping from treating that lower input slot as absent.
    graph->insertInput(consumer);
  }
  graph->insertOutput(consumer);
  graph->finalize();
  graph->finalizeDscSenGraph(producer);
  return {producer, restickify, consumer};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: " << argv[0]
              << " <producer-sdsc.json> <restickify-dataop-sdsc.json>"
              << " <consumer-sdsc.json> <out-dir>"
              << " <backend:senulator|sentient|senpcfg>"
              << " [producer-output-index restickify-input-index"
              << " restickify-output-index consumer-input-index]\n";
    return 2;
  }

  const std::string producer_path = argv[1];
  const std::string restickify_path = argv[2];
  const std::string consumer_path = argv[3];
  const std::string out_dir = argv[4];
  const std::string backend_name = argv[5];
  const SenTargets backend = parse_backend(backend_name);
  const int producer_output_index = argc > 6 ? std::stoi(argv[6]) : 0;
  const int restickify_input_index = argc > 7 ? std::stoi(argv[7]) : 0;
  const int restickify_output_index = argc > 8 ? std::stoi(argv[8]) : 0;
  const int consumer_input_index = argc > 9 ? std::stoi(argv[9]) : 0;

  std::filesystem::create_directories(out_dir);

  auto producer_sdsc = load_sdsc(producer_path, backend);
  auto restickify_sdsc = load_sdsc(restickify_path, backend);
  auto consumer_sdsc = load_sdsc(consumer_path, backend);

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
  auto nodes = populate_chain(graph, producer_sdsc->name_, restickify_sdsc->name_,
                              consumer_sdsc->name_, producer_output_index,
                              restickify_input_index, restickify_output_index,
                              consumer_input_index);
  deep_rt.dSenGraph = graph;
  deep_rt.dsgNodeToSdsc_[nodes[0]] = producer_sdsc;
  deep_rt.dsgNodeToSdsc_[nodes[1]] = restickify_sdsc;
  deep_rt.dsgNodeToSdsc_[nodes[2]] = consumer_sdsc;
  deep_rt.be_usage_.dtversion = 2;
  deep_rt.be_usage_.addInfo(producer_sdsc.get(), DeepRt::CodeGenTools::DCC);
  deep_rt.be_usage_.addInfo(restickify_sdsc.get(), DeepRt::CodeGenTools::DCC);
  deep_rt.be_usage_.addInfo(consumer_sdsc.get(), DeepRt::CodeGenTools::DCC);

  deep_rt.staticDsg_ = new sengraph::DscSenGraph(1);
  populate_chain(deep_rt.staticDsg_, producer_sdsc->name_ + "_static",
                 restickify_sdsc->name_ + "_static",
                 consumer_sdsc->name_ + "_static", producer_output_index,
                 restickify_input_index, restickify_output_index,
                 consumer_input_index);
  deep_rt.precompDsg_ = new sengraph::DscSenGraph(1);
  populate_chain(deep_rt.precompDsg_, producer_sdsc->name_ + "_precomp",
                 restickify_sdsc->name_ + "_precomp",
                 consumer_sdsc->name_ + "_precomp", producer_output_index,
                 restickify_input_index, restickify_output_index,
                 consumer_input_index);
  deep_rt.dynamicDsg_ = new sengraph::DscSenGraph(1);
  populate_chain(deep_rt.dynamicDsg_, producer_sdsc->name_ + "_dynamic",
                 restickify_sdsc->name_ + "_dynamic",
                 consumer_sdsc->name_ + "_dynamic", producer_output_index,
                 restickify_input_index, restickify_output_index,
                 consumer_input_index);

  std::cout << "producer=" << producer_sdsc->name_
            << " dldscs=" << producer_sdsc->dscs_.size()
            << " dataops=" << producer_sdsc->dataOpdscs_.size() << "\n";
  std::cout << "restickify=" << restickify_sdsc->name_
            << " dldscs=" << restickify_sdsc->dscs_.size()
            << " dataops=" << restickify_sdsc->dataOpdscs_.size() << "\n";
  std::cout << "consumer=" << consumer_sdsc->name_
            << " dldscs=" << consumer_sdsc->dscs_.size()
            << " dataops=" << consumer_sdsc->dataOpdscs_.size() << "\n";
  std::cout << "graph_nodes=" << graph->nodeSeqOfDevice(0).size() << "\n";
  std::cout << "edges=producer:" << producer_output_index
            << "->restickify:" << restickify_input_index
            << ", restickify:" << restickify_output_index
            << "->consumer:" << consumer_input_index << "\n";

  deep_rt.runSchedulerCodeGenInitPipeline();
  deep_rt.printAndExport(4);

  producer_sdsc->exportJson(out_dir + "/after_pipeline_producer.json");
  restickify_sdsc->exportJson(out_dir + "/after_pipeline_restickify.json");
  consumer_sdsc->exportJson(out_dir + "/after_pipeline_consumer.json");

  std::cout << "done\n";
  return 0;
}
