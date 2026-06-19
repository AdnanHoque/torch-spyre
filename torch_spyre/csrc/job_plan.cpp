/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "job_plan.h"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "flex/runtime_stream/runtime_operation.hpp"
#include "spyre_allocator.h"
#include "util/processSpyreCodeArtifacts.h"

namespace spyre {

std::unique_ptr<flex::RuntimeOperation> JobPlanStepH2D::construct(
    LaunchContext&) const {
  auto op = std::make_unique<flex::RuntimeOperationH2D>(host_address_,
                                                        &device_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

void JobPlanStepH2D::write(std::ostream& os) const {
  os << "  H2D (Host-to-Device)\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Device bytes: " << device_address_.total_size() << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepD2H::construct(
    LaunchContext&) const {
  auto op = std::make_unique<flex::RuntimeOperationD2H>(&device_address_,
                                                        host_address_);
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

void JobPlanStepD2H::write(std::ostream& os) const {
  os << "  D2H (Device-to-Host)\n";
  os << "    Device bytes: " << device_address_.total_size() << "\n";
  os << "    Host address: " << host_address_ << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepCompute::construct(
    LaunchContext& ctx) const {
  std::vector<const flex::CompositeAddress*> tensor_allocs;
  if (bind_io_addresses_) {
    for (auto& tensor : ctx.inputs_outputs) {
      flex::CompositeAddress* address =
          &(static_cast<SharedOwnerCtx*>(
                tensor.storage().data_ptr().get_context())
                ->composite_addr);
      tensor_allocs.push_back(address);
    }
  }
  auto op = std::make_unique<flex::RuntimeOperationCompute>(
      &binary_address_, std::move(tensor_allocs));
  op->setPipelineBarrier(pipeline_barrier_);
  return op;
}

void JobPlanStepCompute::write(std::ostream& os) const {
  os << "  Device Compute\n";
  os << "    Binary bytes: " << binary_address_.total_size() << "\n";
  os << "    Bind I/O addresses: " << (bind_io_addresses_ ? "yes" : "no")
     << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

// Installed Flex uses region_id as the PF physical base for a single chunk, so
// the host-correction DMVA is the backing region base plus the chunk offset.
static int64_t composite_address_to_dmva(
    const flex::CompositeAddress& composite_address) {
  size_t num_chunks = composite_address.chunks().size();
  TORCH_CHECK(num_chunks == 1, "Interleaved not supported yet");

  const auto& addr = composite_address.chunks()[0].addr;
  return static_cast<int64_t>(addr.region_id + addr.offset);
}

std::unique_ptr<flex::RuntimeOperation> JobPlanStepHostCompute::construct(
    LaunchContext& ctx) const {
  auto make_host_callback = [this](auto&& callback) {
    return std::make_unique<flex::RuntimeOperationHostCallback>(
        pipeline_barrier_, std::forward<decltype(callback)>(callback),
        nullptr);
  };

  // Case 1: input_buffer_ is provided
  if (input_buffer_ != nullptr) {
    return make_host_callback([this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_,
                                             input_buffer_);
    });
  }

  // Case 2: fake symbols (ishape_ is {0})
  // Further discussion is required on "ishape". For now, it's vector<int64_t>,
  // and it's {0}, it's for fake symbols
  if (ishape_.size() == 1 && ishape_[0] == 0) {
    return make_host_callback([this](void*) {
      deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, nullptr);
    });
  }

  // Case 3: extract addresses from context tensors
  std::vector<int64_t> addresses(ctx.inputs_outputs.size());
  int addr_idx = 0;
  for (auto& tensor : ctx.inputs_outputs) {
    int64_t addr = composite_address_to_dmva(
        (static_cast<SharedOwnerCtx*>(tensor.storage().data_ptr().get_context())
             ->composite_addr));
    addresses[addr_idx++] = addr;
  }

  return make_host_callback([this, addresses](void*) {
    deeptools::processComputeOnHostCommand(*hcm_, output_buffer_, &addresses);
  });
}

void JobPlanStepHostCompute::write(std::ostream& os) const {
  os << "  Host Compute\n";
  os << "    Output buffer: " << output_buffer_ << "\n";
  os << "    HCM metadata: " << (hcm_ ? "present" : "null") << "\n";
  os << "    Pipeline barrier: " << (pipeline_barrier_ ? "enabled" : "disabled")
     << "\n";
}

std::ostream& operator<<(std::ostream& os, const JobPlan& plan) {
  os << "============ JobPlan =============\n";
  os << "Total steps: " << plan.steps.size() << "\n";

  // Job allocation
  size_t addr_idx = 0;
  for (const auto& addr : plan.job_allocation) {
    if (addr_idx == 0) {
      os << "Job allocation bytes: " << addr.total_size() << "\n";
    } else {
      os << "Program " << addr_idx - 1
         << " bytes: " << addr.total_size() << "\n";
    }
    ++addr_idx;
  }

  // Expected input shapes
  if (!plan.expected_input_shapes.empty()) {
    os << "Expected input shapes (" << plan.expected_input_shapes.size()
       << " tensors):\n";
    for (size_t i = 0; i < plan.expected_input_shapes.size(); ++i) {
      os << "  Input " << i << ": [";
      for (size_t j = 0; j < plan.expected_input_shapes[i].size(); ++j) {
        if (j > 0) os << ", ";
        os << plan.expected_input_shapes[i][j];
      }
      os << "]\n";
    }
  }

  // Pinned buffers
  os << "Pinned buffers: " << plan.pinned_buffers.size() << "\n";
  for (size_t i = 0; i < plan.pinned_buffers.size(); ++i) {
    const auto& buf = plan.pinned_buffers[i];
    os << "  Buffer " << i << ": ptr=" << buf.data() << ", size=" << buf.size()
       << " bytes\n";
  }

  // Detailed step information
  os << "\nDetailed Steps:\n";
  for (size_t i = 0; i < plan.steps.size(); ++i) {
    os << "Step " << i << ": ";
    os << *plan.steps[i];
  }

  os << "==================================\n";
  return os;
}

}  // namespace spyre
