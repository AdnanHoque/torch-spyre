#include <dlfcn.h>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>

#include <dsc/superdsc.h>

class Dxp;
class DcgManager;

namespace {

const char* dim_name(PrimaryDimTypes dim) {
  switch (dim) {
    case IN:
      return "IN";
    case OUT:
      return "OUT";
    case IJ:
      return "IJ";
    case MB:
      return "MB";
    case X:
      return "X";
    case Y:
      return "Y";
    case KIJ:
      return "KIJ";
    case I:
      return "I";
    case J:
      return "J";
    case KI:
      return "KI";
    case KJ:
      return "KJ";
    case X1:
      return "X1";
    default:
      return "UNKNOWN";
  }
}

const char* ds_type_name(DsTypes type) {
  switch (type) {
    case INPUT:
      return "INPUT";
    case OUTPUT:
      return "OUTPUT";
    case KERNEL:
      return "KERNEL";
    case KERNEL_IDX:
      return "KERNEL_IDX";
    case INPUT_SCALE:
      return "INPUT_SCALE";
    case KERNEL_SCALE:
      return "KERNEL_SCALE";
    case INTERNAL:
      return "INTERNAL";
    default:
      return "NOT_SET";
  }
}

using CoreletSplitFn = void (*)(Dxp*, SuperDsc*);
using RunDcgFn = void (*)(DcgManager*, SuperDsc&);

void print_sdsc(const char* entrypoint, const SuperDsc& sdsc) {
  static std::mutex output_mutex;
  static std::atomic<unsigned long> dump_index{0};
  std::ostringstream output;
  output << "CORELET_AUDIT entrypoint=" << entrypoint
         << " sdsc=" << sdsc.name_
         << " super_cores=" << sdsc.numCoresUsed_ << " work_slices=";
  for (const auto& [dim, count] : sdsc.numWkSlicesPerDim_) {
    output << dim_name(dim) << ':' << count << ',';
  }
  output << '\n';

  for (const auto& dsc : sdsc.dscs_) {
    output << "CORELET_AUDIT entrypoint=" << entrypoint
           << " dsc=" << dsc.name_ << " cores=" << dsc.numCoresUsed_
           << " corelets=" << dsc.numCoreletsUsed_;
    for (const auto& [stage_id, stage] : dsc.dataStageParam_) {
      output << " stage=" << stage_id << " split=";
      for (const auto& [dim, pieces] : stage.ss_.coreletSplit_) {
        output << dim_name(dim) << ':';
        for (size_t index = 0; index < pieces.size(); ++index) {
          if (index != 0) {
            output << '+';
          }
          output << pieces[index];
        }
        output << ',';
      }
    }
    output << '\n';

    const bool is_matmul =
        dsc.name_.find("bmm") != std::string::npos ||
        dsc.name_.find("Bmm") != std::string::npos ||
        dsc.name_.find("BatchMatMul") != std::string::npos;
    if (!is_matmul) {
      continue;
    }

    output << "LAYOUT_AUDIT entrypoint=" << entrypoint
           << " dsc=" << dsc.name_ << " global_in=" << dsc.N_.in_
           << " global_out=" << dsc.N_.out_
           << " global_mb=" << dsc.N_.mb_
           << " core_in=" << dsc.CoreD_.in_
           << " core_out=" << dsc.CoreD_.out_
           << " core_mb=" << dsc.CoreD_.mb_
           << " corelet_in=" << dsc.CoreletD_.in_
           << " corelet_out=" << dsc.CoreletD_.out_
           << " corelet_mb=" << dsc.CoreletD_.mb_ << '\n';
    for (const auto& compute : dsc.computeOp_) {
      const auto op_name = EnumsConversion::opFuncsToString.find(
          compute.opFuncName);
      output << "LAYOUT_AUDIT compute_op="
             << (op_name == EnumsConversion::opFuncsToString.end()
                     ? "UNKNOWN"
                     : op_name->second)
             << " data_format="
             << static_cast<int>(compute.attributes_.dataFormat_) << '\n';
    }
    for (const auto& lds : dsc.labeledDs_) {
      output << "LAYOUT_AUDIT lds=" << lds.dsName_
             << " type=" << ds_type_name(lds.dsType_)
             << " word_bytes=" << lds.wordLength
             << " data_format=" << static_cast<int>(lds.dataFormat_);
      const auto primary = dsc.primaryDsInfo_.find(lds.dsType_);
      if (primary != dsc.primaryDsInfo_.end()) {
        output << " layout=";
        for (const auto dim : primary->second.layoutDimOrder_) {
          output << dim_name(dim) << ',';
        }
        output << " stick=";
        for (size_t index = 0;
             index < primary->second.stickDimOrder_.size(); ++index) {
          output << dim_name(primary->second.stickDimOrder_[index]) << ':'
                 << primary->second.stickSize_[index] << ',';
        }
      }
      output << '\n';
    }
  }
  const std::string text = output.str();
  std::lock_guard<std::mutex> lock(output_mutex);
  const ssize_t unused = write(STDERR_FILENO, text.data(), text.size());
  (void)unused;

  const char* dump_dir = std::getenv("CORELET_AUDIT_DUMP_DIR");
  if (dump_dir != nullptr && dump_dir[0] != '\0') {
    std::string safe_name = sdsc.name_;
    for (char& value : safe_name) {
      if (!std::isalnum(static_cast<unsigned char>(value)) && value != '-' &&
          value != '_') {
        value = '_';
      }
    }
    const std::string dump_path =
        std::string(dump_dir) + "/" + std::to_string(dump_index++) + "_" +
        entrypoint + "_" + safe_name + ".json";
    sdsc.exportJson(dump_path, 4);
  }
}

}  // namespace

extern "C" void audited_do_corelet_split(Dxp*, SuperDsc*)
    asm("_ZN3Dxp18doCoreletSplitSdscEP8SuperDsc");

extern "C" void audited_do_corelet_split(Dxp* self, SuperDsc* sdsc) {
  static auto real = reinterpret_cast<CoreletSplitFn>(
      dlsym(RTLD_NEXT, "_ZN3Dxp18doCoreletSplitSdscEP8SuperDsc"));
  if (real == nullptr) {
    std::cerr << "CORELET_AUDIT error=dlsym_failed message=" << dlerror()
              << '\n';
    std::abort();
  }

  real(self, sdsc);
  print_sdsc("DxpCoreletSplit", *sdsc);
}

extern "C" void audited_run_dcg(DcgManager*, SuperDsc&)
    asm("_ZN10DcgManager6runDcgER8SuperDsc");

extern "C" void audited_run_dcg(DcgManager* self, SuperDsc& sdsc) {
  static auto real = reinterpret_cast<RunDcgFn>(
      dlsym(RTLD_NEXT, "_ZN10DcgManager6runDcgER8SuperDsc"));
  if (real == nullptr) {
    std::cerr << "CORELET_AUDIT error=dlsym_runDcg_failed message=" << dlerror()
              << '\n';
    std::abort();
  }
  print_sdsc("DcgRun", sdsc);
  real(self, sdsc);
}

extern "C" void audited_run_dcg_for_dl_ops(DcgManager*, SuperDsc&)
    asm("_ZN10DcgManager14runDcgForDlOpsER8SuperDsc");

extern "C" void audited_run_dcg_for_dl_ops(DcgManager* self, SuperDsc& sdsc) {
  static auto real = reinterpret_cast<RunDcgFn>(
      dlsym(RTLD_NEXT, "_ZN10DcgManager14runDcgForDlOpsER8SuperDsc"));
  if (real == nullptr) {
    std::cerr << "CORELET_AUDIT error=dlsym_runDcgForDlOps_failed message="
              << dlerror() << '\n';
    std::abort();
  }
  print_sdsc("DcgRunDlOps", sdsc);
  real(self, sdsc);
}

extern "C" void audited_run_dcg_for_data_dl_ops(DcgManager*, SuperDsc&)
    asm("_ZN10DcgManager21runDcgForDataOpsDlOpsER8SuperDsc");

extern "C" void audited_run_dcg_for_data_dl_ops(DcgManager* self,
                                                  SuperDsc& sdsc) {
  static auto real = reinterpret_cast<RunDcgFn>(
      dlsym(RTLD_NEXT, "_ZN10DcgManager21runDcgForDataOpsDlOpsER8SuperDsc"));
  if (real == nullptr) {
    std::cerr
        << "CORELET_AUDIT error=dlsym_runDcgForDataOpsDlOps_failed message="
        << dlerror() << '\n';
    std::abort();
  }
  print_sdsc("DcgRunDataDlOps", sdsc);
  real(self, sdsc);
}
