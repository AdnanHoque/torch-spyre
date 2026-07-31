#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include <dsc/dsc2.h>
#include <dsc/superdsc.h>

namespace {

using DistributeFoldBaseFn = void (*)(
    const dsc2::FoldParamInfoType&, int64_t, int64_t, int64_t, int64_t,
    int64_t, std::vector<dsc2::FoldParamInfoType>&);

constexpr const char* kSymbol =
    "_Z18distributeFoldBaseRKN4dsc217FoldParamInfoTypeElllllRSt6vectorIS0_SaIS0_EE";

using DistributeTemporalFn = void (*)(
    const DesignSpaceConfig*, const PrimaryDimAndKind&,
    const dsc2::ScheduleNode*, int, const PadType&, const PadType&,
    SenComponents, SenComponents, const dsc2::VectorOfLoopAndDim&,
    const std::vector<dsc2::FoldParamInfoType>&,
    dsc2::LoopDistributionParamPerNodeType&,
    std::vector<dsc2::FoldParamInfoType>&, int, int);

constexpr const char* kTemporalSymbol =
    "_ZN4dsc232distributeElemArrToTemporalLoopsEPK17DesignSpaceConfigRK17PrimaryDimAndKindPKNS_12ScheduleNodeEiRK7PadTypeSB_13SenComponentsSC_RKSt6vectorINS_20LoopDistributionInfoESaISE_EERKSD_INS_17FoldParamInfoTypeESaISJ_EERSt3mapIPKNS_8LoopNodeESO_IK15PrimaryDimTypesNS_25LoopDistributionParamTypeESt4lessIST_ESaISt4pairIST_SU_EEESV_ISR_ESaISX_IKSR_S10_EEERSL_ii";

}  // namespace

extern "C" void probed_distribute_fold_base(
    const dsc2::FoldParamInfoType&, int64_t, int64_t, int64_t, int64_t,
    int64_t, std::vector<dsc2::FoldParamInfoType>&) asm(
    "_Z18distributeFoldBaseRKN4dsc217FoldParamInfoTypeElllllRSt6vectorIS0_SaIS0_EE");

extern "C" void probed_distribute_fold_base(
    const dsc2::FoldParamInfoType& orig_fold, int64_t num_segments,
    int64_t segment_size, int64_t segment_stride, int64_t chunk_size,
    int64_t chunk_stride,
    std::vector<dsc2::FoldParamInfoType>& folds_after_distribution) {
  char message[512];
  const int length = std::snprintf(
      message, sizeof(message),
      "FOLD_PROBE alpha=%ld beta=%ld cardinality=%ld label=%s "
      "num_segments=%ld segment_size=%ld segment_stride=%ld "
      "chunk_size=%ld chunk_stride=%ld output_size=%zu\n",
      static_cast<long>(orig_fold.alpha), static_cast<long>(orig_fold.beta),
      static_cast<long>(orig_fold.cardinality), orig_fold.foldDimLabel.c_str(),
      static_cast<long>(num_segments), static_cast<long>(segment_size),
      static_cast<long>(segment_stride), static_cast<long>(chunk_size),
      static_cast<long>(chunk_stride), folds_after_distribution.size());
  if (length > 0) {
    const size_t bytes =
        static_cast<size_t>(length) < sizeof(message) ? length : sizeof(message);
    const ssize_t unused = write(STDERR_FILENO, message, bytes);
    (void)unused;
  }

  static auto real = reinterpret_cast<DistributeFoldBaseFn>(
      dlsym(RTLD_NEXT, kSymbol));
  if (real == nullptr) {
    const char* error = dlerror();
    if (error != nullptr) {
      const ssize_t unused = write(STDERR_FILENO, error, std::strlen(error));
      (void)unused;
    }
    std::abort();
  }
  real(orig_fold, num_segments, segment_size, segment_stride, chunk_size,
       chunk_stride, folds_after_distribution);
}

extern "C" void probed_distribute_temporal(
    const DesignSpaceConfig*, const PrimaryDimAndKind&,
    const dsc2::ScheduleNode*, int, const PadType&, const PadType&,
    SenComponents, SenComponents, const dsc2::VectorOfLoopAndDim&,
    const std::vector<dsc2::FoldParamInfoType>&,
    dsc2::LoopDistributionParamPerNodeType&,
    std::vector<dsc2::FoldParamInfoType>&, int, int) asm(
    "_ZN4dsc232distributeElemArrToTemporalLoopsEPK17DesignSpaceConfigRK17PrimaryDimAndKindPKNS_12ScheduleNodeEiRK7PadTypeSB_13SenComponentsSC_RKSt6vectorINS_20LoopDistributionInfoESaISE_EERKSD_INS_17FoldParamInfoTypeESaISJ_EERSt3mapIPKNS_8LoopNodeESO_IK15PrimaryDimTypesNS_25LoopDistributionParamTypeESt4lessIST_ESaISt4pairIST_SU_EEESV_ISR_ESaISX_IKSR_S10_EEERSL_ii");

extern "C" void probed_distribute_temporal(
    const DesignSpaceConfig* dsc, const PrimaryDimAndKind& dim,
    const dsc2::ScheduleNode* owner, int target_lds_index,
    const PadType& reference_padding, const PadType& target_padding,
    SenComponents size_component, SenComponents propagation_component,
    const dsc2::VectorOfLoopAndDim& loop_chain,
    const std::vector<dsc2::FoldParamInfoType>& element_arrangement,
    dsc2::LoopDistributionParamPerNodeType& loop_parameters,
    std::vector<dsc2::FoldParamInfoType>& distributed_element_arrangement,
    int target_corelet_id, int report_level) {
  char header[1024];
  const char* dsc_name = dsc == nullptr ? "<null>" : dsc->name_.c_str();
  const char* owner_name = owner == nullptr ? "<null>" : owner->name_.c_str();
  const char* lds_name = "<invalid>";
  int lds_type = -1;
  if (dsc != nullptr && target_lds_index >= 0 &&
      static_cast<size_t>(target_lds_index) < dsc->labeledDs_.size()) {
    lds_name = dsc->labeledDs_.at(target_lds_index).dsName_.c_str();
    lds_type = static_cast<int>(dsc->labeledDs_.at(target_lds_index).dsType_);
  }
  const int header_length = std::snprintf(
      header, sizeof(header),
      "TEMPORAL_PROBE begin dsc=%s dim=%d kind=%d owner=%s "
      "target_lds=%d lds_name=%s lds_type=%d size_component=%d "
      "propagation_component=%d corelet=%d report=%d loops=%zu folds=%zu\n",
      dsc_name, static_cast<int>(dim.dim_), static_cast<int>(dim.kind_),
      owner_name, target_lds_index, lds_name, lds_type,
      static_cast<int>(size_component),
      static_cast<int>(propagation_component), target_corelet_id, report_level,
      loop_chain.size(), element_arrangement.size());
  if (header_length > 0) {
    const size_t bytes = static_cast<size_t>(header_length) < sizeof(header)
                             ? header_length
                             : sizeof(header);
    const ssize_t unused = write(STDERR_FILENO, header, bytes);
    (void)unused;
  }
  for (size_t index = 0; index < element_arrangement.size(); ++index) {
    const auto& fold = element_arrangement.at(index);
    char fold_message[512];
    const int fold_length = std::snprintf(
        fold_message, sizeof(fold_message),
        "TEMPORAL_PROBE fold=%zu alpha=%ld beta=%ld cardinality=%ld label=%s\n",
        index, static_cast<long>(fold.alpha), static_cast<long>(fold.beta),
        static_cast<long>(fold.cardinality), fold.foldDimLabel.c_str());
    if (fold_length > 0) {
      const size_t bytes = static_cast<size_t>(fold_length) < sizeof(fold_message)
                               ? fold_length
                               : sizeof(fold_message);
      const ssize_t unused = write(STDERR_FILENO, fold_message, bytes);
      (void)unused;
    }
  }

  static auto real = reinterpret_cast<DistributeTemporalFn>(
      dlsym(RTLD_NEXT, kTemporalSymbol));
  if (real == nullptr) {
    std::abort();
  }
  real(dsc, dim, owner, target_lds_index, reference_padding, target_padding,
       size_component, propagation_component, loop_chain, element_arrangement,
       loop_parameters, distributed_element_arrangement, target_corelet_id,
       report_level);
}
