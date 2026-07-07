# DXP Full-Granite Activation Replay Summary

## dxp_replay_source_chunk_fix_20260707_192431
returncode: 124
stdout_bytes: 0
stderr_bytes: 0

stderr_tail: empty

## dxp_replay_source_chunk_fix_percore63_total512_20260707_192941
returncode: 134
stdout_bytes: 0
stderr_bytes: 8144869

stderr_tail:
      }
      %2678 = sentient.for %arg2 = %0 iter_args(%arg3 = %245) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %2697 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-4-508", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %2698 = sentient.scalar_add %2697, %18 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %2698 : index
      }
      %2679 = sentient.for %arg2 = %0 iter_args(%arg3 = %244) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %2697 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-5-508", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %2698 = sentient.scalar_add %2697, %18 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %2698 : index
      }
      %2680 = sentient.for %arg2 = %0 iter_args(%arg3 = %243) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %2697 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-6-508", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %2698 = sentient.scalar_add %2697, %18 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %2698 : index
      }
      %2681 = sentient.for %arg2 = %0 iter_args(%arg3 = %242) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %2697 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-7-508", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %2698 = sentient.scalar_add %2697, %18 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %2698 : index
      }
      sentient.for %arg2 = %0 {element_sizes = [-1 : i32], programHeader = [], regIndices = [0 : i32], regLocales = [#sentient<reg_type lccr>]}{
        %2697 = sentient.if eq, %0, %arg2 : index -> (index) {element_sizes = [-1 : i32], regIndices = [1 : i32], regLocales = [#sentient<reg_type ear>]}{
          sentient.yield %1294 : index
        } else{
          sentient.yield %1293 : index
        }
        %2698 = sentient.if sge, %arg2, %0 : index -> (index) {element_sizes = [16 : i32], regIndices = [0 : i32], regLocales = [#sentient<reg_type lar>]}{
          sentient.yield %142 : index
        } else{
          %2700 = sentient.if sge, %arg2, %3 : index -> (index) {element_sizes = [16 : i32], regIndices = [0 : i32], regLocales = [#sentient<reg_type lar>]}{
            sentient.yield %140 : index
          } else{
            %2701 = sentient.if sge, %arg2, %2 : index -> (index) {element_sizes = [16 : i32], regIndices = [0 : i32], regLocales = [#sentient<reg_type lar>]}{
              sentient.yield %139 : index
            } else{
              sentient.yield %138 : index
            }
            sentient.yield %2701 : index
          }
          sentient.yield %2700 : index
        }
        %2699 = sentient.load_and_send mutable_addr(%2698), immutable_addr(%2572), increment(%144), consumer(%2697)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-8-508", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      }
      sentient.sync {dbgName = "SSRF(MSSync #15519, MSSync #15520)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      %2682 = sentient.scalar_copy %141 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index
      %2683 = sentient.load_and_send mutable_addr(%2682), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-0-510", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      %2684 = sentient.scalar_copy %140 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index
      %2685 = sentient.load_and_send mutable_addr(%2684), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-1-510", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      %2686 = sentient.scalar_copy %139 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index
      %2687 = sentient.load_and_send mutable_addr(%2686), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-2-510", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      %2688 = sentient.scalar_copy %138 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index
      %2689 = sentient.load_and_send mutable_addr(%2688), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-3-510", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      %2690 = sentient.scalar_copy %241 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index
      %2691 = sentient.load_and_send mutable_addr(%2690), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-4-510", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      %2692 = sentient.scalar_copy %137 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index
      %2693 = sentient.load_and_send mutable_addr(%2692), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-5-510", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      %2694 = sentient.scalar_copy %240 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index
      %2695 = sentient.load_and_send mutable_addr(%2694), immutable_addr(%2572), increment(%144), consumer(%1294)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-6-510", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
      sentient.sync {dbgName = "SSRF(MSSync #15521, MSSync #15522)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #15523, MSSync #15524)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #15525, MSSync #15526)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #15527, MSSync #15528)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(c31-l3su-sync-recv-lxsu0-lxsu1, c31-l3su-sync-send-lxsu0-lxsu1)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu1>, #sentient<consumer_unit lxsu0>]}
      %2696:2 = sentient.for %arg2 = %58 iter_args(%arg3 = %238, %arg4 = %2624) -> (index, index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, 0 : i32, -1 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type ear>, #sentient<reg_type lar>, #sentient<reg_type ear>]}{
        %src_res, %dst_res = sentient.load_and_store src(%1297), dst(%1296), src_mutable_addr(%arg3), src_immutable_addr(%2571), src_inc(%250), dst_mutable_addr(%arg4), dst_immutable_addr(%2575), dst_inc(%250) {burst_size = 12 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-transfer-lds4-src:lx-dst:hbm", dir = #sentient<direction PseudoRandom>, element_size = 16 : i32, element_sizes = [16 : i32, 16 : i32], regIndices = [0 : i32, 0 : i32], regLocales = [#sentient<reg_type lar>, #sentient<reg_type ear>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
        %2697 = sentient.scalar_add %dst_res, %4 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type ear>} : index, index
        sentient.yield %src_res, %2697 : index, index
      }
      uniform.yield
    }
  } {element_sizes = []}
}
Require larger IBUFF
Max IBUFF(256) Current IBUFF(487) for unit:
%1232 = dataflow.get_unit {core = 0 : i32, name = "l3su", num_folds = 1 : i32, type = "l3su"} : index
error: Unable to lower successfully the module for sdsc: 7_batchmatmul
terminate called after throwing an instance of 'DtException'
  what():  DtException: DCC causes the compilation failure, file /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools/dcc/src/Driver/dcc.cpp line 563
timeout: the monitored command dumped core

## dxp_replay_source_chunk_fix_percore16_total128_20260707_193312
returncode: 134
stdout_bytes: 0
stderr_bytes: 21872079

stderr_tail:
        %1476 = sentient.scalar_sub %152, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1464 = sentient.for %arg2 = %172 iter_args(%arg3 = %176) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-6-1974", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %151, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1465 = sentient.for %arg2 = %172 iter_args(%arg3 = %177) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-7-1974", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %149, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      sentient.sync {dbgName = "SSRF(MSSync #36437, MSSync #36438)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      %1466 = sentient.for %arg2 = %172 iter_args(%arg3 = %92) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-0-1976", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %148, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1467 = sentient.for %arg2 = %172 iter_args(%arg3 = %132) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-1-1976", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %146, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1468 = sentient.for %arg2 = %172 iter_args(%arg3 = %131) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-2-1976", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %145, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1469 = sentient.for %arg2 = %172 iter_args(%arg3 = %130) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-3-1976", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %143, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1470 = sentient.for %arg2 = %172 iter_args(%arg3 = %93) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-4-1976", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %8, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1471 = sentient.for %arg2 = %172 iter_args(%arg3 = %129) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-5-1976", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %7, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1472 = sentient.for %arg2 = %172 iter_args(%arg3 = %128) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-6-1976", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %6, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      %1473 = sentient.for %arg2 = %172 iter_args(%arg3 = %127) -> (index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type lar>]}{
        %1475 = sentient.load_and_send mutable_addr(%arg3), immutable_addr(%1229), increment(%84), consumer(%1144)  {burst_size = 16 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-ringDT-lx-ring-OL-7-1976", element_size = 16 : i32, interleaved_group = 0 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>, shuffle_mode = #sentient<shuffle_mode noshuffle>, total_elements = 64 : i32} : index, index, index, index : index
        %1476 = sentient.scalar_sub %5, %1475 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type lar>} : index, index
        sentient.yield %1476 : index
      }
      sentient.sync {dbgName = "SSRF(MSSync #36439, MSSync #36440)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #36441, MSSync #36442)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #36443, MSSync #36444)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #36445, MSSync #36446)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #36447, MSSync #36448)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #36449, MSSync #36450)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #36451, MSSync #36452)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #36453, MSSync #36454)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(MSSync #36455, MSSync #36456)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu0>, #sentient<consumer_unit lxsu1>, #sentient<consumer_unit l3lu>]}
      sentient.sync {dbgName = "SSRF(c31-l3su-sync-recv-lxsu0-lxsu1, c31-l3su-sync-send-lxsu0-lxsu1)", implicit_sync_memory_boundary = -1 : si32, mode = #sentient<sync_mode sendrecv>, soft = false, units = [#sentient<consumer_unit lxsu1>, #sentient<consumer_unit lxsu0>]}
      %1474:2 = sentient.for %arg2 = %45 iter_args(%arg3 = %141, %arg4 = %1249) -> (index, index) {element_sizes = [-1 : i32, 16 : i32, 16 : i32, 16 : i32, 16 : i32], programHeader = [], regIndices = [0 : i32, 0 : i32, 0 : i32, -1 : i32, -1 : i32], regLocales = [#sentient<reg_type lccr>, #sentient<reg_type lar>, #sentient<reg_type ear>, #sentient<reg_type lar>, #sentient<reg_type ear>]}{
        %src_res, %dst_res = sentient.load_and_store src(%1147), dst(%1146), src_mutable_addr(%arg3), src_immutable_addr(%1228), src_inc(%179), dst_mutable_addr(%arg4), dst_immutable_addr(%1232), dst_inc(%179) {burst_size = 12 : i32, chunk_size = 64 : i32, chunk_stride = 0 : i32, dbgName = "c31-l3su-transfer-lds4-src:lx-dst:hbm", dir = #sentient<direction PseudoRandom>, element_size = 16 : i32, element_sizes = [16 : i32, 16 : i32], regIndices = [0 : i32, 0 : i32], regLocales = [#sentient<reg_type lar>, #sentient<reg_type ear>], shuffle_mode = #sentient<shuffle_mode noshuffle>, stride = 64 : i32, total_elements = 64 : i32} : index, index, index, index, index, index, index, index : index, index
        %1475 = sentient.scalar_add %dst_res, %4 {element_size = 16 : i32, regIndex = 0 : i32, regLocale = #sentient<reg_type ear>} : index, index
        sentient.yield %src_res, %1475 : index, index
      }
      uniform.yield
    }
  } {element_sizes = []}
}
Require larger IBUFF
Max IBUFF(256) Current IBUFF(1110) for unit:
%1082 = dataflow.get_unit {core = 0 : i32, name = "l3su", num_folds = 1 : i32, type = "l3su"} : index
error: Unable to lower successfully the module for sdsc: 7_batchmatmul
terminate called after throwing an instance of 'DtException'
  what():  DtException: DCC causes the compilation failure, file /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools/dcc/src/Driver/dcc.cpp line 563
timeout: the monitored command dumped core

