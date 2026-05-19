module {
%asdin:1 = ddl.dimension{} : index
%asdout:1 = ddl.dimension {}: index

%slice_layout = ddl.layout() {is_order_fixed=false}
%stick_layout_input = ddl.layout(%asdin) {is_order_fixed=false}
%global_layout_input = ddl.layout(%asdin, %asdout) {}
%stick_layout_output = ddl.layout(%asdout) {is_order_fixed=false}
%global_layout_output = ddl.layout(%asdin, %asdout) {}

%type_fp16 = ddl.type {data_type="SEN169_FP16"}
%inptensor_fp16 = ddl.tensor(%slice_layout, %stick_layout_input, %global_layout_input, [%type_fp16]) : index
%outtensor_fp16 = ddl.tensor(%slice_layout, %stick_layout_output, %global_layout_output, [%type_fp16]) : index
%ist_fp16_op = ddl.operation_bind([%type_fp16], [%inptensor_fp16], [%outtensor_fp16]) {opFuncName="interslicetranspose_fp16", required=false}
ddl.constraint(%ist_fp16_op){min_num_valid = 1, max_num_valid = 1}
ddl.constraint() {min_num_cores = 1}

%zero_const = ddl.operand_constant{name="0.0"}
%one_const = ddl.operand_constant{name="1.0"}
%inptensor = ddl.alias_one_tensor_of(%inptensor_fp16)
%outtensor = ddl.alias_one_tensor_of(%outtensor_fp16)
%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="lxlu_input"}
%outtensor_lx_allocation = ddl.get_external_data_transfer_allocation (%outtensor) {memory="lx", data_connect="lxsu_input"}

ddl.dataflow {
   %d_datastage = ddl.get_external_datastage{property = "core"}
   %b_datastage = ddl.get_external_datastage {property = "chunk"}
   %l0subchunk_datastage = ddl.datastage {strategy="minimize"}
   %bottom_datastage = ddl.datastage{strategy="minimize"}
   ddl.datastage_constraint(%l0subchunk_datastage, %outtensor, %asdout) {values=["1"]}
   ddl.datastage_constraint(%bottom_datastage, %outtensor, %asdout) {values=["0.125"]}
   ddl.loop (%d_datastage, %b_datastage, %asdin, %asdout){label="chunk_loop"} {
      ddl.loop (%b_datastage, %l0subchunk_datastage, %asdin, %asdout){label="subchunk_loop"} {
         %inptensor_l0_allocation = ddl.allocate(%inptensor) {memory="l0", num_buffers=-1:si64}
         ddl.implicit_sync(%inptensor_l0_allocation)
         %src_inp_lxl0 = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="lxlu_input"}
         %dst_inp_lxl0 = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0su", data_connect="l0_input"}
         ddl.data_transfer(%src_inp_lxl0, [%dst_inp_lxl0]) {}
         %outtensor_arf_allocation = ddl.allocate(%outtensor) {memory="ptarf"}
         ddl.loop(%l0subchunk_datastage, %bottom_datastage, %asdout) {label="bottom_loop"} {
            %inptensor_arf_allocation = ddl.allocate(%inptensor) {memory="ptarf"}
            %src_inp_l0arf = ddl.unit(%inptensor, %inptensor_l0_allocation) {unit="l0lu", data_connect="l0_input"}
            %dst_inp_l0arf = ddl.unit(%inptensor, %inptensor_arf_allocation) {unit="pt", data_connect="arf_pt"}
            %src_out_compute = ddl.unit(%outtensor, %outtensor_arf_allocation) {unit="pt", data_connect="compute_out"}
            ddl.data_transfer(%src_inp_l0arf, [%dst_inp_l0arf]) {}
            ddl.compute([%dst_inp_l0arf, %one_const, %zero_const], [%src_out_compute]) {computetype="MACC", unit="pt"}
         }
         %src_out_arflx = ddl.unit(%outtensor, %outtensor_arf_allocation) {unit="pt", data_connect="compute_out"}
         %dst_out_arflx = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", vias=["pe"], data_connect="lxsu_input"}
         ddl.data_transfer(%src_out_arflx, [%dst_out_arflx]) {}
      }
   }
}
ddl.transformations {}
}
