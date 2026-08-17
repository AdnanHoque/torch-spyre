#map_0 = affine_map<(d0)[s0] -> (s0 + 128*d0)>
module {
	func.func @sdsc_bundle(%arg_0_base_addr: !sdscbundle.input_arg<index>, %arg_1_base_addr: !sdscbundle.input_arg<index>, %arg_2_base_addr: !sdscbundle.input_arg<index>, %arg_3_base_addr: !sdscbundle.input_arg<index>, %arg_4_base_addr: !sdscbundle.input_arg<index>, %arg_5_base_addr: !sdscbundle.input_arg<index>, %arg_6_base_addr: !sdscbundle.input_arg<index>) {
		%arg_0 = sdscbundle.input_arg_extract value from %arg_0_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_1 = sdscbundle.input_arg_extract value from %arg_1_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_2 = sdscbundle.input_arg_extract value from %arg_2_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_3 = sdscbundle.input_arg_extract value from %arg_3_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_4 = sdscbundle.input_arg_extract value from %arg_4_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_5 = sdscbundle.input_arg_extract value from %arg_5_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_6 = sdscbundle.input_arg_extract value from %arg_6_base_addr : !sdscbundle.input_arg<index> -> index
		%c0 = arith.constant 0 : index
		%c1 = arith.constant 1 : index
		%loop_bound_0 = arith.constant 2 : index
		sdscbundle.sdsc_execute (%arg_0) {sdsc_filename="sdsc_0.json", "symbol_ids"=[-1]}
		sdscbundle.sdsc_execute (%arg_1) {sdsc_filename="sdsc_1.json", "symbol_ids"=[-2]}
		scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
			%addr_0 = affine.apply #map_0(%i_0)[%arg_2]
			sdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_2.json", "symbol_ids"=[-3]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_3.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_4.json", "symbol_ids"=[]}
			%addr_1 = affine.apply #map_0(%i_0)[%arg_3]
			sdscbundle.sdsc_execute (%addr_1) {sdsc_filename="sdsc_5.json", "symbol_ids"=[-4]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_6.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_7.json", "symbol_ids"=[]}
			%addr_2 = affine.apply #map_0(%i_0)[%arg_4]
			sdscbundle.sdsc_execute (%addr_2) {sdsc_filename="sdsc_8.json", "symbol_ids"=[-5]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_9.json", "symbol_ids"=[]}
			%addr_3 = affine.apply #map_0(%i_0)[%arg_5]
			sdscbundle.sdsc_execute (%addr_3) {sdsc_filename="sdsc_10.json", "symbol_ids"=[-6]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_11.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_12.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_13.json", "symbol_ids"=[]}
		}
		sdscbundle.sdsc_execute (%arg_6) {sdsc_filename="sdsc_14.json", "symbol_ids"=[-7]}
		return
	}
}
