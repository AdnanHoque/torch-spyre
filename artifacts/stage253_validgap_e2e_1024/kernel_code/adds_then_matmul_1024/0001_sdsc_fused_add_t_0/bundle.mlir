module {
	func.func @sdsc_bundle() {
		sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0_ReStickifyOpHBM.json"}
		sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1_add.json"}
		sdscbundle.sdsc_execute () {sdsc_filename="sdsc_2_add.json"}
		sdscbundle.sdsc_execute () {sdsc_filename="sdsc_3_ReStickifyOpHBM.json"}
		return
	}
}
