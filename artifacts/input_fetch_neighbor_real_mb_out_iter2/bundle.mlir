module {
	func.func @sdsc_bundle() {
		sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}
		sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1.json"}
		return
	}
}
