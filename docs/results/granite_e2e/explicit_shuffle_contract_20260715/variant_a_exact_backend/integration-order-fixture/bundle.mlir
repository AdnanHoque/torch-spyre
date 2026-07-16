module {
  func.func @sdsc_bundle() {
    sdscbundle.sdsc_execute () {sdsc_filename = "sdsc_0.json", "symbol_ids" = []}
    sdscbundle.sdsc_execute () {sdsc_filename = "sdsc_1.json", "symbol_ids" = []}
    sdscbundle.sdsc_execute () {sdsc_filename = "sdsc_2.json", "symbol_ids" = []}
    return
  }
}
