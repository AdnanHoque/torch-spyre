module {
  func.func @sdsc_bundle() {
    sdscbundle.sdsc_execute () {sdsc_filename = "sdsc_0.json", "symbol_ids" = []}
    return
  }
}
