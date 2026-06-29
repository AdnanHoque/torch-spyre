module {
  func.func @buf21_attention_value_relayout_repro() {
    sdscbundle.sdsc_execute () {sdsc_filename="buf21_batchmatmul.json"}
    return
  }
}
