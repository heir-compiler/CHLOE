// RUN: chloe-opt --func2heir --canonicalize
module {
  func.func @data_analysis(%arg0: !heir.lweciphervec<512 x f32>) -> !heir.lwecipher<f32> {
    %0 = heir.encode() {message = 1.000000e+00 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %1 = heir.encode() {message = 0.000000e+00 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %2 = heir.materialize(%1) : (!heir.plain) -> !heir.lwecipher<f32>
    %3 = heir.encode() {message = 2.000000e+01 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %4 = affine.for %arg1 = 0 to 512 iter_args(%arg2 = %2) -> (!heir.lwecipher<f32>) {
      %5 = heir.extract_init(%arg0, %arg1) : (!heir.lweciphervec<512 x f32>, index) -> !heir.lwecipher<f32>
      %6 = heir.compare(%5, %3) {predicate = 4 : i64} : (!heir.lwecipher<f32>, !heir.plain) -> !heir.lwecipher<f32>
      %7 = heir.lweadd(%arg2, %5) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %8 = heir.lwesub(%0, %6) {noise = -1.000000e+00 : f64} : (!heir.plain, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %9 = heir.lwemul(%7, %6) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %10 = heir.lwemul(%arg2, %8) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %11 = heir.lweadd(%9, %10) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      affine.yield %11 : !heir.lwecipher<f32>
    }
    return %4 : !heir.lwecipher<f32>
  }
}

