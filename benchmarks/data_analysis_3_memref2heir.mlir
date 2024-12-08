// RUN: chloe-opt --memref2heir --canonicalize
module {
  func.func @data_analysis(%arg0: memref<512xf32>) -> f32 {
    %0 = heir.encode() {message = 1.000000e+00 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %1 = heir.encode() {message = 0.000000e+00 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %2 = heir.materialize(%1) : (!heir.plain) -> f32
    %3 = heir.encode() {message = 2.000000e+01 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %4 = affine.for %arg1 = 0 to 512 iter_args(%arg2 = %2) -> (f32) {
      %5 = heir.materialize(%arg0) : (memref<512xf32>) -> !heir.lweciphervec<512 x f32>
      %6 = heir.extract_init(%5, %arg1) : (!heir.lweciphervec<512 x f32>, index) -> !heir.lwecipher<f32>
      %7 = heir.compare(%6, %3) {predicate = 4 : i64} : (!heir.lwecipher<f32>, !heir.plain) -> !heir.lwecipher<f32>
      %8 = heir.materialize(%arg2) : (f32) -> !heir.lwecipher<f32>
      %9 = heir.lweadd(%8, %6) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %10 = heir.lwesub(%0, %7) {noise = -1.000000e+00 : f64} : (!heir.plain, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %11 = heir.lwemul(%9, %7) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %12 = heir.materialize(%arg2) : (f32) -> !heir.lwecipher<f32>
      %13 = heir.lwemul(%12, %10) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %14 = heir.lweadd(%11, %13) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %15 = heir.materialize(%14) : (!heir.lwecipher<f32>) -> f32
      affine.yield %15 : f32
    }
    return %4 : f32
  }
}

