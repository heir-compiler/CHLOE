// RUN: chloe-opt --loop-analyze --canonicalize
module {
  func.func @data_analysis(%arg0: !heir.lweciphervec<512 x f32>) -> !heir.lwecipher<f32> {
    %0 = heir.encode() {message = 1.000000e+00 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %1 = heir.encode() {message = 0.000000e+00 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %2 = heir.materialize(%1) : (!heir.plain) -> !heir.lwecipher<f32>
    %3 = heir.encode() {message = 2.000000e+01 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %4 = heir.define() : () -> !heir.lweciphervec<512 x f32>
    %5 = affine.for %arg1 = 0 to 512 iter_args(%arg2 = %2) -> (!heir.lwecipher<f32>) {
      %7 = heir.extract_init(%arg0, %arg1) : (!heir.lweciphervec<512 x f32>, index) -> !heir.lwecipher<f32>
      %8 = heir.compare(%7, %3) {predicate = 4 : i64} : (!heir.lwecipher<f32>, !heir.plain) -> !heir.lwecipher<f32>
      heir.insert_init(%8, %4, %arg1) : (!heir.lwecipher<f32>, !heir.lweciphervec<512 x f32>, index) -> 
      affine.yield %7 : !heir.lwecipher<f32>
    }
    %6 = affine.for %arg1 = 0 to 512 iter_args(%arg2 = %2) -> (!heir.lwecipher<f32>) {
      %7 = heir.extract_init(%4, %arg1) : (!heir.lweciphervec<512 x f32>, index) -> !heir.lwecipher<f32>
      %8 = heir.extract_init(%arg0, %arg1) : (!heir.lweciphervec<512 x f32>, index) -> !heir.lwecipher<f32>
      %9 = heir.lweadd(%arg2, %8) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %10 = heir.lwesub(%0, %7) {noise = -1.000000e+00 : f64} : (!heir.plain, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %11 = heir.lwemul(%9, %7) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %12 = heir.lwemul(%arg2, %10) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %13 = heir.lweadd(%11, %12) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      affine.yield %13 : !heir.lwecipher<f32>
    }
    return %6 : !heir.lwecipher<f32>
  }
}

