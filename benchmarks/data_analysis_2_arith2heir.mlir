module {
  func.func @data_analysis(%arg0: memref<512xf32>) -> f32 {
    %0 = heir.encode() {message = 1.000000e+00 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %1 = heir.encode() {message = 0.000000e+00 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %2 = heir.materialize(%1) : (!heir.plain) -> f32
    %3 = heir.encode() {message = 2.000000e+01 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plain
    %4 = affine.for %arg1 = 0 to 512 iter_args(%arg2 = %2) -> (f32) {
      %5 = affine.load %arg0[%arg1] : memref<512xf32>
      %6 = heir.materialize(%5) : (f32) -> !heir.lwecipher<f32>
      %7 = heir.compare(%6, %3) {predicate = 4 : i64} : (!heir.lwecipher<f32>, !heir.plain) -> !heir.lwecipher<f32>
      %8 = heir.materialize(%arg2) : (f32) -> !heir.lwecipher<f32>
      %9 = heir.materialize(%5) : (f32) -> !heir.lwecipher<f32>
      %10 = heir.lweadd(%8, %9) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %11 = heir.lwesub(%0, %7) {noise = -1.000000e+00 : f64} : (!heir.plain, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %12 = heir.lwemul(%10, %7) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %13 = heir.materialize(%arg2) : (f32) -> !heir.lwecipher<f32>
      %14 = heir.lwemul(%13, %11) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %15 = heir.lweadd(%12, %14) {noise = -1.000000e+00 : f64} : (!heir.lwecipher<f32>, !heir.lwecipher<f32>) -> !heir.lwecipher<f32>
      %16 = heir.materialize(%15) : (!heir.lwecipher<f32>) -> f32
      affine.yield %16 : f32
    }
    return %4 : f32
  }
}

