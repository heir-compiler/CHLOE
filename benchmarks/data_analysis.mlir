module {
  func.func @data_analysis(%arg0: memref<512xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+01 : f32
    %0 = affine.for %arg1 = 0 to 512 iter_args(%arg2 = %cst) -> (f32) {
      %1 = affine.load %arg0[%arg1] : memref<512xf32>
      %2 = arith.cmpf olt, %1, %cst_0 : f32
      %3 = scf.if %2 -> (f32) {
        %4 = arith.addf %arg2, %1 : f32
        scf.yield %4 : f32
      } else {
        scf.yield %arg2 : f32
      }
      affine.yield %3 : f32
    }
    return %0 : f32
  }
}