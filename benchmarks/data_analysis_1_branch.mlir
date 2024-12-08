// RUN: chloe-opt --branch --canonicalize
module {
  func.func @data_analysis(%arg0: memref<512xf32>) -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+01 : f32
    %0 = affine.for %arg1 = 0 to 512 iter_args(%arg2 = %cst_0) -> (f32) {
      %1 = affine.load %arg0[%arg1] : memref<512xf32>
      %2 = arith.cmpf olt, %1, %cst_1 : f32
      %3 = arith.addf %arg2, %1 : f32
      %4 = arith.uitofp %2 : i1 to f32
      %5 = arith.subf %cst, %4 : f32
      %6 = arith.mulf %3, %4 : f32
      %7 = arith.mulf %arg2, %5 : f32
      %8 = arith.addf %6, %7 : f32
      affine.yield %8 : f32
    }
    return %0 : f32
  }
}

