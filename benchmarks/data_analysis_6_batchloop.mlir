// RUN: chloe--opt --loop-batch --canonicalize
module {
  func.func @data_analysis(%arg0: !heir.lweciphervec<512 x f32>) -> !heir.lwecipher<f32> {
    %0 = heir.encode() {message = 2.000000e+01 : f32, noise = -1.000000e+00 : f64} : () -> !heir.plainvector
    %1 = heir.batchcompare(%arg0, %0) {predicate = 4 : i64} : (!heir.lweciphervec<512 x f32>, !heir.plainvector) -> !heir.rlwecipher<512 x f32>
    %2 = heir.repack(%arg0) {noise = -1.000000e+00 : f64} : (!heir.lweciphervec<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %3 = heir.rlwemul(%1, %2) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %4 = heir.rotate(%3) {i = 256 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %5 = heir.rlweadd(%3, %4) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %6 = heir.rotate(%5) {i = 128 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %7 = heir.rlweadd(%5, %6) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %8 = heir.rotate(%7) {i = 64 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %9 = heir.rlweadd(%7, %8) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %10 = heir.rotate(%9) {i = 32 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %11 = heir.rlweadd(%9, %10) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %12 = heir.rotate(%11) {i = 16 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %13 = heir.rlweadd(%11, %12) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %14 = heir.rotate(%13) {i = 8 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %15 = heir.rlweadd(%13, %14) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %16 = heir.rotate(%15) {i = 4 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %17 = heir.rlweadd(%15, %16) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %18 = heir.rotate(%17) {i = 2 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %19 = heir.rlweadd(%17, %18) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %20 = heir.rotate(%19) {i = 1 : si32, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %21 = heir.rlweadd(%19, %20) {noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>, !heir.rlwecipher<512 x f32>) -> !heir.rlwecipher<512 x f32>
    %22 = heir.extract(%21) {col = 0 : index, noise = -1.000000e+00 : f64} : (!heir.rlwecipher<512 x f32>) -> !heir.lwecipher<f32>
    return %22 : !heir.lwecipher<f32>
  }
}

