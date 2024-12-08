// RUN: emitc-translate --mlir-to-cpp
module {
  func.func @data_analysis(%arg0: !emitc.opaque<"std::vector<LWECipher>">) -> !emitc.opaque<"LWECipher"> {
    %0 = "emitc.constant"() <{value = #emitc.opaque<"PolyPlain(MemoryManager::GetPool())">}> : () -> !emitc.opaque<"PolyPlain">
    %1 = "emitc.constant"() <{value = 2.000000e+01 : f64}> : () -> f64
    emitc.call_opaque "encode_simd"(%1, %0) : (f64, !emitc.opaque<"PolyPlain">) -> ()
    %2 = emitc.call_opaque "batch_comparison"(%arg0, %0) : (!emitc.opaque<"std::vector<LWECipher>">, !emitc.opaque<"PolyPlain">) -> !emitc.opaque<"RLWECipher">
    %3 = emitc.call_opaque "lwes2rlwe"(%arg0) : (!emitc.opaque<"std::vector<LWECipher>">) -> !emitc.opaque<"RLWECipher">
    %4 = emitc.call_opaque "rlwe_multiply"(%2, %3) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %5 = emitc.call_opaque "rlwe_rotate"(%4) {args = [0 : index, 256 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %6 = emitc.call_opaque "rlwe_add"(%4, %5) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %7 = emitc.call_opaque "rlwe_rotate"(%6) {args = [0 : index, 128 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %8 = emitc.call_opaque "rlwe_add"(%6, %7) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %9 = emitc.call_opaque "rlwe_rotate"(%8) {args = [0 : index, 64 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %10 = emitc.call_opaque "rlwe_add"(%8, %9) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %11 = emitc.call_opaque "rlwe_rotate"(%10) {args = [0 : index, 32 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %12 = emitc.call_opaque "rlwe_add"(%10, %11) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %13 = emitc.call_opaque "rlwe_rotate"(%12) {args = [0 : index, 16 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %14 = emitc.call_opaque "rlwe_add"(%12, %13) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %15 = emitc.call_opaque "rlwe_rotate"(%14) {args = [0 : index, 8 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %16 = emitc.call_opaque "rlwe_add"(%14, %15) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %17 = emitc.call_opaque "rlwe_rotate"(%16) {args = [0 : index, 4 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %18 = emitc.call_opaque "rlwe_add"(%16, %17) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %19 = emitc.call_opaque "rlwe_rotate"(%18) {args = [0 : index, 2 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %20 = emitc.call_opaque "rlwe_add"(%18, %19) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %21 = emitc.call_opaque "rlwe_rotate"(%20) {args = [0 : index, 1 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %22 = emitc.call_opaque "rlwe_add"(%20, %21) : (!emitc.opaque<"RLWECipher">, !emitc.opaque<"RLWECipher">) -> !emitc.opaque<"RLWECipher">
    %23 = emitc.call_opaque "load"(%22) {args = [0 : index, 0 : si32]} : (!emitc.opaque<"RLWECipher">) -> !emitc.opaque<"LWECipher">
    return %23 : !emitc.opaque<"LWECipher">
  }
}