get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(  ${dialect_libs} )
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIREmitCDialect
        MLIROptLib
        HEIRHEIRDialect
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        # MLIRMemRef
        MLIRParser
        MLIRPass
        MLIRInferTypeOpInterface
        MLIRControlFlowInterfaces
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms  
        HEIRArithToHEIR
        HEIRUnrollLoop
        HEIRMemrefToHEIR
        HEIRFuncToHEIR
        HEIRHEIRToEmitC
        HEIRBranch
        HEIRSplitLoop
        HEIRBatchLoop
        )
add_llvm_executable(chloe-opt chloe-opt.cpp)

llvm_update_compile_flags(chloe-opt)
target_link_libraries(heir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(heir-opt)
