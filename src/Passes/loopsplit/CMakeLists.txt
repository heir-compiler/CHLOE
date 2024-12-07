set(LLVM_TARGET_DEFINITIONS SplitLoop.td)
mlir_tablegen(LowerArithToHEIR.cpp.inc -gen-rewriters)
add_public_tablegen_target(splitloop)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


add_heir_conversion_library(HEIRSplitLoop
        SplitLoop.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HEIRHEIRDialect
        )