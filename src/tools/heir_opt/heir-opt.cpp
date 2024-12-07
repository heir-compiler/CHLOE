//===- abc-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "heir/IR/FHE/HEIRDialect.h"
#include "heir/Passes/arith2heir/LowerArithToHEIR.h"
#include "heir/Passes/memref2heir/LowerMemrefToHEIR.h"
#include "heir/Passes/heir2emitc/LowerHEIRToEmitC.h"
#include "heir/Passes/func2heir/FuncToHEIR.h"
#include "heir/Passes/branch/Branch.h"
#include "heir/Passes/unroll/UnrollLoop.h"
#include "heir/Passes/loopsplit/SplitLoop.h"
#include "heir/Passes/loopbatch/BatchLoop.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
using namespace arith;
using namespace affine;
using namespace heir;

void pipelineBuilder(OpPassManager &manager)
{
    // manager.addPass(std::make_unique<LowerArithToHEIRPass>());
    manager.addPass(std::make_unique<BranchPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<LowerArithToHEIRPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<LowerMemrefToHEIRPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<FuncToHEIRPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<SplitLoopPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<BatchLoopPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<LowerHEIRToEmitCPass>());
    manager.addPass(createCanonicalizerPass());


}


int main(int argc, char **argv)
{
    mlir::MLIRContext context;
    context.enableMultithreading();

    mlir::DialectRegistry registry;
    registry.insert<HEIRDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<emitc::EmitCDialect>();
    context.loadDialect<HEIRDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<affine::AffineDialect>();
    context.loadDialect<tensor::TensorDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    context.loadDialect<memref::MemRefDialect>();
    context.loadDialect<emitc::EmitCDialect>();
    // Uncomment the following to include *all* MLIR Core dialects, or selectively
    // include what you need like above. You only need to register dialects that
    // will be *parsed* by the tool, not the one generated
    // registerAllDialects(registry);

    // Uncomment the following to make *all* MLIR core passes available.
    // This is only useful for experimenting with the command line to compose
    // registerAllPasses();

    registerCanonicalizerPass();
    registerAffineLoopUnrollPass();
    registerCSEPass();
    registerAffineLoopInvariantCodeMotionPass();
    PassRegistration<LowerArithToHEIRPass>();
    PassRegistration<LowerMemrefToHEIRPass>();
    PassRegistration<LowerHEIRToEmitCPass>();
    PassRegistration<UnrollLoopPass>();
    PassRegistration<FuncToHEIRPass>();
    PassRegistration<BranchPass>();
    PassRegistration<SplitLoopPass>();
    PassRegistration<BatchLoopPass>();

    PassPipelineRegistration<>("full-pass", "Run all passes", pipelineBuilder);

    return asMainReturnCode(MlirOptMain(argc, argv, "HEIR optimizer driver\n", registry));
}
