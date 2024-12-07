#ifndef HEIR_PASSES_BATCHLOOP_H_
#define HEIR_PASSES_BATCHLOOP_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct BatchLoopPass : public mlir::PassWrapper<BatchLoopPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "loop-batch";
    }
};

#endif // HEIR_PASSES_SPLITLOOP_H_