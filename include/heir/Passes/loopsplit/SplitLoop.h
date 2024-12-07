#ifndef HEIR_PASSES_SPLITLOOP_H_
#define HEIR_PASSES_SPLITLOOP_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct SplitLoopPass : public mlir::PassWrapper<SplitLoopPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "loop-analyze";
    }
};

#endif // HEIR_PASSES_SPLITLOOP_H_