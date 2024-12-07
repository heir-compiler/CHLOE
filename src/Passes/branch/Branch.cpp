#include <queue>
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APSInt.h"

#include "heir/Passes/branch/Branch.h"

using namespace mlir;
using namespace arith;
using namespace heir;

void BranchPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
  registry.insert<heir::HEIRDialect,
                  mlir::affine::AffineDialect,
                  ArithDialect,
                  func::FuncDialect,
                  mlir::scf::SCFDialect>();
}

LogicalResult unbranchIfOpWithExplicitOps(scf::IfOp ifOp, IRRewriter &rewriter) {
    // Ensure the ifOp has a single return value
    if (ifOp.getResults().size() != 1) {
        return failure();
    }

    OpBuilder builder(ifOp);
    builder.setInsertionPoint(ifOp);

    auto loc = ifOp.getLoc();
    auto &thenBlock = ifOp.getThenRegion().front();
    auto &elseBlock = ifOp.getElseRegion().front();

    // mlir::OpBuilder builder(ifOp);

    // Clone all operations from the then block
    llvm::SmallVector<mlir::Value> thenBlockResults;
    for (auto &op : thenBlock.getOperations()) {
        if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(&op)) {
            for (mlir::Value operand : yieldOp.getOperands()) {
                // Check if the operand is defined outside the ifOp
                if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                    thenBlockResults.push_back(operand);
                    break;
                }
                if (!ifOp->isProperAncestor(operand.getDefiningOp())) {
                    thenBlockResults.push_back(operand);
                }
            }
            break;
        }
        auto clonedOp = builder.clone(op);
        thenBlockResults.push_back(clonedOp->getResult(0));
    }

    // Clone all operations from the else block
    llvm::SmallVector<mlir::Value> elseBlockResults;
    for (auto &op : elseBlock.getOperations()) {
        if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(&op)) {
            for (mlir::Value operand : yieldOp.getOperands()) {
                // Check if the operand is defined outside the ifOp
                if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
                    elseBlockResults.push_back(operand);
                    break;
                }
                if (!ifOp->isProperAncestor(operand.getDefiningOp())) {
                    elseBlockResults.push_back(operand);
                }
            }
            break;
        }
        auto clonedOp = builder.clone(op);
        elseBlockResults.push_back(clonedOp->getResult(0));
    }

    // Combine the then and else results into the new value
    mlir::Value ifBranchValue = thenBlockResults.back();
    mlir::Value elseBranchValue = elseBlockResults.back();
    mlir::Value condition = ifOp.getCondition();

    // Create expressions to replace the ifOp
    auto one = builder.create<mlir::arith::ConstantOp>(loc, builder.getF32FloatAttr(1.0));
    auto condFloat = builder.create<mlir::arith::UIToFPOp>(loc, builder.getF32Type(), condition);
    auto condNeg = builder.create<mlir::arith::SubFOp>(loc, one, condFloat);

    auto thenValueScaled = builder.create<mlir::arith::MulFOp>(loc, ifBranchValue, condFloat);
    auto elseValueScaled = builder.create<mlir::arith::MulFOp>(loc, elseBranchValue, condNeg);
    auto finalValue = builder.create<mlir::arith::AddFOp>(loc, thenValueScaled, elseValueScaled);

    // Replace all uses of the original ifOp
    ifOp.getResult(0).replaceAllUsesWith(finalValue);

    // Erase the original ifOp
    ifOp.erase();

    return success();
}

// Unfold the If-Else Blocks in a MLIR function 
void BranchPass::runOnOperation()
{
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        if (f.walk([&](Operation *op)
                {

        if (scf::IfOp ifOp = llvm::dyn_cast_or_null<scf::IfOp>(op)) {
            if (unbranchIfOpWithExplicitOps(ifOp, rewriter).failed())
            return WalkResult::interrupt();
        }
        return WalkResult(success()); })
                .wasInterrupted())
        signalPassFailure();
        }

}