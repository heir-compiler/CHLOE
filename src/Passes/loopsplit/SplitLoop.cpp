#include <iostream>
#include <memory>
#include <deque>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "heir/Passes/loopsplit/SplitLoop.h"

using namespace mlir;
using namespace arith;
using namespace affine;
using namespace heir;
    
void SplitLoopPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<heir::HEIRDialect>();
}

/// Enum to classify operation types
enum class OpType { Arithmetic, Logic, Other };

std::string enumToString(OpType type) {
  switch (type) {
    case OpType::Arithmetic: return "Arithmetic";
    case OpType::Logic: return "Logic";
    case OpType::Other: return "Other";
  }
  return "Unknown";
}

/// Helper to classify operations into types
// For loadOp, StoreOp and ConstantOp, it depends on the uses of the operation
OpType getOperationType(Operation *op) {
    // if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::ConstantOp>(op)) {
    if (isa<LWEAddOp, LWEMulOp, LWESubOp>(op)) {
        return OpType::Arithmetic;
    // } else if (isa<arith::CmpFOp, arith::SelectOp, scf::IfOp, arith::UIToFPOp>(op)) {
    } else if (isa<FHECmpOp, FHELUTForLTOp, FHELUTForGTOp, FHEFuncCallOp>(op)) {
        return OpType::Logic;
    }
    return OpType::Other;
}

std::optional<int64_t> getConstantValue(mlir::Value value) {
    if (auto constantOp = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return constantOp.value(); 
    }
    return std::nullopt; 
}

/// Represents a block of contiguous operations of the same type
struct Segment {
    OpType type;
    SmallVector<Operation *> ops;

    Segment(OpType t) : type(t) {}
};

// structure for op and its neighbor ops
struct OpNode {
    Operation *op;                       // this op
    llvm::SmallVector<Operation *> adj;  // its adjacency ops
};


llvm::DenseMap<Operation *, llvm::DenseSet<Operation *>> buildAdjacencyList(AffineForOp forOp) {
    llvm::DenseMap<Operation *, llvm::DenseSet<Operation *>> adjList;

    for (Operation &op : forOp.getBody()->getOperations()) {

        // Get its Defining Ops
        for (Value operand : op.getOperands()) {
            if (auto *definingOp = operand.getDefiningOp()) {
                adjList[&op].insert(definingOp);
            }
        }

        // Get its User ops
        for (Value result : op.getResults()) {
            for (Operation *user : result.getUsers()) {
                adjList[&op].insert(user);
            }
        }
    }

    return adjList;
}

bool existOtherSegOp(AffineForOp newForOp, unsigned segID, 
                llvm::DenseMap<Operation*, unsigned> forOpMap)
{
    bool flag = false;
    for (auto &op : *newForOp.getBody()) {
        if (forOpMap[&op] != segID) {
            flag = true;
            break;
        }
    }
    return flag;
}


// llvm::SmallVector<mlir::Operation *> getAdjacentOps(
//     const llvm::SmallVector<OpNode> &adjacencyList, mlir::Operation *op) {
//     for (const OpNode &node : adjacencyList) 
//         if (node.op == op) return node.adj;
//     return {}; 
// }


/// Splits a loop into blocks and then creates segmented loops
void splitLoop(AffineForOp forOp) {
    
    mlir::Block *bodyBlock = forOp.getBody();
    mlir::OpBuilder builder(forOp);
    // Step 1: Allocate OpTye for each operation within the loop
    // Map to store the OpType of each operation
    llvm::DenseMap<mlir::Operation*, OpType> opTypeMap; 

    // Allocate OpType for all operations except for LoadOp/StoreOp/YieldOp
    for (Operation &op : *bodyBlock) {
        OpType opType = getOperationType(&op);
        opTypeMap[&op] = opType;
    }

    // Step 2: Allocate OpType for StoreOp/LoadOp and
    // copy affine load for n times if there is more than one use by different OpTypes
    
    // Allocate OpType for AffineStoreOp and final yieldOp
    for (Operation &op : *bodyBlock) {
        if (auto storeOp = llvm::dyn_cast<FHEInsertOp>(&op)) {
            Value memref = storeOp.getMemref(); 
            if (Operation *defOp = memref.getDefiningOp()) {
                if (opTypeMap.count(defOp)) {
                OpType sourceType = opTypeMap[defOp];
                opTypeMap[&op] = sourceType; // 将 StoreOp 的类型设置为一致
                }
            }
        }
        if (auto storeOp = llvm::dyn_cast<AffineYieldOp>(&op)) {
            Value operand = storeOp.getOperands()[0]; 
            if (Operation *defOp = operand.getDefiningOp()) {
                if (opTypeMap.count(defOp)) {
                OpType sourceType = opTypeMap[defOp];
                opTypeMap[&op] = sourceType; // 将 StoreOp 的类型设置为一致
                }
            }
        }
    }
    // Allocate OpType for AffineLoadOp
    for (Operation &op : *bodyBlock) {
        if (auto loadOp = llvm::dyn_cast<FHEExtractOp>(&op)) {
            Value loadResult = loadOp.getResult();
            unsigned numUses = 0;
            for (auto &use : loadResult.getUses()) {
                (void)use; // To avoid unused variable warnings
                ++numUses;
            }
            // if (numUses == 1 && opTypeMap[loadOp] == OpType::Other) {
            if (numUses == 1 ) {
                for (Operation *user : loadResult.getUsers()) {
                    OpType userType = opTypeMap[user];
                    opTypeMap[loadOp] = userType;
                }
                continue;
            }
            for (Operation *user : loadResult.getUsers()) {
                OpType userType = opTypeMap[user];

                if (userType == OpType::Other) {
                    llvm::outs() << "hello\n";
                    continue;
                } 

                builder.setInsertionPoint(user);
                auto clonedLoadOp = builder.clone(op);
                clonedLoadOp = llvm::dyn_cast<FHEExtractOp>(clonedLoadOp);
                opTypeMap[clonedLoadOp] = userType;

                for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                    if (user->getOperand(i) == loadResult) {
                        user->setOperand(i, clonedLoadOp->getResult(0));
                    }
                }

            }
            if (op.use_empty()) op.erase();
            else if (op.hasOneUse()) {
                for (Operation *user : loadResult.getUsers()) {
                    opTypeMap[&op] = opTypeMap[user];
                }
            }
        }
    }

    // Step 3: Define a vector and add Store/Load Op if the result is used by other segments
    builder.setInsertionPoint(forOp);

    // TODO: Get the memref type to define, shall be replaced in the next version 
    Value inductionVar = forOp.getInductionVar();
    Type memrefType;
    bool setType = false;
    // Traverse the operations in the loop body
    for (Operation &op : *forOp.getBody()) {
        // Check if the induction variable is used as an operand
        for (Value operand : op.getOperands()) {
            if (operand == inductionVar) {
                if (auto loadOp = llvm::dyn_cast<FHEExtractOp>(&op)) {
                    memrefType = loadOp.getVector().getType();
                    setType = true;
                    break;
                }
            }
        }
        if (setType) break;
    }

    for (Operation &op : *bodyBlock) {
        OpType thisType = opTypeMap[&op];
        bool needStore = false;
        // Check if storeOp/LoadOp is needed
        for (Value result : op.getResults()) {
            for (Operation *user : result.getUsers()) {
                OpType userType = opTypeMap[user];
                if (userType != thisType) {
                    needStore = true;
                }
            }
        }
        if (needStore) {
            builder.setInsertionPoint(forOp);
            auto forLoc = forOp.getLoc();
            auto allocOp = builder.create<FHEDefineOp>(forLoc, memrefType);
            Value opResult = op.getResult(0);
            builder.setInsertionPointAfter(&op);
            auto opLoc = op.getLoc();
            auto storeOp = builder.create<FHEInsertOp>(opLoc, opResult, allocOp.getResult(), inductionVar);
            opTypeMap[storeOp] = opTypeMap[&op];
            
            // Bug: loadOp should be defined n times for n users
            auto storeLoc = storeOp.getLoc();
            auto loadOp = builder.create<FHEExtractOp>(storeLoc, opResult.getType(), allocOp.getResult(), inductionVar);
            for (Operation *user : opResult.getUsers()) {
                auto userType = opTypeMap[user];
                if (userType != thisType) {
                    opTypeMap[loadOp] = userType;

                    for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                        if (user->getOperand(i) == opResult) {
                            user->setOperand(i, loadOp.getResult());
                        }
                    }
                }
            }
            // Bug: We have to iterate the user twice to convert all the uses
            for (Operation *user : opResult.getUsers()) {
                auto userType = opTypeMap[user];
                if (userType != thisType) {

                    opTypeMap[loadOp] = userType;

                    for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                        if (user->getOperand(i) == opResult) {
                            user->setOperand(i, loadOp.getResult());
                        }
                    }
                }
            }
        }
    }

    // Step 4: split the loop body into different segments 
    // and create new pure arith/logic loops through BFS iteration
    llvm::DenseMap<Operation *, llvm::DenseSet<Operation *>> adjList = buildAdjacencyList(forOp);
    llvm::DenseMap<Operation *, unsigned> opSegMap;
    llvm::DenseSet<Operation *> visited;
    unsigned segID = 0;

    auto bfs = [&](Operation *startOp) {
        std::deque<Operation *> queue;
        queue.push_back(startOp);
        visited.insert(startOp);
        opSegMap[startOp] = segID;

        while (!queue.empty()) {
            auto nodeOp = queue.front();
            queue.pop_front();

            for (auto neighborOp : adjList[nodeOp]) {
                if (visited.find(neighborOp) == visited.end() 
                && opTypeMap[neighborOp] == opTypeMap[startOp]) {
                    visited.insert(neighborOp);
                    opSegMap[neighborOp] = segID;
                    queue.push_back(neighborOp);
                }
            }
        }
    };
    
    
    for (Operation &op : *bodyBlock) {
        if (!visited.contains(&op)) {
            segID += 1;
            bfs(&op);
        }
    }

    // Step 5: Create new pure loops based on the above segmentation
    // AffineBound lowerBound = forOp.getLowerBound();
    // AffineBound upperBound = forOp.getUpperBound();
    // Value step = forOp.getStep();
    for (unsigned i = 0; i < segID; i++) {
        builder.setInsertionPoint(forOp);
        llvm::DenseMap<Operation*, unsigned> forOpMap;
        auto forOperation = forOp.getOperation();
        auto newForOp= llvm::dyn_cast<AffineForOp>(builder.clone(*forOperation));
        
        auto forOpIt = bodyBlock->begin();
        auto newForOpIt = newForOp.getBody()->begin();

        // Copy Operation segment map
        for (; forOpIt != bodyBlock->end(); ++forOpIt, ++newForOpIt) {
            Operation *oldOp = &*forOpIt;
            Operation *newOp = &*newForOpIt;
            forOpMap[newOp] = opSegMap[oldOp];
        }
        
        bool hasYieldFlag = true;
        while(existOtherSegOp(newForOp, i + 1, forOpMap)) {
            // You cannot erase an Op directly while iterating the block
            llvm::SmallVector<Operation *> toDeleteOps;
            for (auto &op : *newForOp.getBody()) {
                if (op.use_empty() && forOpMap[&op] != i + 1) {
                    
                    // op.print(llvm::outs());
                    // llvm::outs() << "\n";
                    if (op.getUses().empty()) {
                        // llvm::outs() << "No uses\n";
                        toDeleteOps.push_back(&op);
                    }
                }
            }
        
            for (auto &op : toDeleteOps) {
                auto yieldLoc = op->getLoc();
                op->erase();
                // Cut the dependency of YieldOp but create a new essential terminator
                if (isa<AffineYieldOp>(op)) {
                    hasYieldFlag = false;
                    builder.setInsertionPointToEnd(newForOp.getBody());
                    for (auto &op : *newForOp.getBody()) {
                        if (forOpMap[&op] == i + 1 && op.getResult(0).getType() == newForOp.getResultTypes().front()) {
                            auto newYieldOp = builder.create<AffineYieldOp>(yieldLoc, op.getResult(0));
                            forOpMap[newYieldOp] = forOpMap[&op];
                            break;
                        }
                            
                    }
                }

            }
        }

        if (hasYieldFlag) {
            forOp.replaceAllUsesWith(newForOp);
        }
        
    }

    // Finally erase the original loop
    if (forOp.use_empty()) forOp.erase();
    // builder.cloneRegionBefore()
}


// Unroll all the for loops in the input program,
// but we do not actually use this pass because 
// perhaps there exists multiple nested for loops
// for now, we use the built-in "affine-loop-unroll" pass
void SplitLoopPass::runOnOperation()
{
    ConversionTarget target(getContext());
    target.addLegalDialect<affine::AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect, ArithDialect>();
    target.addLegalDialect<heir::HEIRDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<scf::ForOp>();
    // target.addIllegalOp<AffineForOp>();

    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    // TODO: There's likely a much better way to do this that's not this kind of manual walk!
    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<AffineForOp>()))
        {
            // analyzeLoop(op, rewriter);
            splitLoop(op);
        }
    }
    
}

