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
#include "heir/Passes/loopbatch/BatchLoop.h"

using namespace mlir;
using namespace arith;
using namespace affine;
using namespace heir;
    
void BatchLoopPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<heir::HEIRDialect>();
}

enum class OpType { Arithmetic, Logic, Other };

OpType OperationType(Operation *op) {
    if (isa<LWEAddOp, LWEMulOp, LWESubOp>(op)) {
        return OpType::Arithmetic;
    } else if (isa<FHECmpOp, FHELUTForLTOp, FHELUTForGTOp, FHEFuncCallOp>(op)) {
        return OpType::Logic;
    }
    return OpType::Other;
}


bool isLogicLoop(AffineForOp forOp) {
    for (auto &op : *forOp.getBody()) {
        if (OperationType(&op) == OpType::Arithmetic) {
            return false;
        }
        else if (OperationType(&op) == OpType::Logic) {
            return true;
        }
        else continue;
    }
    return false;
}

std::optional<Value> getOutsideOperand(FHECmpOp cmpOp, AffineForOp forOp)
{
    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    if (forOp.isDefinedOutsideOfLoop(lhs)) return lhs;
    else if (forOp.isDefinedOutsideOfLoop(rhs)) return rhs;

    return std::nullopt;
}

std::optional<Value> getIterArgFromElseMulOp(LWEMulOp mulOp, AffineForOp forOp)
{
    auto iterArgs = forOp.getRegionIterArgs();
    auto mulOperands = mulOp.getOperands();

    for (Value operand : mulOperands) {
        for (auto iterArg : iterArgs) {
            if (operand == iterArg) return iterArg;
        }
    }
    return std::nullopt;
}

void identifyElseBranch(AffineForOp forOp, LWESubOp &outerSubOp, LWEMulOp &outerMulOp, 
                        LWEAddOp &outerAggOp, Value &outeriterArg, Value &condition)
{
    for (auto &op : *forOp.getBody()) {
        // Check else Branch
        if (LWESubOp subOp = dyn_cast_or_null<LWESubOp>(op)) {
            Value minuend = subOp.getOperand(0);
            Value subtrahend = subOp.getOperand(1);

            auto oneOp = dyn_cast_or_null<FHEEncodeOp>(minuend.getDefiningOp());
            if (!oneOp || oneOp.getMessage().convertToFloat() != 1) continue;
            
            for (auto *user : subOp.getResult().getUsers()) {
                auto elseMulOp = dyn_cast_or_null<LWEMulOp>(user);
                if (!elseMulOp) continue;
                std::optional<Value> iterArgOp = getIterArgFromElseMulOp(elseMulOp, forOp);
                if (!iterArgOp.has_value()) continue;
                auto aggOp = dyn_cast<LWEAddOp>(*elseMulOp.getResult().getUsers().begin());
                if (!aggOp) continue;
                outeriterArg = iterArgOp.value();
                outerMulOp = elseMulOp;
                outerSubOp = subOp;
                condition = subtrahend;
                outerAggOp = aggOp;
            }
        }
    }
}

void identifyIfBranch(AffineForOp forOp, LWEAddOp aggOp, LWEMulOp elseMulOp, Value iterArg,
                        Value condition, LWEMulOp &outerIfMulOp, LWEAddOp &outerIfAggOp)
{
    if (aggOp.getNumOperands() != 2) return;
    LWEAddOp ifAggOp;
    for (auto operand : aggOp.getOperands()) {
        if (operand.getDefiningOp() == elseMulOp) {
            continue;
        }
        if (auto ifMulOp = dyn_cast_or_null<LWEMulOp>(operand.getDefiningOp())) {
            if (ifMulOp.getOperand(0) == condition) 
                ifAggOp = dyn_cast_or_null<LWEAddOp>(ifMulOp.getOperand(1).getDefiningOp());       
            else if (ifMulOp.getOperand(1) == condition) 
                ifAggOp = dyn_cast_or_null<LWEAddOp>(ifMulOp.getOperand(0).getDefiningOp());
            else return;

            for (auto ifAggOperand : ifAggOp.getOperands()) {
                if (ifAggOperand == iterArg) {
                    outerIfMulOp = ifMulOp;
                    outerIfAggOp = ifAggOp;
                    break;
                }
            }
            
        }
        
    }
}

bool isIterationOp(FHEExtractOp op, Value inducVar)
{
    for (auto loadOperand : op.getOperands()) 
        if (loadOperand == inducVar) return true;
    
    return false;
}

void BatchLoop(AffineForOp forOp)
{
    mlir::Block *bodyBlock = forOp.getBody();
    mlir::OpBuilder builder(forOp);

    auto inducVar = forOp.getInductionVar();
    auto iterArgs = forOp.getRegionIterArgs();

    bool isLogic = isLogicLoop(forOp);

    // If this loop is a logic loop
    if (isLogic) {
        // if loop-carried variables are used in logic loop
        // we cannot batch this loop
        for (auto &op : *bodyBlock) {
            for (unsigned i = 0; i < op.getNumOperands(); i++) {
                for (auto &iterArg : iterArgs) {
                    if (op.getOperand(i) == iterArg) return;
                }
            }
        }

        // Pattern Matching
        // FHEExtractOp -> FHELUTOp -> FHEInsertOp
        for (auto &op : *bodyBlock) {
            if (auto extractOp = dyn_cast<FHEExtractOp>(op)) {
                auto extractIndex = extractOp.getI();
                auto extractVector = extractOp.getVector();
                Value extractResult = extractOp.getResult();
                if (extractIndex.front() != inducVar) return;

                for (Operation *user : extractResult.getUsers()) {
                    auto cmpOp = dyn_cast_or_null<FHECmpOp>(user);
                    if (!cmpOp) continue;;
                    std::optional<Value> outsideOperandOp = getOutsideOperand(cmpOp, forOp);
                    Value cmpLhs = cmpOp.getLhs();
                    Value cmpRhs = cmpOp.getRhs();
                    auto cmpPredicate = cmpOp.getPredicate();
                    if (!outsideOperandOp.has_value()) return;
                    Value outsideOperand = outsideOperandOp.value();

                    for (Operation *cmpUser : cmpOp.getResult().getUsers()) {
                        auto insertOp = dyn_cast_or_null<FHEInsertOp>(cmpUser);
                        if (!insertOp) continue;
                        if (insertOp.getIndex().front() != inducVar) return;
                        Value insertVector = insertOp.getMemref();

                        auto encodeOp = dyn_cast<FHEEncodeOp>(outsideOperand.getDefiningOp());
                        if (!encodeOp) return;
                        builder.setInsertionPoint(encodeOp);
                        auto batchEncodeOp = builder.create<FHEEncodeOp>(
                                encodeOp.getLoc(), PlainVectorType::get(builder.getContext()), 
                                encodeOp.getMessageAttr(), encodeOp.getNoiseAttr());
                        Value batchOperand = batchEncodeOp.getResult();

                        builder.setInsertionPoint(forOp);
                        FHEBatchCmpOp batchCmpOp;
                        
                        // Create RLWECipherType 
                        Type allocaType = insertVector.getType();
                        if (!allocaType.isa<LWECipherVectorType>()) return;
                        LWECipherVectorType lwevecType = dyn_cast<LWECipherVectorType>(allocaType); 
                        RLWECipherType rlweType = RLWECipherType::get(builder.getContext(), 
                                                    lwevecType.getPlaintextType(), lwevecType.getSize(),
                                                    lwevecType.getCipherNoise());
                        
                        if (cmpLhs == outsideOperand)
                            batchCmpOp = builder.create<FHEBatchCmpOp>(forOp.getLoc(), 
                                                rlweType, cmpPredicate, batchOperand, extractVector);
                        else if (cmpRhs == outsideOperand)
                            batchCmpOp = builder.create<FHEBatchCmpOp>(forOp.getLoc(), 
                                                rlweType, cmpPredicate, extractVector, batchOperand);
                        auto depackOp = builder.create<FHEDepackOp>(batchCmpOp.getLoc(), allocaType, batchCmpOp.getResult());
                        auto defineVectorOp = insertVector.getDefiningOp();
                        defineVectorOp->replaceAllUsesWith(depackOp);
                    }


                }
            }
        }
        if (forOp.use_empty()) forOp.erase();
        return;
    }

    // For Arithmetic Loop

    // Identify if-else branch pattern
    Value iterArg;
    Value condition;
    LWESubOp subOp;
    LWEMulOp elseMulOp;
    LWEAddOp aggOp;
    LWEMulOp ifMulOp;
    LWEAddOp ifAggOp;
    Value ifAggData;
    identifyElseBranch(forOp, subOp, elseMulOp, aggOp, iterArg, condition);

    // subOp.print(llvm::outs());
    // llvm::outs() << "\n";
    // elseMulOp.print(llvm::outs());
    // llvm::outs() << "\n";
    // aggOp.print(llvm::outs());
    // llvm::outs() << "\n\n\n";

    if (elseMulOp && aggOp)
        identifyIfBranch(forOp, aggOp, elseMulOp, iterArg, condition, ifMulOp, ifAggOp);
    
    if (ifMulOp && ifAggOp) {
        for (auto ifAggOperand : ifAggOp.getOperands()) {
            if (ifAggOperand != iterArg) ifAggData = ifAggOperand;
        }
        builder.setInsertionPoint(aggOp);
        auto filterMulOp = builder.create<LWEMulOp>(aggOp.getLoc(), condition.getType(), 
                                ValueRange({condition, ifAggData}), aggOp.getNoise());
        auto newAggOp = builder.create<LWEAddOp>(aggOp.getLoc(), iterArg.getType(), 
                                ValueRange({iterArg, filterMulOp.getResult()}), aggOp.getNoise());
        aggOp.replaceAllUsesWith(newAggOp.getResult());
    }
    
    
    // identify Aggregation Pattern
    bool isIterated = false;
    LWECipherVectorType lwevecType;
    RLWECipherType rlweType;
    Value aggData; 
    for (auto &op : *bodyBlock) {
        if (auto yieldOp = dyn_cast_or_null<AffineYieldOp>(&op)) {
            if (yieldOp.getNumOperands() != 1) return;
            Value yieldInput = yieldOp.getOperand(0);
            aggOp = dyn_cast_or_null<LWEAddOp>(yieldInput.getDefiningOp());
            if (aggOp.getOperand(0) == iterArgs.front()) 
                ifAggData = aggOp.getOperand(1); 
            else if (aggOp.getOperand(1) == iterArgs.front()) 
                ifAggData = aggOp.getOperand(0);
            else return;

            // Pure Summation
            if (auto loadOp = dyn_cast_or_null<FHEExtractOp>(ifAggData.getDefiningOp())) {
                isIterated = isIterationOp(loadOp, inducVar);
                Value loadVector = loadOp.getVector();
                lwevecType = dyn_cast<LWECipherVectorType>(loadVector.getType()); 
                rlweType = RLWECipherType::get(builder.getContext(), 
                                            lwevecType.getPlaintextType(), lwevecType.getSize(),
                                            lwevecType.getCipherNoise());
                builder.setInsertionPoint(forOp);
                auto packAggOp = builder.create<FHERepackOp>(forOp.getLoc(), rlweType, loadVector);
                aggData = packAggOp.getResult();
            }
            // Conditional Summation
            else if (auto aggMulOp = dyn_cast_or_null<LWEMulOp>(ifAggData.getDefiningOp())) {
                if (aggMulOp.getNumOperands() != 2) return;
                auto aggMulOperand0Op = dyn_cast_or_null<FHEExtractOp>(aggMulOp.getOperand(0).getDefiningOp());
                auto aggMulOperand1Op = dyn_cast_or_null<FHEExtractOp>(aggMulOp.getOperand(1).getDefiningOp());

                if (isIterationOp(aggMulOperand0Op, inducVar) && isIterationOp(aggMulOperand1Op, inducVar)) {
                    Value operand0 = aggMulOperand0Op.getVector();
                    Value operand1 = aggMulOperand1Op.getVector();
                    Value matOperand0, matOperand1;
                    
                    if (operand0.getType().isa<LWECipherVectorType>()) {
                        lwevecType = dyn_cast<LWECipherVectorType>(operand0.getType()); 
                        rlweType = RLWECipherType::get(builder.getContext(), 
                                                    lwevecType.getPlaintextType(), lwevecType.getSize(),
                                                    lwevecType.getCipherNoise());

                        builder.setInsertionPoint(forOp);
                        auto repackOp0 = builder.create<FHERepackOp>(forOp.getLoc(), rlweType, operand0, aggMulOp.getNoise());
                        matOperand0 = repackOp0.getResult();
                    }
                    else matOperand0 = operand0;

                    if (operand1.getType().isa<LWECipherVectorType>()) {
                        lwevecType = dyn_cast<LWECipherVectorType>(operand1.getType()); 
                        rlweType = RLWECipherType::get(builder.getContext(), 
                                                    lwevecType.getPlaintextType(), lwevecType.getSize(),
                                                    lwevecType.getCipherNoise());

                        builder.setInsertionPoint(forOp);
                        auto repackOp1 = builder.create<FHERepackOp>(forOp.getLoc(), rlweType, operand1, aggMulOp.getNoise());
                        matOperand1 = repackOp1.getResult();
                    }
                    else matOperand1 = operand1;

                    builder.setInsertionPoint(forOp);
                    auto batchMulOp = builder.create<RLWEMulOp>(forOp.getLoc(), rlweType, 
                                                    ValueRange({matOperand0, matOperand1}), aggMulOp.getNoise());
                    aggData = batchMulOp.getResult();
                }

            }

            // Add Rotate-Add sequence
            if (aggData) {
                unsigned iterTimes = rlweType.getSize();
                Value prev = aggData;
                Value added;
                APFloat rd_noise(-1.0);
                for (int i = iterTimes / 2; i > 0; i /= 2)
                {
                    auto rotated_down = builder.create<FHERotateOp>(forOp.getLoc(), prev.getType(), prev, i, rd_noise);
                    added = builder.create<RLWEAddOp>(forOp.getLoc(), prev.getType(), ValueRange({ prev, rotated_down }), rd_noise);
                    prev = added;
                }
                auto finalExtractOp = builder.create<FHEExtractfinalOp>(forOp.getLoc(), 
                                        forOp.getResult(0).getType(), added, 
                                        IntegerAttr::get(IndexType::get(builder.getContext()),0),
                                        Attribute(), rd_noise);
                forOp.replaceAllUsesWith(finalExtractOp);
            }
        }
    }
}

void BatchLoopPass::runOnOperation()
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
            BatchLoop(op);
        }
    }
    
}