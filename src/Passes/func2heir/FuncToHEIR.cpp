// Author: Zian Zhao
#include "heir/Passes/func2heir/FuncToHEIR.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace arith;
using namespace affine;
using namespace heir;

void FuncToHEIRPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithDialect>();
    registry.insert<AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<HEIRDialect>();
}

// Take data type conversion for FHE extract operations and
// Convert FHEExtractOp to FHEExtractfinalOp
class FHEExtractPattern final : public OpConversionPattern<FHEExtractOp>
{
public:
    using OpConversionPattern<FHEExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEExtractOp op, typename FHEExtractOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType)
            return failure();
        
        auto indices_size = op.getI().size();

        // LWECipherVector: one-dimensional data
        if (indices_size == 1) {
            auto cOp = op.getI().front().getDefiningOp<arith::ConstantOp>();
            if (!cOp)
            {
                emitError(op.getLoc(),
                        "cannot find the definition of index in heir.extract_init op!");
                return failure();
            }
            auto indexAttr = cOp.getValue().cast<IntegerAttr>();

            rewriter.replaceOpWithNewOp<FHEExtractfinalOp>(op, dstType, op.getVector(), indexAttr, Attribute(),
            FloatAttr::get(Float64Type::get(rewriter.getContext()), -1.0));
        }
        // LWECipherMatrix: two-dimensional data
        else if (indices_size == 2) {
            auto row_cOp = op.getI().front().getDefiningOp<arith::ConstantOp>();
            auto col_cOp = op.getI().back().getDefiningOp<arith::ConstantOp>();

            if(!row_cOp || !col_cOp) {
                emitError(op.getLoc(),
                        "cannot find the definition of indices in heir.extract_init op!");
                return failure();
            }
            auto row_indexAttr = row_cOp.getValue().cast<IntegerAttr>();
            auto col_indexAttr = col_cOp.getValue().cast<IntegerAttr>();

            llvm::SmallVector<Attribute> materialized_index;
            materialized_index.push_back(row_indexAttr);
            materialized_index.push_back(col_indexAttr);

            rewriter.replaceOpWithNewOp<FHEExtractfinalOp>(op, dstType, op.getVector(), col_indexAttr, row_indexAttr,
            FloatAttr::get(Float64Type::get(rewriter.getContext()), -1.0));
        }

        return success();
    }
};

// Take data type conversion for FHE insert operations and
// Convert FHEInsertOp to FHEInsertfinalOp
class FHEInsertPattern final : public OpConversionPattern<FHEInsertOp>
{
public:
    using OpConversionPattern<FHEInsertOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEInsertOp op, typename FHEInsertOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {   
        auto valueType = getTypeConverter()->convertType(op.getValue().getType());
        if (!valueType)
            return failure();
        auto new_value = typeConverter->materializeTargetConversion(rewriter, op.getValue().getLoc(), 
                                                                    valueType, op.getValue()); 

        auto indices_size = op.getIndex().size();
        
        // LWECipherVector: one-dimensional data
        if (indices_size == 1) {
            auto cOp = op.getIndex().front().getDefiningOp<arith::ConstantOp>();
            if (!cOp)
            {
                emitError(op.getLoc(),
                        "cannot find the definition of index in heir.extract_init op!");
                return failure();
            }
            auto indexAttr = cOp.getValue().cast<IntegerAttr>();
            rewriter.replaceOpWithNewOp<FHEInsertfinalOp>(op, op.getValue(), op.getMemref(), indexAttr, Attribute(),
            FloatAttr::get(Float64Type::get(rewriter.getContext()), -1.0));
        }
        // LWECipherMatrix: two-dimensional data
        else if (indices_size == 2) {
            auto row_cOp = op.getIndex().front().getDefiningOp<arith::ConstantOp>();
            auto col_cOp = op.getIndex().back().getDefiningOp<arith::ConstantOp>();

            if(!row_cOp || !col_cOp) {
                emitError(op.getLoc(),
                        "cannot find the definition of indices in heir.extract_init op!");
                return failure();
            }
            auto row_indexAttr = row_cOp.getValue().cast<IntegerAttr>();
            auto col_indexAttr = col_cOp.getValue().cast<IntegerAttr>();
            llvm::SmallVector<Attribute> materialized_index;
            materialized_index.push_back(row_indexAttr);
            materialized_index.push_back(col_indexAttr);

            rewriter.replaceOpWithNewOp<FHEInsertfinalOp>(op, new_value, op.getMemref(), col_indexAttr, row_indexAttr,
            FloatAttr::get(Float64Type::get(rewriter.getContext()), -1.0));
        }

        return success();
    }
};

// Convert the input/output data type of FHEVectorLoadOp
class FHEVectorLoadPattern final : public OpConversionPattern<FHEVectorLoadOp>
{
public:
    using OpConversionPattern<FHEVectorLoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEVectorLoadOp op, typename FHEVectorLoadOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType) 
            return failure();
        
        auto memrefType = typeConverter->convertType(op.getMemref().getType());
        auto new_memref = typeConverter->materializeTargetConversion(rewriter, op.getMemref().getLoc(),
                                                                        memrefType, op.getMemref());
        
        auto cOp = op.getIndices().getDefiningOp<arith::ConstantOp>();
        if (!cOp)
        {
            emitError(op.getLoc(),
                    "cannot find the definition of index in heir.extract_init op!");
            return failure();
        }
        auto indexAttr = cOp.getValue().cast<IntegerAttr>();

        rewriter.replaceOpWithNewOp<FHEVectorLoadfinalOp>(op, dstType, new_memref, 
                    indexAttr, FloatAttr::get(Float64Type::get(rewriter.getContext()), -1.0));

        return success();

    }

};

// Convert types of arguments into ciphertext types and transform func::CallOp
// to heir::FHEFuncCallOp
class FuncCallPattern final : public OpConversionPattern<func::CallOp>
{
protected:
    using OpConversionPattern<func::CallOp>::typeConverter;
public:
    using OpConversionPattern<func::CallOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::CallOp op, typename func::CallOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
        // Only support one result
        rewriter.setInsertionPoint(op);

        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType) {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                // assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
                materialized_operands.push_back(o);
        }
        auto func_name = op.getCallee();

        if (op.getNumResults() > 1)
            return failure();
        if (op.getNumResults() == 1) {
            auto resultType = getTypeConverter()->convertType(op.getResult(0).getType());
            if (!resultType)
                return failure();
            
            // for fun.call@sgn() in min_index testbench
            if (func_name == "sgn") {
                if (op.getNumOperands() == 1) {
                    rewriter.replaceOpWithNewOp<FHELUTForGTOp>(op, resultType, materialized_operands.front(),
                            FloatAttr::get(Float64Type::get(rewriter.getContext()), -1.0));
                    return success();
                }
                else 
                    return failure();
            }

            if (func_name == "lut_lsz") {
                if (op.getNumOperands() == 1) {
                    rewriter.replaceOpWithNewOp<FHELUTForLTOp>(op, resultType, materialized_operands.front(),
                            FloatAttr::get(Float64Type::get(rewriter.getContext()), -1.0));
                    return success();
                }
                else 
                    return failure();
            }
        
            rewriter.replaceOpWithNewOp<FHEFuncCallOp>(op, 
                TypeRange(resultType), func_name, ArrayAttr(), ArrayAttr(), materialized_operands);
            
            // // To determine RNS modulus for TFHE Bootstrapping
            // std::string func_name_str = func_name.str();
            // size_t found = func_name_str.find("lut");
            // if (found == std::string::npos)
            //     rewriter.replaceOpWithNewOp<FHEFuncCallOp>(op, 
            //         TypeRange(resultType), func_name, ArrayAttr(), ArrayAttr(), materialized_operands);
            // else {
            //     auto result = op.getResult(0);
            //     while (!result.getUses().empty()) {
            //         // auto resUses = result.getUses();
            //         for (OpOperand &u: result.getUses()) {
            //             Operation *owner = u.getOwner();
            //         }
            //     }
            // }

        } else {
            return failure();
        }
        
        return success();
    }
};

// convert the type of return value in a function block
class ReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    if (op.getNumOperands() != 1)
    {
      emitError(op.getLoc(), "Currently only single value return operations are supported.");
      return failure();
    }
    auto dstType = this->getTypeConverter()->convertType(op.getOperandTypes().front());
    if (!dstType)
      return failure();

    rewriter.setInsertionPoint(op);
    auto returnCipher = typeConverter->materializeTargetConversion(rewriter,
                                                                    op.getLoc(),
                                                                    dstType, op.getOperands());
    
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, returnCipher);

    return success();
  }
};

// Convert the types of function arguments in a function block
class FunctionConversionPattern final : public OpConversionPattern<func::FuncOp>
{
public:
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::FuncOp op, typename func::FuncOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // Compute the new signature of the function.
        TypeConverter::SignatureConversion signatureConversion(op.getFunctionType().getNumInputs());
        SmallVector<Type> newResultTypes;
        if (failed(typeConverter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return failure();
        if (typeConverter->convertSignatureArgs(op.getFunctionType().getInputs(), signatureConversion).failed())
            return failure();
        auto new_functype = FunctionType::get(getContext(), signatureConversion.getConvertedTypes(), newResultTypes);

        // https://github.com/llvm/llvm-project/pull/78260
        // rewriter.startRootUpdate(op);
        rewriter.startOpModification(op);
        op.setType(new_functype);
        for (auto it = op.getRegion().args_begin(); it != op.getRegion().args_end(); ++it)
        {
            auto arg = *it;
            auto oldType = arg.getType();
            auto newType = typeConverter->convertType(oldType);
            arg.setType(newType);
            if (newType != oldType)
            {
                rewriter.setInsertionPointToStart(&op.getBody().getBlocks().front());
                auto m_op = typeConverter->materializeSourceConversion(rewriter, arg.getLoc(), oldType, arg);
                arg.replaceAllUsesExcept(m_op, m_op.getDefiningOp());
            }
        }
        // rewriter.finalizeRootUpdate(op);
        rewriter.finalizeOpModification(op);

        return success();
    }
};

bool isLoopInvariant(Operation *op, AffineForOp forOp)
{
    auto numOperands = op->getNumOperands();
    if (numOperands == 0) {
        // forOp.moveOutOfLoop(op);
        return true;
    }
    // bool operandNotInLoop = true;
    // for (size_t i = 0; i < numOperands; i++) {
    //     if (!forOp.isDefinedOutsideOfLoop(op->getOperand(i)))
    //         operandNotInLoop = false;
    // }
    // return operandNotInLoop;
    return false;
}

LogicalResult ForTypeConversion(IRRewriter &rewriter, MLIRContext *context, 
                                    AffineForOp op, TypeConverter typeConverter)
{
    rewriter.setInsertionPointAfter(op);
    
    auto lbOperands = op.getLowerBoundOperands();
    auto lbMap = op.getLowerBoundMap();
    auto ubOperands = op.getUpperBoundOperands();
    auto ubMap = op.getUpperBoundMap();
    auto inducVar = op.getInductionVar();
    auto iterArgs = op.getRegionIterArgs();
    auto inits = op.getInits();
    uint64_t step = op.getStep().getSExtValue();
    
    // auto results = op.getResults();

    // auto newForOp = rewriter.create<AffineForOp>(op.getLoc(), lbOperands, lbMap, 
    //                 ubOperands, ubMap, step, iterArgs);
    
    // for (auto &iterArg : iterArgs) {
        // auto definingOp = llvm::dyn_cast<ConstantOp>(iterArg.getDefiningOp());
        // iterArg.getType().print(llvm::outs());
        // if (auto definingOp = iterArg.getDefiningOp())
        //     definingOp->print(llvm::outs());
            // op.iter
        // llvm::outs() << "\n";
    // }
    
    llvm::SmallVector<Value> newInits;
    for (size_t i = 0; i < inits.size(); i++) {
        if (auto matOp = inits[i].getDefiningOp<FHEMaterializeOp>()) {
            // To ensure the init value of iterArg is identical to yield type
            // We have to add a MaterializeOp to convert the inital Type to LWECipherType
            if (auto encodeOp = matOp.getInput().getDefiningOp<FHEEncodeOp>()) {
                rewriter.setInsertionPointAfter(encodeOp);
                LWECipherType lweType = LWECipherType::get(context, Float32Type::get(context), 
                    NoiseType::get(context, FloatAttr::get(Float32Type::get(context), -1.0)));
                auto newMatOp = rewriter.create<FHEMaterializeOp>(
                        encodeOp.getLoc(), lweType, encodeOp.getResult());
                newInits.push_back(newMatOp.getResult());
                continue;
            }
        }
        newInits.push_back(inits[i]);
    }

    rewriter.setInsertionPointAfter(op);

    auto newForOp = rewriter.create<AffineForOp>(op.getLoc(), lbOperands, lbMap, 
                    ubOperands, ubMap, step, ValueRange(newInits));

    Block *bodyBlock = op.getBody();

    auto newIterArgs = newForOp.getRegionIterArgs();
    auto newInducVar = newForOp.getInductionVar();


    rewriter.setInsertionPointToStart(newForOp.getBody());

    for (auto &op : *bodyBlock) {
        if (AffineYieldOp yieldOp = dyn_cast_or_null<AffineYieldOp>(op)) {
            rewriter.setInsertionPoint(yieldOp);
            unsigned numOperand = yieldOp.getNumOperands();
            llvm::SmallVector<Value> materialized_operand; 
            auto forOpIt = bodyBlock->begin();
            auto newForOpIt = newForOp.getBody()->begin();
            for (unsigned i = 0; i < numOperand; i++) {
                for (; newForOpIt != newForOp.getBody()->end(); ++forOpIt, ++newForOpIt) {
                    Operation *oldOp = &*forOpIt;
                    Operation *newOp = &*newForOpIt;
                    if (yieldOp.getOperand(i) == oldOp->getResult(0)) {
                        if (auto matOp = dyn_cast<FHEMaterializeOp>(newOp)) 
                            materialized_operand.push_back(matOp.getInput());
                        else materialized_operand.push_back(newOp->getResult(0));
                    }
                }
            }
            rewriter.setInsertionPointToEnd(newForOp.getBody());
            rewriter.create<AffineYieldOp>(newForOp.getLoc(), materialized_operand);
            continue;
        }
        auto thisOp = rewriter.clone(op);

        for (unsigned i = 0; i < thisOp->getNumOperands(); ++i) {
            for (size_t i = 0; i < newIterArgs.size(); i++) {
                if (op.getOperand(i) == iterArgs[i]) {
                    if (auto matOp = dyn_cast<FHEMaterializeOp>(thisOp)) {
                        rewriter.replaceOpWithNewOp<FHEMaterializeOp>(matOp, matOp.getType(), newIterArgs[i]);
                    }
                }
            }
            if (op.getOperand(i) == inducVar) {
                thisOp->setOperand(i, newInducVar);
            }
            auto forOpIt = bodyBlock->begin();
            auto newForOpIt = newForOp.getBody()->begin();
            for (; newForOpIt != newForOp.getBody()->end(); ++forOpIt, ++newForOpIt) {
                Operation *oldOp = &*forOpIt;
                Operation *newOp = &*newForOpIt;
                if (op.getOperand(i) == oldOp->getResult(0)) thisOp->setOperand(i, newOp->getResult(0));    
            }
        }
    }

    rewriter.setInsertionPointAfter(newForOp);
    // auto matOp = rewriter.create<FHEMaterializeOp>(newForOp.getLoc(), op.getResult(0).getType(), newForOp.getResult(0));
    // op.replaceAllUsesWith(matOp.getResult());
    for (size_t i = 0; i < op.getNumResults(); i++) {
        Value result = op.getResult(i);
        auto matOp = rewriter.create<FHEMaterializeOp>(newForOp.getLoc(), result.getType(), newForOp.getResult(0));
        for (auto *user : result.getUsers()) {
            // user->print(llvm::outs());
            // user->replaceUsesOfWith(result, matOp.getResult());
            for (size_t j = 0; j < user->getNumOperands(); j++) {
                if (user->getOperand(j) == result) user->setOperand(j, matOp.getResult());
            }
        }
        // Execute twice, weird
        for (auto *user : result.getUsers()) {
            // user->print(llvm::outs());
            // user->replaceUsesOfWith(result, matOp.getResult());
            for (size_t j = 0; j < user->getNumOperands(); j++) {
                if (user->getOperand(j) == result) user->setOperand(j, matOp.getResult());
            }
        }
    }
    if (op.use_empty()) op.erase();

    
    // Move loop-invariant operations (such as ConstantOp) out of the loop
    // 遍历 block 中的操作
    // for (auto &op : *newForOp.getBody()) {
    // // 判断操作是否是 loop-invariant（例如：如果它依赖常量或不依赖循环变量）
    //     if (isLoopInvariant(&op, newForOp)) {
    //         op.print(llvm::outs());
    //         llvm::outs() << "Hello There\n";
    //         // newForOp.moveOutOfLoop(&op);
    //         rewriter.setInsertionPoint(newForOp);
    //         rewriter.moveOpBefore(&op, newForOp);
    //     }
    // }

    // newForOp.print(llvm::outs());
    // llvm::outs() << "\n" << "origin Op:";
    // op.print(llvm::outs());
    // mlir::Block *parentBlock = op->getBlock();
    
    // 获取 Block 所在的父级 Region
    // mlir::Region *parentRegion = parentBlock->getParent();

    // parentRegion.
    
    // 遍历 Region 来查找 funcOp
    // for (auto &parentOp : parentBlock->getOperations()) {
    //     // 如果 parentOp 是 funcOp，打印其内容
    //     // if (auto funcOp = llvm::dyn_cast_or_null<func::FuncOp>(parentOp)) {
    //     // llvm::outs() << "Found funcOp: ";
    //     parentOp.print(llvm::outs());
    //     llvm::outs() << "\n";
    //     // }
    // }


    return success();

}

// Transform Function Block and Function Call operations into FHE operations and
// ciphertext data types
void FuncToHEIRPass::runOnOperation()
{
    auto type_converter = TypeConverter();

    // Add type converter to convert plaintext data type to LWECiphertext type
    type_converter.addConversion([&](Type t) {
        if (t.isa<Float32Type>())
            return std::optional<Type>(LWECipherType::get(&getContext(), t,
                NoiseType::get(&getContext(), FloatAttr::get(Float32Type::get(&getContext()), -1.0))));
        else if (t.isa<MemRefType>())
        {
            int size = -155;
            auto new_t = t.cast<MemRefType>();
            if (new_t.hasStaticShape() && new_t.getShape().size()==1) {
                size = new_t.getShape().front();
                return std::optional<Type>(LWECipherVectorType::get(&getContext(), new_t.getElementType(), size,
                NoiseType::get(&getContext(), FloatAttr::get(Float32Type::get(&getContext()), -1.0))));
            }
            else if (new_t.hasStaticShape() && new_t.getShape().size()==2) {
                auto row = new_t.getShape().front();
                auto col = new_t.getShape().back();
                return std::optional<Type>(LWECipherMatrixType::get(&getContext(), new_t.getElementType(), row, col,
                NoiseType::get(&getContext(), FloatAttr::get(Float32Type::get(&getContext()), -1.0))));
            }
            else
                return std::optional<Type>(t);
        }
        else
            return std::optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<LWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<Float32Type>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherMatrixType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt);
    });
    type_converter.addArgumentMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<LWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<Float32Type>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherMatrixType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt);
    });
    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<Float32Type>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<LWECipherType>())
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<MemRefType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<LWECipherVectorType>())
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
            else if (auto ot = old_type.dyn_cast_or_null<LWECipherMatrixType>())
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
            else 
                return std::optional<Value>(std::nullopt);
        }
        return std::optional<Value>(std::nullopt);
    });
    
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect, ArithDialect>();
    target.addLegalDialect<HEIRDialect>();
    target.addLegalOp<ModuleOp>();
    // target.addIllegalOp<FHEExtractOp>();
    target.addIllegalOp<FHEInsertOp>();
    target.addIllegalOp<FHEVectorLoadOp>();
    // We cannot 'remove' FuncOp
    target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op) {
        auto fop = llvm::dyn_cast<func::FuncOp>(op);
        for (auto t : op->getOperandTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : op->getResultTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getInputs())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getResults())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        return true;
    });
    target.addIllegalOp<func::CallOp>();
    
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](Operation *op) { return type_converter.isLegal(op->getOperandTypes()); });

    // target.addDynamicallyLegalOp<FHEExtractfinalOp>(
    //     [&](Operation *op) {return type_converter.isLegal(op->getResultTypes()); });

    IRRewriter rewriter(&getContext());
    
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
            FHEInsertPattern, FunctionConversionPattern, 
                ReturnPattern, FuncCallPattern, FHEVectorLoadPattern>(type_converter, patterns.getContext()); 
    
    if (mlir::failed(mlir::applyFullConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();

    auto &block = getOperation()->getRegion(0).getBlocks().front();

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        // We must translate in order of appearance for this to work, so we walk manually
        if(f.walk([&](Operation *op)
                {
        if (AffineForOp for_op = llvm::dyn_cast_or_null<AffineForOp>(op)) {
            if (ForTypeConversion(rewriter, &getContext(), for_op, type_converter).failed())
                return WalkResult::interrupt();
        }
        return WalkResult(success()); })
                .wasInterrupted())
        signalPassFailure();
    }
    
}