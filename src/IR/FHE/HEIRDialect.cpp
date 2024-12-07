/*
Authors: HECO
Modified by Zian Zhao
Copyright:
Copyright (c) 2020 ETH Zurich.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "heir/IR/FHE/HEIRDialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace heir;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heir/IR/FHE/HEIRTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "heir/IR/FHE/HEIR.cpp.inc"

/// simplifies away materialization(materialization(x)) to x if the types work
::mlir::OpFoldResult heir::FHEMaterializeOp::fold(heir::FHEMaterializeOp::FoldAdaptor adaptor)
{
    if (auto m_op = getInput().getDefiningOp<heir::FHEMaterializeOp>()) {
        if (m_op.getInput().getType() == getResult().getType())
            return m_op.getInput();
        if (auto mm_op = m_op.getInput().getDefiningOp<heir::FHEMaterializeOp>()) {
            if (mm_op.getInput().getType() == getResult().getType())
                return mm_op.getInput();
        }
    }
    else if (getInput().getType() == getResult().getType()) {
        return getInput();
    }
    // To eliminate the i1 Type generated in unbraching (if) process
    else if (auto m_op = getInput().getDefiningOp<arith::UIToFPOp>()) {
        if (auto mm_op = m_op.getIn().getDefiningOp<FHEMaterializeOp>()) {
            if (mm_op.getInput().getType() == getResult().getType())
                return mm_op.getInput();
        }
    }
    return {};
    

}

::mlir::OpFoldResult heir::FHERepackOp::fold(heir::FHERepackOp::FoldAdaptor adaptor)
{
    if (auto m_op = getInput().getDefiningOp<heir::FHEDepackOp>()) {
        if (m_op.getInput().getType() == getResult().getType())
            return m_op.getInput();
    }
    return {};

}

::mlir::OpFoldResult heir::FHEDepackOp::fold(heir::FHEDepackOp::FoldAdaptor adaptor)
{
    if (auto m_op = getInput().getDefiningOp<heir::FHERepackOp>()) {
        if (m_op.getInput().getType() == getResult().getType())
            return m_op.getInput();
    }
    return {};

}

/// simplify rotate(cipher, 0) to cipher
::mlir::OpFoldResult heir::FHERotateOp::fold(heir::FHERotateOp::FoldAdaptor adaptor)
{
    if (getI() == 0)
        return getCipher();

    return {};
}

// /// simplifies add(x,0) and add(x) to x
// ::mlir::OpFoldResult heir::RLWEAddOp::fold(heir::RLWEAddOp::FoldAdaptor adaptor)
// {
//     // auto neutral_element = 0;
//     // SmallVector<Value> new_operands;
//     // for (auto v : getX())
//     // {
//     //     bool omit = false;
//         // if (auto cst_op = v.getDefiningOp<FHEEncodeOp>())
//         // {
//         //     if (auto dea = cst_op.getMessage().dyn_cast_or_null<Float32Attr>())
//         //     {
//         //         if (dea.size() == 1)
//         //         {
//         //             if (dea.getElementType().isIntOrIndex())
//         //             {
//         //                 if (dea.value_begin<const IntegerAttr>()->getInt() == neutral_element)
//         //                     omit = true;
//         //             }
//         //             else if (dea.getElementType().isIntOrFloat())
//         //             {
//         //                 // because we've already excluded IntOrIndex, it must be float
//         //                 if (dea.value_begin<const FloatAttr>()->getValueAsDouble() == neutral_element)
//         //                     omit = true;
//         //             }
//         //         }
//         //     }
//         //     else if (auto ia = cst_op.value().dyn_cast_or_null<IntegerAttr>())
//         //     {
//         //         if (ia.getInt() == neutral_element)
//         //             omit = true;
//         //     }
//         //     else if (auto fa = cst_op.value().dyn_cast_or_null<FloatAttr>())
//         //     {
//         //         if (fa.getValueAsDouble() == neutral_element)
//         //             omit = true;
//         //     }
//         // }
//     //     if (!omit)
//     //         new_operands.push_back(v);
//     // }
//     // getXMutable().assign(new_operands);
//     if (getX().size() > 1)
//         return getResult();
//     else
//         return getX().front();
// }
// /// simplifies sub(x,0) and sub(x) to x
// ::mlir::OpFoldResult heir::RLWESubOp::fold(heir::RLWESubOp::FoldAdaptor adaptor)
// {
//     // auto neutral_element = 0;
//     // SmallVector<Value> new_operands;
//     // for (auto v : x())
//     // {
//     //     bool omit = false;
//     //     if (auto cst_op = v.getDefiningOp<FHEEncodeOp>())
//     //     {
//     //         if (auto dea = cst_op.value().dyn_cast_or_null<DenseElementsAttr>())
//     //         {
//     //             if (dea.size() == 1)
//     //             {
//     //                 if (dea.getElementType().isIntOrIndex())
//     //                 {
//     //                     if (dea.value_begin<const IntegerAttr>()->getInt() == neutral_element)
//     //                         omit = true;
//     //                 }
//     //                 else if (dea.getElementType().isIntOrFloat())
//     //                 {
//     //                     // because we've already excluded IntOrIndex, it must be float
//     //                     if (dea.value_begin<const FloatAttr>()->getValueAsDouble() == neutral_element)
//     //                         omit = true;
//     //                 }
//     //             }
//     //         }
//     //         else if (auto ia = cst_op.value().dyn_cast_or_null<IntegerAttr>())
//     //         {
//     //             if (ia.getInt() == neutral_element)
//     //                 omit = true;
//     //         }
//     //         else if (auto fa = cst_op.value().dyn_cast_or_null<FloatAttr>())
//     //         {
//     //             if (fa.getValueAsDouble() == neutral_element)
//     //                 omit = true;
//     //         }
//     //     }
//     //     if (!omit)
//     //         new_operands.push_back(v);
//     // }
//     // xMutable().assign(new_operands);
//     if (getX().size() > 1)
//         return getResult();
//     else
//         return getX().front();
// }
// /// simplifies mul(x,1) and mul(x) to x
// ::mlir::OpFoldResult heir::RLWEMulOp::fold(heir::RLWEMulOp::FoldAdaptor adaptor)
// {
//     // auto neutral_element = 1;
//     // SmallVector<Value> new_operands;
//     // for (auto v : x())
//     // {
//     //     bool omit = false;
//     //     if (auto cst_op = v.getDefiningOp<FHEEncodeOp>())
//     //     {
//     //         if (auto dea = cst_op.value().dyn_cast_or_null<DenseElementsAttr>())
//     //         {
//     //             if (dea.size() == 1)
//     //             {
//     //                 if (dea.getElementType().isIntOrIndex())
//     //                 {
//     //                     if (dea.value_begin<const IntegerAttr>()->getInt() == neutral_element)
//     //                         omit = true;
//     //                 }
//     //                 else if (dea.getElementType().isIntOrFloat())
//     //                 {
//     //                     // because we've already excluded IntOrIndex, it must be float
//     //                     if (dea.value_begin<const FloatAttr>()->getValueAsDouble() == neutral_element)
//     //                         omit = true;
//     //                 }
//     //             }
//     //         }
//     //         else if (auto ia = cst_op.value().dyn_cast_or_null<IntegerAttr>())
//     //         {
//     //             if (ia.getInt() == neutral_element)
//     //                 omit = true;
//     //         }
//     //         else if (auto fa = cst_op.value().dyn_cast_or_null<FloatAttr>())
//     //         {
//     //             if (fa.getValueAsDouble() == neutral_element)
//     //                 omit = true;
//     //         }
//     //     }
//     //     if (!omit)
//     //         new_operands.push_back(v);
//     // }
//     // xMutable().assign(new_operands);
//     if (getX().size() > 1)
//         return getResult();
//     else
//         return getX().front();
// }

//===----------------------------------------------------------------------===//
// FHE dialect definitions
//===----------------------------------------------------------------------===//
#include "heir/IR/FHE/HEIRDialect.cpp.inc"
void HEIRDialect::initialize()
{
    // Registers all the Types into the FHEDialect class
    addTypes<
#define GET_TYPEDEF_LIST
#include "heir/IR/FHE/HEIRTypes.cpp.inc"
        >();

    // Registers all the Operations into the FHEDialect class
    addOperations<
#define GET_OP_LIST
#include "heir/IR/FHE/HEIR.cpp.inc"
        >();
}