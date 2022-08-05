//===- UnsignedWhenEquivalent.cpp - Pass to replace signed operations with
// unsigned
// ones when all their arguments and results are statically non-negative --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;

static SmallVector<StringRef> getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<StringRef>(nParallelLoops, getParallelIteratorTypeName());
}

static SmallVector<Value> condenseValues(const SmallVector<Value> &values) {
  SmallVector<Value> condensedValues;
  for (auto value : values)
    if (value)
      condensedValues.push_back(value);
  return condensedValues;
}

static Value
createLinalgBodyCalculationForElementwiseOp(Operation *op, ValueRange args,
                                            ArrayRef<Type> resultTypes,
                                            PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto elementTy =
      op->getOperand(0).getType().cast<ShapedType>().getElementType();

  if (isa<arith::AddIOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::AddIOp>(loc, resultTypes, args);

  return nullptr;
}

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

namespace {
struct ConvertAddIOpToSubIOp : public ConversionPattern {

  ConvertAddIOpToSubIOp(MLIRContext *ctx, PatternBenefit benefit = 1)
      : ConversionPattern(AddIOp::getOperationName(), benefit, ctx) {}

  LogicalResult match(Operation *op) const final {
    bool isAddIOp = cast<AddIOp>(op);
    if (!isAddIOp)
      return failure();

    // Check for the operands
    bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
    return hasTensorResult ? success() : failure();
  }

  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();

    assert(op->getNumResults() == 1 &&
           "All elementwise ops should only return a single result.");

    auto results = op->getResults();
    auto resultTy = op->getResult(0).getType().dyn_cast<ShapedType>();
    assert(resultTy && "All results must be a shaped type");

    unsigned rank = resultTy.getRank();

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Type> bodyArgTypes;

    for (Value in : op->getOperands())
      bodyArgTypes.emplace_back(getElementTypeOrSelf(in.getType()));

    SmallVector<Type> opResultTypes;
    SmallVector<Value> initTensors;

    SmallVector<Value> dynDims;
    dynDims.resize(results.front().getType().cast<ShapedType>().getRank());

    for (auto arg : op->getOperands()) {
      auto operandTy = arg.getType().cast<ShapedType>();
      for (int i = 0; i < operandTy.getRank(); i++)
	assert (!operandTy.isDynamicDim (i) && "Dimensions cannot be dynamic yet");
    }

    SmallVector<Value> filteredDims = condenseValues(dynDims);

    for (auto result : results) {
      auto resultTy = result.getType().template cast<ShapedType>();
      initTensors.push_back(rewriter.create<linalg::InitTensorOp>(
          loc, filteredDims, resultTy.getShape(), resultTy.getElementType()));
      opResultTypes.push_back(result.getType());
    }

    auto bodyResultTypes = llvm::to_vector<4>(llvm::map_range(
        initTensors, [](Value v) { return getElementTypeOrSelf(v); }));

    SmallVector<Value, 2> operands2;
    SmallVector<AffineMap, 3> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + bodyResultTypes.size());

    // Input indexing maps may be broadcasted.
    for (Value operand : op->getOperands()) {
      ShapedType type = operand.getType().cast<ShapedType>();

      if (type.getShape() == resultTy.getShape()) {
        operands2.push_back(operand);
        indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
        continue;
      }

      // this is for the result
      SmallVector<int64_t, 5> newShape;
      SmallVector<AffineExpr, 4> affineExprs;
      newShape.reserve(type.getRank());
      for (const auto &it : llvm::enumerate(type.getShape())) {
        if (it.value() == resultTy.getDimSize(it.index())) {
          newShape.push_back(it.value());
          affineExprs.push_back(
              mlir::getAffineDimExpr(it.index(), rewriter.getContext()));
        }
      }

      assert(newShape.size() == rank && "New shape must have same rank");

      operands2.push_back(operand);
      indexingMaps.push_back(AffineMap::get(
          /*dimCount=*/type.getRank(), /*symbolCount=*/0, affineExprs,
          rewriter.getContext()));
    }

    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    bool didEncounterError = false;
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, opResultTypes, operands2, initTensors, indexingMaps,
        getNParallelLoopsAttrs(rank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value opResult = createLinalgBodyCalculationForElementwiseOp(
              op, blockArgs.take_front(op->getNumOperands()), bodyResultTypes,
              rewriter);
          if (!opResult) {
            didEncounterError = true;
            return;
          }
          nestedBuilder.create<linalg::YieldOp>(loc, opResult);
        });

    assert(!didEncounterError && "Must not encounter errors");

    rewriter.replaceOp(op, linalgOp->getResults());
  }
};
} // namespace

namespace {
struct MyPass : public MyPassBase<MyPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    ConversionTarget target(*ctx);
    target.addLegalDialect<ArithmeticDialect, LinalgDialect>();
    target.addDynamicallyLegalOp<AddIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertAddIOpToSubIOp>(ctx);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::arith::createMyPass() {
  return std::make_unique<MyPass>();
}
