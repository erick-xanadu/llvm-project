//===- ArithmeticToLinalg.cpp - Arithmetic to Linalg dialect conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithmeticToLinalg/ArithmeticToLinalg.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::tensor;
using namespace mlir::cf;

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

template <typename SrcOp>
static Value
createLinalgBodyCalculationForElementwiseOp(Operation *op, ValueRange args,
                                            ArrayRef<Type> resultTypes,
                                            PatternRewriter &rewriter) {
  Location loc = op->getLoc();

  if (isa<arith::CmpFOp>(op)) {
    CmpFOp cmp = cast<arith::CmpFOp>(op);
    return rewriter.create<CmpFOp>(loc, cmp.getPredicate(), args[0], args[1]);
  } else if (isa<arith::CmpIOp>(op)) {
    CmpIOp cmp = cast<arith::CmpIOp>(op);
    return rewriter.create<CmpIOp>(loc, cmp.getPredicate(), args[0], args[1]);
  }

  if (isa<SrcOp>(op))
    return rewriter.create<SrcOp>(loc, resultTypes, args);

  return nullptr;
}

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

template <typename SrcOp>
static LogicalResult
elementwiseMatchAndRewriteHelper(Operation *operation,
                                 PatternRewriter &rewriter) {
  auto loc = operation->getLoc();

  assert(operation->getNumResults() == 1 &&
         "All arith elementwise ops should only return a single result.");

  auto results = operation->getResults();
  auto resultTy = operation->getResult(0).getType().dyn_cast<ShapedType>();

  if (!resultTy)
    return rewriter.notifyMatchFailure(operation,
                                       "All results must be a shaped type");

  unsigned rank = resultTy.getRank();

  // Construct the indexing maps needed for linalg.generic ops.
  SmallVector<Type> bodyArgTypes;

  for (Value in : operation->getOperands())
    bodyArgTypes.emplace_back(getElementTypeOrSelf(in.getType()));

  SmallVector<Type> opResultTypes;
  SmallVector<Value> initTensors;

  SmallVector<Value> dynDims;
  dynDims.resize(results.front().getType().cast<ShapedType>().getRank());

  assert(operation->getOperand(0) && "There must be at least one operand.");
  auto arg0 = operation->getOperand(0);
  auto arg0Rank = arg0.getType().cast<ShapedType>().getRank();
  for (auto arg : operation->getOperands()) {
    auto operandRank = arg.getType().cast<ShapedType>().getRank();
    assert(arg0Rank == operandRank && "All operands must match in rank.");
  }

  for (int i = 0; i < arg0Rank; i++) {
    bool isCurrentDimensionDynamic = false;
    for (auto arg : operation->getOperands()) {
      auto operandTy = arg.getType().cast<ShapedType>();
      isCurrentDimensionDynamic |= operandTy.isDynamicDim(i);
    }

    // Either operand could be dynamic
    if (isCurrentDimensionDynamic) {
      // We need to create a constant integer index
      // that will allow us to use the DimOp operator...
      auto indexVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(), i));
      SmallVector<Value> dimensionsToCompare;
      dimensionsToCompare.resize(operation->getNumOperands());

      // Which are the dimensions to compare?
      unsigned int j = 0;
      for (auto arg : operation->getOperands()) {
        dimensionsToCompare[j] =
            rewriter.create<tensor::DimOp>(loc, arg, indexVal);
        j++;
      }

      // Get the zeroth dimension size
      j = 0;
      auto zerothDimensionToCompare = dimensionsToCompare[j++];

      // Re-use this DimOp for the initTensors.
      // If the results are not equal, it won't matter since
      // we will hit the assertion.
      if (!dynDims[i]) {
        dynDims[i] = zerothDimensionToCompare;
      }

      for (; j < operation->getNumOperands(); j++) {
        auto jthDimensionToCompare = dimensionsToCompare[j];
        auto value = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, zerothDimensionToCompare,
            jthDimensionToCompare);
        rewriter.create<mlir::cf::AssertOp>(
            loc, value,
            rewriter.getStringAttr("Dimensions have to be same size."));
      }
    }
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

  SmallVector<Value, 2> operands;
  SmallVector<AffineMap, 2> indexingMaps;
  indexingMaps.reserve(operation->getNumOperands() + bodyResultTypes.size());

  // Input indexing maps may be broadcasted.
  for (Value operand : operation->getOperands()) {
    ShapedType type = operand.getType().cast<ShapedType>();

    if (type.getShape() == resultTy.getShape()) {
      operands.push_back(operand);
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
      continue;
    }

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

    assert(newShape.size() == rank && "New shape must have the same rank.");

    operands.push_back(operand);
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/type.getRank(), /*symbolCount=*/0, affineExprs,
        rewriter.getContext()));
  }

  indexingMaps.append(operation->getNumResults(),
                      rewriter.getMultiDimIdentityMap(rank));

  bool didEncounterError = false;
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, opResultTypes, operands, initTensors, indexingMaps,
      getNParallelLoopsAttrs(rank),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value opResult = createLinalgBodyCalculationForElementwiseOp<SrcOp>(
            operation, blockArgs.take_front(operation->getNumOperands()),
            bodyResultTypes, rewriter);
        if (!opResult) {
          didEncounterError = true;
          return;
        }
        nestedBuilder.create<linalg::YieldOp>(loc, opResult);
      });

  if (didEncounterError)
    return failure();

  rewriter.replaceOp(operation, linalgOp->getResults());
  return success();
}

namespace {
template <typename SrcOp>
struct PointwiseConverter : public OpRewritePattern<SrcOp> {

  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const final {

    bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
    if (!hasTensorResult)
      return failure();

    return elementwiseMatchAndRewriteHelper<SrcOp>(op, rewriter);
  }
};
} // namespace

namespace {
struct ConvertArithmeticToLinalgPass
    : public ConvertArithmeticToLinalgBase<ConvertArithmeticToLinalgPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    ConversionTarget target(*ctx);
    target.addLegalDialect<ArithmeticDialect, LinalgDialect, TensorDialect,
                           ControlFlowDialect>();

    target.addDynamicallyLegalOp<AddFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<AddIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<AndIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<BitcastOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    // ConstantOp : Bufferization is achieved through BufferizableOpInterface
    target.addDynamicallyLegalOp<CeilDivSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<CeilDivUIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<CmpFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<CmpIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<DivFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<DivSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<DivUIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<ExtFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<ExtSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<ExtUIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<FPToSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<FPToUIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<FloorDivSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    // IndexCastOp: Bufferization is achieved through BufferizableOpInterface
    target.addDynamicallyLegalOp<MaxFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<MaxSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<MaxUIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<MinFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<MinSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<MinUIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<MulFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<MulIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<NegFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<OrIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<RemFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<RemSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<RemUIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<SIToFPOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    // SelectOp: Bufferization is achieved through BufferizableOpInterface
    target.addDynamicallyLegalOp<ShLIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<ShRSIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<ShRUIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<SubFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<SubIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<TruncFOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<TruncIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<UIToFPOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });
    target.addDynamicallyLegalOp<XOrIOp>([&](Operation *op) {
      return !any_of(op->getResultTypes(), isaTensor);
    });

    RewritePatternSet patterns(ctx);

    patterns.add<PointwiseConverter<AddFOp>>(ctx);
    patterns.add<PointwiseConverter<AddIOp>>(ctx);
    patterns.add<PointwiseConverter<AndIOp>>(ctx);
    patterns.add<PointwiseConverter<BitcastOp>>(ctx);
    patterns.add<PointwiseConverter<CeilDivSIOp>>(ctx);
    patterns.add<PointwiseConverter<CeilDivUIOp>>(ctx);
    patterns.add<PointwiseConverter<CmpFOp>>(ctx);
    patterns.add<PointwiseConverter<CmpIOp>>(ctx);
    // ConstantOp : Bufferization is achieved through BufferizableOpInterface
    patterns.add<PointwiseConverter<DivFOp>>(ctx);
    patterns.add<PointwiseConverter<DivSIOp>>(ctx);
    patterns.add<PointwiseConverter<DivUIOp>>(ctx);
    patterns.add<PointwiseConverter<ExtFOp>>(ctx);
    patterns.add<PointwiseConverter<ExtSIOp>>(ctx);
    patterns.add<PointwiseConverter<ExtUIOp>>(ctx);
    patterns.add<PointwiseConverter<FPToSIOp>>(ctx);
    patterns.add<PointwiseConverter<FPToUIOp>>(ctx);
    patterns.add<PointwiseConverter<FloorDivSIOp>>(ctx);
    // IndexCastOp: Bufferization is achieved through BufferizableOpInterface
    patterns.add<PointwiseConverter<MaxFOp>>(ctx);
    patterns.add<PointwiseConverter<MaxSIOp>>(ctx);
    patterns.add<PointwiseConverter<MaxUIOp>>(ctx);
    patterns.add<PointwiseConverter<MinFOp>>(ctx);
    patterns.add<PointwiseConverter<MinSIOp>>(ctx);
    patterns.add<PointwiseConverter<MinUIOp>>(ctx);
    patterns.add<PointwiseConverter<MulFOp>>(ctx);
    patterns.add<PointwiseConverter<MulIOp>>(ctx);
    patterns.add<PointwiseConverter<NegFOp>>(ctx);
    patterns.add<PointwiseConverter<OrIOp>>(ctx);
    patterns.add<PointwiseConverter<RemFOp>>(ctx);
    patterns.add<PointwiseConverter<RemSIOp>>(ctx);
    patterns.add<PointwiseConverter<RemUIOp>>(ctx);
    patterns.add<PointwiseConverter<SIToFPOp>>(ctx);
    // SelectOp: Bufferization is achieved through BufferizableOpInterface
    patterns.add<PointwiseConverter<ShLIOp>>(ctx);
    patterns.add<PointwiseConverter<ShRSIOp>>(ctx);
    patterns.add<PointwiseConverter<ShRUIOp>>(ctx);
    patterns.add<PointwiseConverter<SubFOp>>(ctx);
    patterns.add<PointwiseConverter<SubIOp>>(ctx);
    patterns.add<PointwiseConverter<TruncFOp>>(ctx);
    patterns.add<PointwiseConverter<TruncIOp>>(ctx);
    patterns.add<PointwiseConverter<UIToFPOp>>(ctx);
    patterns.add<PointwiseConverter<XOrIOp>>(ctx);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::arith::createConvertArithmeticToLinalgPass() {
  return std::make_unique<ConvertArithmeticToLinalgPass>();
}
