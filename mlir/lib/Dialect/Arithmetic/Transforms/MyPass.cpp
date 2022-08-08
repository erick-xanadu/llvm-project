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

template <typename SrcOp>
static Value
createLinalgBodyCalculationForElementwiseOp(Operation *op, ValueRange args,
                                            ArrayRef<Type> resultTypes,
                                            PatternRewriter &rewriter) {
  Location loc = op->getLoc();

  if (isa<arith::CmpFOp>(op))
  {
    CmpFOp cmp= cast<arith::CmpFOp> (op);
    return rewriter.create<CmpFOp>(loc, cmp.getPredicate(), args[0], args[1]);
  } else if (isa<arith::CmpIOp>(op))
  {
    CmpIOp cmp = cast<arith::CmpIOp> (op);
    return rewriter.create<CmpIOp>(loc, cmp.getPredicate(), args[0], args[1]);
  }

  if (isa<SrcOp>(op))
    return rewriter.create<SrcOp>(loc, resultTypes, args);

  return nullptr;
}

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

template <typename SrcOp>
static LogicalResult
elementwiseMatchAndRewriteHelper(Operation *op,
                                 PatternRewriter &rewriter) {
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
          Value opResult = createLinalgBodyCalculationForElementwiseOp<SrcOp>(
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
    return success ();
}

namespace {
template <typename SrcOp>
struct PointwiseConverter : public OpRewritePattern <SrcOp> {

  using OpRewritePattern<SrcOp>::OpRewritePattern;


  LogicalResult matchAndRewrite(SrcOp op, 
               PatternRewriter &rewriter) const final {

    bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
    if (!hasTensorResult)
	    return failure ();

    return elementwiseMatchAndRewriteHelper<SrcOp> (op, rewriter);
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
    // constant
    patterns.add<PointwiseConverter<DivFOp>>(ctx);
    patterns.add<PointwiseConverter<DivSIOp>>(ctx);
    patterns.add<PointwiseConverter<DivUIOp>>(ctx);
    patterns.add<PointwiseConverter<ExtFOp>>(ctx);
    patterns.add<PointwiseConverter<ExtSIOp>>(ctx);
    patterns.add<PointwiseConverter<ExtUIOp>>(ctx);
    patterns.add<PointwiseConverter<FPToSIOp>>(ctx);
    patterns.add<PointwiseConverter<FPToUIOp>>(ctx);
    patterns.add<PointwiseConverter<FloorDivSIOp>>(ctx);
    // index cast: Implemented by someone else
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
    patterns.add<PointwiseConverter<ShLIOp>>(ctx);
    patterns.add<PointwiseConverter<ShRSIOp>>(ctx);
    patterns.add<PointwiseConverter<ShRUIOp>>(ctx);
    patterns.add<PointwiseConverter<SubFOp>>(ctx);
    patterns.add<PointwiseConverter<SubIOp>>(ctx);
    patterns.add<PointwiseConverter<TruncFOp>>(ctx);
    patterns.add<PointwiseConverter<TruncIOp>>(ctx);
    patterns.add<PointwiseConverter<UIToFPOp>>(ctx);
    patterns.add<PointwiseConverter<XOrIOp>>(ctx);
    // select

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::arith::createMyPass() {
  return std::make_unique<MyPass>();
}
