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
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::arith;

namespace {
struct ConvertAddIOpToSubIOp : public ConversionPattern {

    ConvertAddIOpToSubIOp (MLIRContext *ctx, PatternBenefit benefit = 1)
	    : ConversionPattern(AddIOp::getOperationName(), benefit, ctx) {}

    LogicalResult match(Operation *op) const final {
      return cast<AddIOp>(op) ? success () : failure ();
    }

    void rewrite(Operation *op, ArrayRef<Value> operands,
		    ConversionPatternRewriter &rewriter) const final {
	    rewriter.replaceOpWithNewOp <SubIOp> (op, op->getResultTypes (),
			    operands, op->getAttrs ());
    }
  };
}

namespace {
struct MyPass
    : public MyPassBase<MyPass> {

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    ConversionTarget target(*ctx);
    target.addLegalDialect<ArithmeticDialect>();
    target.addIllegalOp<AddIOp>();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertAddIOpToSubIOp>(ctx);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<Pass>
mlir::arith::createMyPass() {
  return std::make_unique<MyPass>();
}
