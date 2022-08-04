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

using namespace mlir;
using namespace mlir::arith;

namespace {
struct MyPass
    : public MyPassBase<MyPass> {

  void runOnOperation() override {
  }
};
} // end anonymous namespace

std::unique_ptr<Pass>
mlir::arith::createMyPass() {
  return std::make_unique<MyPass>();
}
