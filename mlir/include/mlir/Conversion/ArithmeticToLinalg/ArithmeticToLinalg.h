//===- ArithmeticToLinalg.h - Arith to Linalg dialect conversion *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHMETICTOLINALG_ARITHMETICTOLINALG_H
#define MLIR_CONVERSION_ARITHMETICTOLINALG_ARITHMETICTOLINALG_H

#include <memory>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace arith {
std::unique_ptr<Pass> createConvertArithmeticToLinalgPass();
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHMETICTOLINALG_ARITHMETICTOLINALG_H
