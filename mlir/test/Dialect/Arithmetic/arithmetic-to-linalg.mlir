// RUN: mlir-opt -my-pass %s | FileCheck %s

module {
  func.func @transformaddf(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
// CHECK: %2 = arith.addf %arg2, %arg3 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformaddi(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.addi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.addi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }
}

