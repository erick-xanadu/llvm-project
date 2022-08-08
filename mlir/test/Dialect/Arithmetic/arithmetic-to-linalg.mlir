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

  func.func @transformandi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.andi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.andi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformceildivsi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.ceildivsi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.ceildivsi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformceildivui (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.ceildivui %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.ceildivui %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformdivf (%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.divf %arg0, %arg1 : tensor<4x4xf32>
// CHECK: %2 = arith.divf %arg2, %arg3 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformdivsi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.divsi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.divsi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformdivui (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.divui %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.divui %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformfloordivsi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.floordivsi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.floordivsi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformmaxf (%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.maxf %arg0, %arg1 : tensor<4x4xf32>
// CHECK: %2 = arith.maxf %arg2, %arg3 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformmaxsi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.maxsi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.maxsi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformmaxui (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.maxui %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.maxui %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

}

