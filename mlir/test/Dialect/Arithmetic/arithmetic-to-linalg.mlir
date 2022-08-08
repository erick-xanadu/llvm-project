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

  func.func @transformcmpf (%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.cmpf ogt, %arg0, %arg1 : tensor<4x4xf32>
// CHECK: %2 = arith.cmpf ogt, %arg2, %arg3 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi1>
  }

  func.func @transformcmpi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi1> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.cmpi sge, %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.cmpi sge, %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi1>
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

  func.func @transformextf (%arg0: tensor<4x4xf32>) -> tensor<4x4xf64> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.extf %arg0 : tensor<4x4xf32> to tensor<4x4xf64>
// CHECK: %2 = arith.extf %arg1 : f32 to f64
// CHECK: linalg.yield
    return %0 : tensor<4x4xf64>
  }

  func.func @transformextsi (%arg0: tensor<4x4xi32>) -> tensor<4x4xi64> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.extsi %arg0 : tensor<4x4xi32> to tensor<4x4xi64>
// CHECK: %2 = arith.extsi %arg1 : i32 to i64
// CHECK: linalg.yield
    return %0 : tensor<4x4xi64>
  }

  func.func @transformextui (%arg0: tensor<4x4xi32>) -> tensor<4x4xi64> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.extui %arg0 : tensor<4x4xi32> to tensor<4x4xi64>
// CHECK: %2 = arith.extui %arg1 : i32 to i64
// CHECK: linalg.yield
    return %0 : tensor<4x4xi64>
  }

  func.func @transformfptosi (%arg0: tensor<4x4xf32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.fptosi %arg0 : tensor<4x4xf32> to tensor<4x4xi32>
// CHECK: %2 = arith.fptosi %arg1 : f32 to i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformfptoui (%arg0: tensor<4x4xf32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.fptoui %arg0 : tensor<4x4xf32> to tensor<4x4xi32>
// CHECK: %2 = arith.fptoui %arg1 : f32 to i32
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

  func.func @transformminf (%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.minf %arg0, %arg1 : tensor<4x4xf32>
// CHECK: %2 = arith.minf %arg2, %arg3 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformminsi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.minsi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.minsi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformminui (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.minui %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.minui %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformulf (%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.mulf %arg0, %arg1 : tensor<4x4xf32>
// CHECK: %2 = arith.mulf %arg2, %arg3 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformuli (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.muli %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.muli %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformnegf (%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.negf %arg0 : tensor<4x4xf32>
// CHECK: %2 = arith.negf %arg1 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformori (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.ori %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.ori %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformremf (%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.remf %arg0, %arg1 : tensor<4x4xf32>
// CHECK: %2 = arith.remf %arg2, %arg3 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformremsi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.remsi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.remsi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformremui (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.remui %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.remui %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformsitofp (%arg0: tensor<4x4xi32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.sitofp %arg0 : tensor<4x4xi32> to tensor<4x4xf32>
// CHECK: %2 = arith.sitofp %arg1 : i32 to f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformshli (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.shli %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.shli %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformshrsi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.shrsi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.shrsi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformshrui (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.shrui %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.shrui %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformsubf (%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.subf %arg0, %arg1 : tensor<4x4xf32>
// CHECK: %2 = arith.subf %arg2, %arg3 : f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformsubi (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.subi %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.subi %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

  func.func @transformtruncf (%arg0: tensor<4x4xf64>) -> tensor<4x4xf32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.truncf %arg0 : tensor<4x4xf64> to tensor<4x4xf32>
// CHECK: %2 = arith.truncf %arg1 : f64 to f32
// CHECK: linalg.yield
    return %0 : tensor<4x4xf32>
  }

  func.func @transformxori (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
    %0 = arith.xori %arg0, %arg1 : tensor<4x4xi32>
// CHECK: %2 = arith.xori %arg2, %arg3 : i32
// CHECK: linalg.yield
    return %0 : tensor<4x4xi32>
  }

}

