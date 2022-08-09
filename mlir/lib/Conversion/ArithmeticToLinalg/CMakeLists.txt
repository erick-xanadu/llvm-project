add_mlir_conversion_library(MLIRArithmeticToLinalg
  ArithmeticToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithmeticToLinalg

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRControlFlowDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  )
