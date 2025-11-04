package com.mkproductions.jnn.activationFunctions;

import com.mkproductions.jnn.cpu.entity.Matrix3DFunctionAble;

public enum ActivationFunction3D {
    SIGMOID("sigmoid", (_, _, _, x) -> 1.0 / (1 + Math.exp(-x)), (_, _, _, y) -> y * (1 - y)), // Sigmoid activation function with derivative.
    RE_LU("re_lu", (_, _, _, x) -> Math.max(0, x), (_, _, _, y) -> (y < 0) ? 0 : 1), // Rectified Linear Unit activation function with derivative.
    LINEAR("linear", (_, _, _, x) -> x, (_, _, _, _) -> 1), // Linear activation function with derivative.
    TAN_H("tan_h", (_, _, _, x) -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)), (_, _, _, y) -> 1 - (y * y)), // Hyper tangent activation function with derivative.
    SOFTMAX("softmax", ((_, _, _, _) -> 0.0F), (_, _, _, _) -> 0.0F), // Soft max activation without function or derivative.
    ;
    final String activationFunctionName;
    final public Matrix3DFunctionAble equation;
    final public Matrix3DFunctionAble derivative;

    ActivationFunction3D(String activationFunctionName, Matrix3DFunctionAble equation, Matrix3DFunctionAble derivative) {
        this.activationFunctionName = activationFunctionName;
        this.equation = equation;
        this.derivative = derivative;
    }
}