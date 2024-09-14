package com.mkproductions.jnn.activationFunctions;

import com.mkproductions.jnn.cpu.entity.MapAble;

public enum ActivationFunction {
    SIGMOID("sigmoid", (_, _, x) -> 1.0 / (1 + Math.exp(-x)), (_, _, y) -> y * (1 - y)), // Sigmoid activation function with derivative.
    RE_LU("re_lu", (_, _, x) -> Math.max(0, x), (_, _, y) -> (y < 0) ? 0 : 1), // Rectified Linear Unit activation function with derivative.
    LINEAR("linear", (_, _, x) -> x, (_, _, _) -> 1), // Linear activation function with derivative.
    TAN_H("tan_h", (_, _, x) -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)), (_, _, y) -> 1 - (y * y)), // Hyper tangent activation function with derivative.
    SOFTMAX("softmax", ((_, _, _) -> 0.0F), (_, _, _) -> 0.0F), // Soft max activation without function or derivative.
    ;
    final String activationFunctionName;
    final public MapAble equation;
    final public MapAble derivative;

    ActivationFunction(String activationFunctionName, MapAble equation, MapAble derivative) {
        this.activationFunctionName = activationFunctionName;
        this.equation = equation;
        this.derivative = derivative;
    }
}
