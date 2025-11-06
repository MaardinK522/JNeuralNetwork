package com.mkproductions.jnn.activationFunctions;

import com.mkproductions.jnn.cpu.entity.TensorMapAbleFunction;

public enum ActivationFunction {
    NONE("none", ((_, _) -> 0.0F), (_, _) -> 0.0F), // NONE
    SIGMOID("sigmoid", ((_, value) -> 1.0 / (1 + Math.exp(-value))), (_, y) -> y * (1 - y)), // Sigmoid activation function with derivative.
    RE_LU("re_lu", (_, x) -> Math.max(0, x), (_, y) -> (y < 0) ? 0 : 1), // Rectified Linear Unit activation function with derivative.
    LINEAR("linear", (_, x) -> x, (_, _) -> 1), // Linear activation function with derivative.
    TAN_H("tan_h", (_, x) -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)), (_, y) -> 1 - (y * y)), // Hyper tangent activation function with derivative.
    SOFTMAX("softmax", ((_, _) -> 0.0F), (_, _) -> 0.0F), // Soft max activation without function or derivative.
    ;
    final String activationFunctionName;
    final private TensorMapAbleFunction equation;
    final private TensorMapAbleFunction derivative;

    ActivationFunction(String activationFunctionName, TensorMapAbleFunction equation, TensorMapAbleFunction derivative) {
        this.activationFunctionName = activationFunctionName;
        this.equation = equation;
        this.derivative = derivative;
    }

    public TensorMapAbleFunction getEquation() {
        return equation;
    }

    public TensorMapAbleFunction getDerivative() {
        return derivative;
    }
}