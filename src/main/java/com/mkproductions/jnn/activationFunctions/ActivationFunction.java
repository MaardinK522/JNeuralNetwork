package com.mkproductions.jnn.activationFunctions;

import com.mkproductions.jnn.cpu.entity.TensorMapAbleFunction;

public enum ActivationFunction {
    SIGMOID("sigmoid", ((flatIndex, value) -> 1.0 / (1 + Math.exp(-value))), (flatIndex, y) -> y * (1 - y)), // Sigmoid activation function with derivative.
    RE_LU("re_lu", (flatIndex, x) -> Math.max(0, x), (flatIndex, y) -> (y < 0) ? 0 : 1), // Rectified Linear Unit activation function with derivative.
    LINEAR("linear", (flatIndex, x) -> x, (flatIndex, _) -> 1), // Linear activation function with derivative.
    TAN_H("tan_h", (flatIndex, x) -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)), (flatIndex, y) -> 1 - (y * y)), // Hyper tangent activation function with derivative.
    SOFTMAX("softmax", ((flatIndex, _) -> 0.0F), (flatIndex, _) -> 0.0F), // Soft max activation without function or derivative.
    ;
    final String activationFunctionName;
    final private TensorMapAbleFunction equation;
    final private TensorMapAbleFunction derivative;

    ActivationFunction(String activationFunctionName, TensorMapAbleFunction equation, TensorMapAbleFunction derivative) {
        this.activationFunctionName = activationFunctionName;
        this.equation = equation;
        this.derivative = derivative;
    }

    public String getActivationFunctionName() {
        return activationFunctionName;
    }

    public TensorMapAbleFunction getEquation() {
        return equation;
    }

    public TensorMapAbleFunction getDerivative() {
        return derivative;
    }
}