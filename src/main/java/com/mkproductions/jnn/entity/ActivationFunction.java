package com.mkproductions.jnn.entity;

public enum ActivationFunction {
    SIGMOID(
            "sigmoid",
            (a, b, x) -> 1.0 / (1 + Math.exp(-x)),
            (a, b, y) -> y * (1 - y)
    ), // Sigmoid activation function with derivative.
    RE_LU(
            "re_lu",
            (a, b, x) -> Math.max(0, x),
            (a, b, y) -> (y < 0) ? 0 : 1
    ), // Rectified Linear Unit activation function with derivative.
    TAN_H(
            "tan_h",
            (a, b, x) -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)),
            (a, b, y) -> 1 - (y * y)
    ), // Hyper tangent activation function with derivative.
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
