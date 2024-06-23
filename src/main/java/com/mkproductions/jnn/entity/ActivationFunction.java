package com.mkproductions.jnn.entity;

public enum ActivationFunction {
    SIGMOID("sigmoid", // Name
            (a, b, x) -> 1.0 / (1 + Math.exp(-x)), // Function
            (a, b, y) -> y * (1 - y) // Derivation
    ), // Sigmoid activation function with derivative.
    RE_LU("re_lu", // Name
            (a, b, x) -> Math.max(0, x), // Function
            (a, b, y) -> (y < 0) ? 0 : 1 // Derivation
    ), // Rectified Linear Unit activation function with derivative.
    TAN_H("tan_h", // Name
            (a, b, x) -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)), // Function
            (a, b, y) -> 1 - (y * y) // Derivation
    ) // Hyper tangent activation function with derivative.
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
