package com.mkproductions.jnn.entity;

public enum ActivationFunction {
    SIGMOID("sigmoid", (a, b, x) -> 1.0 / (1 + Math.exp(-x)), (a, b, y) -> y * (1 - y)),
    RE_LU("re_lu", (a, b, x) -> Math.max(0, x), (a, b, y) -> (y < 0) ? 0 : 1),
    TAN_H("tan_h", (a, b, x) -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)), (a, b, y) -> 1 - (y * y));
    final String activationFunctionName;
    final public MapAble equation;
    final public MapAble derivative;

    ActivationFunction(String activationFunctionName, MapAble equation, MapAble derivative) {
        this.activationFunctionName = activationFunctionName;
        this.equation = equation;
        this.derivative = derivative;
    }
}
