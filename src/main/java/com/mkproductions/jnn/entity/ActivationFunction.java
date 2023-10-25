package com.mkproductions.jnn.entity;

public enum ActivationFunction {
    SIGMOID("sigmoid"),
    TAN_H("tanh"),
    RE_LU("reLu");
    final String functionName;

    ActivationFunction(String functionName) {
        this.functionName = functionName;
    }
}
