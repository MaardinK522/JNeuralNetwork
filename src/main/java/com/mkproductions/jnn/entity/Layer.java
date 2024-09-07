package com.mkproductions.jnn.entity;

import com.mkproductions.jnn.entity.activationFunctions.ActivationFunction;

public record Layer(int numberOfNodes, ActivationFunction activationFunction) {
    @Override
    public String toString() {
        return STR."Layer{numberOfNodes=\{numberOfNodes}, activationFunction=\{activationFunction}\{'}'}";
    }
}
