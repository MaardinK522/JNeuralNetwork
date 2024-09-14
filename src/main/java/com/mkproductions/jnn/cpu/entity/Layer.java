package com.mkproductions.jnn.cpu.entity;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;

public record Layer(int numberOfNodes, ActivationFunction activationFunction) {
    @Override
    public String toString() {
        return STR."Layer{numberOfNodes=\{numberOfNodes}, activationFunction=\{activationFunction}\{'}'}";
    }
}
