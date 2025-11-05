package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;

public class DenseLayer extends Layer {
    private final int numberOfNodes;

    public DenseLayer(int numberOfNodes, ActivationFunction activation) {
        super(activation);
        this.numberOfNodes = numberOfNodes;
    }

    @Override
    public String toString() {
        return STR."Layer{numberOfNodes=\{numberOfNodes}, activationFunction=\{getActivationFunction()}}";
    }

    public int getNumberOfNodes() {
        return this.numberOfNodes;
    }
}