package com.mkproductions.jnn.entity;

public class Layer {
    private final ActivationFunction activationFunction;
    private final int numberOfNodes;

    public Layer(int numberOfNodes, ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        this.numberOfNodes = numberOfNodes;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public int getNumberOfNodes() {
        return numberOfNodes;
    }
}
