package org.mkproductions.jnn.entity;

public class Layer {
    private final int activationFunction;
    private final int numberOfNodes;

    public Layer(int activationFunction, int numberOfNodes) {
        this.activationFunction = activationFunction;
        this.numberOfNodes = numberOfNodes;
    }

    public int getActivationFunction() {
        return activationFunction;
    }

    public int getNumberOfNodes() {
        return numberOfNodes;
    }
}
