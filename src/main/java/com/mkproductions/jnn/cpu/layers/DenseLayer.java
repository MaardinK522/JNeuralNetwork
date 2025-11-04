package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;

public record DenseLayer(int numberOfNodes, ActivationFunction activationFunction) implements Layer {
    @Override
    public String toString() {
        return "Layer{numberOfNodes=" + numberOfNodes + ", activationFunction=" + activationFunction + "}";
    }

    @Override
    public void printLayerInfo() {
        System.out.println(this);
    }
}