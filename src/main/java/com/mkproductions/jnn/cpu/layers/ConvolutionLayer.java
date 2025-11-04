package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction3D;
import org.jetbrains.annotations.NotNull;

public record ConvolutionLayer(int filterSize, int numberOfFilters, int stride, int padding, ActivationFunction3D activationFunction) implements Layer {

    @Override
    public @NotNull String toString() {
        return "ConvolutionLayer{filterSize=" + filterSize + ", activationFunction=" + activationFunction + "}";
    }

    @Override
    public void printLayerInfo() {
        System.out.println(this);
    }
}