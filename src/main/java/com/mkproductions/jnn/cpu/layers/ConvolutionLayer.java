package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import org.jetbrains.annotations.NotNull;

public class ConvolutionLayer extends Layer {
    private final int filterSize;
    private final int numberOfFilters;
    private final int stride;
    private final int padding;

    public ConvolutionLayer(int filterSize, int numberOfFilters, int stride, int padding, ActivationFunction activation) {
        super("Convolution", activation);
        this.filterSize = filterSize;
        this.numberOfFilters = numberOfFilters;
        this.stride = stride;
        this.padding = padding;
    }

    public int getFilterSize() {
        return filterSize;
    }

    public int getNumberOfFilters() {
        return numberOfFilters;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    @Override
    public @NotNull String toString() {
        return STR."ConvolutionLayer{filterSize=\{filterSize}, activationFunction=\{getActivationFunction()}}";
    }
}