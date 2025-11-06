package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;

public class FlattenLayer extends Layer {
    public FlattenLayer() {
        super("Flatten", ActivationFunction.NONE);
    }
}