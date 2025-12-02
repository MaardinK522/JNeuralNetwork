package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;

public class FlattenLayer extends Layer {
    public FlattenLayer() {
        super("Flatten", ActivationFunction.NONE);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.reshape(input.getData().length, 1);
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
        Tensor propagatedError = gradients.reshape(input.getShape());
        return new Tensor[]{null, null, propagatedError};
    }

    @Override
    public String toString() {
        return "FlattenLayer{}";
    }
}