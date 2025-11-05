package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;

public abstract class Layer {
    private ActivationFunction activation;

    public Layer(ActivationFunction activation) {
        this.activation = activation;
    }

    public ActivationFunction getActivationFunction() {
        return activation;
    }
}