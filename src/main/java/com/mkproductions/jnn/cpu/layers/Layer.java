package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;

public abstract class Layer {
    private final ActivationFunction activation;
    private final String name;

    public Layer(String name, ActivationFunction activation) {
        this.name = name;
        this.activation = activation;
    }

    public String getName() {
        return this.name;
    }

    public ActivationFunction getActivationFunction() {
        return activation;
    }
}