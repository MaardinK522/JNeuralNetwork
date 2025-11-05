package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;

public class PoolingLayer extends Layer {
    private int poolSize;
    private int stride;

    public PoolingLayer(ActivationFunction activation) {
        super(activation);
    }

    @Override
    public String toString() {
        return STR."PoolingLayer{poolSize=\{poolSize}, stride=\{stride}}";
    }

    public int getPoolSize() {
        return poolSize;
    }

    public int getStride() {
        return stride;
    }
}