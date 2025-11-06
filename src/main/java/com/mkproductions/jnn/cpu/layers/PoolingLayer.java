package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;

public class PoolingLayer extends Layer {
    private final int poolSize;
    private final int stride;

    private final PoolingLayerType poolingLayerType;

    public PoolingLayer(int poolSize, int stride, PoolingLayerType poolingLayerType) {
        super(poolingLayerType == PoolingLayerType.AVG ? "Average Pooling" : "Max Pooling", ActivationFunction.NONE);
        this.poolSize = poolSize;
        this.stride = stride;
        this.poolingLayerType = poolingLayerType;
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

    public PoolingLayerType getPoolingLayerType() {
        return poolingLayerType;
    }

    public enum PoolingLayerType {
        MAX, AVG
    }
}