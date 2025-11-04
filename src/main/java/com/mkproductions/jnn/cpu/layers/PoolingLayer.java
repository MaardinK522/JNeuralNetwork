package com.mkproductions.jnn.cpu.layers;

public record PoolingLayer(int poolSize, int stride) implements Layer {
    @Override
    public String toString() {
        return "PoolingLayer{" + "poolSize=" + poolSize + ", stride=" + stride + '}';
    }

    @Override
    public void printLayerInfo() {
        System.out.println(this);
    }
}