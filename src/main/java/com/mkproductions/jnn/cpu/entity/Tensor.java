package com.mkproductions.jnn.cpu.entity;

import java.util.ArrayList;

public class Tensor {
    private final int rank;
    private final int[] shape;
    private final int[] strides;
    private final double[] data;

    public Tensor(int[] shape) {
        if (shape.length < 1) {
            throw new IllegalArgumentException("Shape must have at least one value.");
        }
        this.shape = shape;
        int dataLength = 1;
        for (int dimension : shape) {
            if (dimension < 1) {
                throw new IllegalArgumentException("Shape must have at least one value.");
            }
            dataLength *= dimension;
        }
        this.data = new double[dataLength];
        // Precomputed multipliers.
        strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        this.rank = shape.length;
    }

    public int getRank() {
        return rank;
    }

    public int[] getShape() {
        return shape;
    }

    public double[] toFlat() {
        return data;
    }

    public static int index(Tensor tensor, int... indices) {
        if (indices.length != tensor.rank) {
            throw new IllegalArgumentException("Index length must be equal to rank.");
        }
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= tensor.shape[i]) {
                throw new IndexOutOfBoundsException(STR."Index out of bounds. Index: \{indices[i]} Shape: \{tensor.shape[i]}");
            }
            index += indices[i] * tensor.strides[i];
        }
        return index;
    }

    public void setEntry(double value, int... indices) {
        this.data[Tensor.index(this, indices)] = value;
    }

    public double getEntry(int... indices) {
        return this.data[Tensor.index(this, indices)];
    }

    public void add(Tensor tensor) {
        if (this.rank != tensor.rank) {
            throw new IllegalArgumentException("Mismatch rank for performing addition operation.");
        }
        for (int dimension = 0; dimension < this.shape.length; dimension++) {
            if (this.shape[dimension] != tensor.shape[dimension]) {
                System.err.println(STR."Mistmatch shape for dimension \{this.shape[dimension]} + \{tensor.shape[dimension]}");
                throw new IllegalArgumentException("Mismatch shape for performing addition operation.");
            }
        }
    }
}