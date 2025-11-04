package com.mkproductions.jnn.cpu.entity;

import java.util.Arrays;
import java.util.Objects;

public class Tensor {
    private final int rank;
    private final int[] shape;
    private final int[] strides;
    private double[] data;

    public Tensor(int... shape) {
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

    public Tensor(double[] data, int[] shape) {
        this(shape);
        this.data = Arrays.copyOf(data, data.length);
    }

    public static void validateTensors(Tensor tensor1, Tensor tensor2) {
        if (tensor1.rank != tensor2.rank) {
            throw new IllegalArgumentException("Index length must be equal to rank.");
        }
        for (int i = 0; i < tensor1.shape.length; i++) {
            if (tensor1.shape[i] < 1 || tensor2.shape[i] < 1 || tensor1.shape[i] != tensor2.shape[i]) {
                throw new IndexOutOfBoundsException(STR."Index out of bounds. Index: \{tensor1.shape[i]} Shape: \{tensor2.shape[i]}");
            }
        }
    }

    public Tensor copy() {
        return new Tensor(this.data, this.shape);
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

    public static int getIndex(Tensor tensor, int[] shape) {
        if (shape.length != tensor.rank) {
            throw new IllegalArgumentException("Index length must be equal to rank.");
        }
        int index = 0;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0 || shape[i] >= tensor.shape[i]) {
                throw new IndexOutOfBoundsException(STR."Index out of bounds. Index: \{shape[i]} Shape: \{tensor.shape[i]}");
            }
            index += shape[i] * tensor.strides[i];
        }
        return index;
    }

    public void setEntry(double value, int... indices) {
        this.data[Tensor.getIndex(this, indices)] = value;
    }

    public double getEntry(int... indices) {
        return this.data[Tensor.getIndex(this, indices)];
    }

    public void add(Tensor tensor) {
        validateTensors(this, tensor);
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] += tensor.data[i];
        }
    }

    public static Tensor add(Tensor tensor1, Tensor tensor2) {
        validateTensors(tensor1, tensor2);
        Tensor result = tensor1.copy();
        result.add(tensor2);
        return result;
    }

    public void subtract(Tensor tensor) {
        validateTensors(this, tensor);
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] -= tensor.data[i];
        }
    }

    public static Tensor subtract(Tensor tensor1, Tensor tensor2) {
        validateTensors(tensor1, tensor2);
        Tensor result = tensor1.copy();
        result.subtract(tensor2);
        return result;
    }

    public void scalarMultiply(double scale) {
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] *= scale;
        }
    }

    public static Tensor scalarMultiply(Tensor tensor, double scale) {
        Tensor result = tensor.copy();
        result.scalarMultiply(scale);
        return result;
    }

    public void mapTensor(TensorMapAbleFunction function) {
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] = function.map(this.data[i]);
        }
    }

    public static Tensor tensorMapping(Tensor tensor, TensorMapAbleFunction function) {
        Tensor result = tensor.copy();
        int[] indices = new int[result.rank];
        for (int i = 0; i < result.data.length; i++) {
            int remaining = i;
            for (int dimension = 0; dimension < tensor.rank; dimension++) {
                indices[dimension] = remaining / tensor.strides[dimension];
                remaining -= indices[dimension] * tensor.strides[dimension];
            }
            result.data[i] = function.map(tensor.data[i]);
        }
        return result;
    }

    public double dotProduct(Tensor tensor) {
        validateTensors(this, tensor);
        double resultSummation = 0;
        for (int i = 0; i < this.data.length; i++) {
            resultSummation += this.data[i] * tensor.data[i];
        }
        return resultSummation;
    }

    public static double dotProduct(Tensor tensor1, Tensor tensor2) {
        validateTensors(tensor1, tensor2);
        return tensor1.dotProduct(tensor2);
    }

    public static Tensor matrixMultiplication(Tensor tensor1, Tensor tensor2) {
        if (tensor1.rank != tensor2.rank || tensor1.rank != 2) {
            throw new IllegalArgumentException(STR."Matrix multiplication not possible. Cause: \{tensor1} != \{tensor2}");
        }
        if (tensor1.shape[1] != tensor2.shape[0]) {
            System.err.println(tensor1);
            System.err.println(tensor2);
            throw new IllegalArgumentException("Mismatch dimension for matrix multiplication.");
        }
        Tensor result = new Tensor(tensor1.shape[0], tensor2.shape[1]);
        for (int row = 0; row < tensor1.shape[0]; row++) {
            for (int column = 0; column < tensor2.shape[1]; column++) {
                double sum = 0;
                for (int i = 0; i < tensor1.shape[1]; i++) {
                    sum += tensor1.data[row * tensor1.shape[1] + i] * tensor2.data[i * tensor2.shape[1] + column];
                }
                result.data[row * result.shape[1] + column] = sum;
            }
        }
        return result;
    }

    public void elementWiseMultiplication(Tensor tensor) {
        validateTensors(this, tensor);
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] *= tensor.data[i];
        }
    }

    public static Tensor elementWiseMultiplication(Tensor tensor1, Tensor tensor2) {
        validateTensors(tensor1, tensor2);
        Tensor result = tensor1.copy();
        result.elementWiseMultiplication(tensor2);
        return result;
    }

    @Override
    public String toString() {
        return STR."Tensor{rank=\{rank}, shape=\{Arrays.toString(shape)}, strides=\{Arrays.toString(strides)}, data=\{Arrays.toString(data)}}";
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Tensor tensor = (Tensor) o;
        return rank == tensor.rank && Objects.deepEquals(shape, tensor.shape) && Objects.deepEquals(strides, tensor.strides) && Objects.deepEquals(data, tensor.data);
    }

    @Override
    public int hashCode() {
        return Objects.hash(rank, Arrays.hashCode(shape), Arrays.hashCode(strides), Arrays.hashCode(data));
    }

    // AI I.D.K. ChatGPT told me to keep it.
    public Tensor reshape(int... newShape) {
        int totalOld = Arrays.stream(this.shape).reduce(1, (a, b) -> a * b);
        int totalNew = Arrays.stream(newShape).reduce(1, (a, b) -> a * b);
        if (totalOld != totalNew) {
            throw new IllegalArgumentException("Reshape cannot change total size.");
        }
        return new Tensor(Arrays.copyOf(this.data, this.data.length), newShape);
    }
}