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

    public Tensor(double[] data, int... shape) {
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

    public Tensor copyShape() {
        return new Tensor(this.shape);
    }

    public static Tensor copy(Tensor tensor) {
        return new Tensor(tensor.data, tensor.shape);
    }

    public static Tensor copyShape(Tensor tensor) {
        return new Tensor(tensor.shape);
    }

    public int getRank() {
        return rank;
    }

    public int[] getShape() {
        return shape;
    }

    public double[] getData() {
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
            this.data[i] = function.map(i, this.data[i]);
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
            result.data[i] = function.map(i, tensor.data[i]);
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

    public static Tensor convolve(Tensor input, Tensor[] filters, int stride, int padding) {
        if (input.rank != 3) {
            throw new IllegalArgumentException("Input tensors must have 3 ranks.(C, H, W)");
        }
        if (filters.length == 0 || filters[0].rank != 3) {
            throw new IllegalArgumentException("Filters tensors must have 3 ranks.(C, H, W)");
        }
        int C_in = input.shape[0];
        //        int H_in = input.shape[1];
        //        int W_in = input.shape[2];

        int F_H = filters[0].shape[0];
        int F_W = filters[0].shape[1];

        if (filters[0].shape[2] != C_in) {
            throw new IllegalArgumentException("Filter depth (C_in) must match input depth.");
        }

        Tensor paddedInput = addPadding(input, padding);

        int H_in_padded = paddedInput.shape[1];
        int W_in_padded = paddedInput.shape[2];

        int C_out = filters.length;
        int H_out = (H_in_padded - F_H) / stride + 1;
        int W_out = (W_in_padded - F_W) / stride + 1;

        if (H_out <= 0 || W_out <= 0) {
            throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
        }

        Tensor output = new Tensor(C_out, H_out, W_out);

        for (int c_out = 0; c_out < C_out; c_out++) {
            Tensor filter = filters[c_out];

            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    double sum = 0.0;

                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    for (int c_in = 0; c_in < C_in; c_in++) {
                        for (int f_h = 0; f_h < F_H; f_h++) {
                            for (int f_w = 0; f_w < F_W; f_w++) {

                                double inputEntry = paddedInput.getEntry(c_in, h_start + f_h, w_start + f_w);

                                double filterEntry = filter.getEntry(f_h, f_w, c_in);

                                sum += inputEntry * filterEntry;
                            }
                        }
                    }
                    output.setEntry(sum, c_out, h_out, w_out);
                }
            }
        }
        return output;
    }

    private static Tensor addPadding(Tensor input, int padding) {
        if (padding == 0) {
            return input;
        }

        int C = input.shape[0];
        int H = input.shape[1];
        int W = input.shape[2];

        int H_padded = H + 2 * padding;
        int W_padded = W + 2 * padding;

        Tensor padded = new Tensor(C, H_padded, W_padded);

        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    padded.setEntry(input.getEntry(c, h, w), c, h + padding, w + padding);
                }
            }
        }
        return padded;
    }

    public static Tensor maxPool(Tensor input, int poolSize, int stride) {
        if (input.rank != 3) {
            throw new IllegalArgumentException("Input for pooling must be a Rank-3 Tensor (C, H, W).");
        }

        int C = input.shape[0];
        int H_in = input.shape[1];
        int W_in = input.shape[2];

        int H_out = (H_in - poolSize) / stride + 1;
        int W_out = (W_in - poolSize) / stride + 1;

        if (H_out <= 0 || W_out <= 0) {
            throw new IllegalArgumentException("Pooling resulted in zero or negative dimensions.");
        }

        Tensor output = new Tensor(C, H_out, W_out);

        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {

                    double max = Double.NEGATIVE_INFINITY;

                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    for (int p_h = 0; p_h < poolSize; p_h++) {
                        for (int p_w = 0; p_w < poolSize; p_w++) {
                            double entry = input.getEntry(c, h_start + p_h, w_start + p_w);
                            max = Math.max(max, entry);
                        }
                    }
                    output.setEntry(max, c, h_out, w_out);
                }
            }
        }
        return output;
    }

    public static Tensor transpose(Tensor tensor) {
        Tensor result = new Tensor(tensor.shape[1], tensor.shape[0]);
        for (int row = 0; row < tensor.shape[0]; row++) {
            for (int column = 0; column < tensor.shape[1]; column++) {
                result.data[column * result.shape[1] + row] = tensor.data[row * tensor.shape[1] + column];
            }
        }
        return result;
    }

    public static Tensor clip(Tensor predictions, double start, double end) {
        if (start >= end) {
            throw new IllegalArgumentException("Start must be less than end.");
        }
        return Tensor.tensorMapping(predictions, ((flatIndex, value) -> Math.max(Math.min(value, end), start)));
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