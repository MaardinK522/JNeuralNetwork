package com.mkproductions.jnn.cpu.entity;

import java.util.Arrays;
import java.util.Objects;

public class Tensor {
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
    }

    public Tensor(double[] data, int... shape) {
        this(shape);
        this.data = Arrays.copyOf(data, data.length);
    }

    public static void validateTensors(Tensor tensor1, Tensor tensor2) {
        if (tensor1.getRank() != tensor2.getRank()) {
            System.err.println(STR."Ranks: (\{tensor1.getRank()}) != (\{tensor2.getRank()})");
            throw new IllegalArgumentException("Index length must be equal to rank.");
        }
        for (int i = 0; i < tensor1.shape.length; i++) {
            if (tensor1.shape[i] < 1 || tensor2.shape[i] < 1 || tensor1.shape[i] != tensor2.shape[i]) {
                System.err.println(tensor1);
                System.err.println(tensor2);
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
        return this.shape.length;
    }

    public int[] getShape() {
        return shape;
    }

    public double[] getData() {
        return data;
    }

    public static int getIndex(Tensor tensor, int[] shape) {
        if (shape.length != tensor.getRank()) {
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
        result.mapTensor(function);
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
        if (tensor1.getRank() != tensor2.getRank() || tensor1.getRank() != 2) {
            System.err.println(tensor1);
            System.err.println(tensor2);
            throw new IllegalArgumentException(STR."Matrix multiplication not possible.");
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

    // AI I.D.K. ChatGPT told me to keep it.
    public Tensor reshape(int... newShape) {
        int totalOld = Arrays.stream(this.shape).reduce(1, (a, b) -> a * b);
        int totalNew = Arrays.stream(newShape).reduce(1, (a, b) -> a * b);
        if (totalOld != totalNew) {
            System.err.println(STR."Old: \{totalOld} New: \{totalNew}");
            throw new IllegalArgumentException("Reshape cannot change total size.");
        }
        return new Tensor(Arrays.copyOf(this.data, this.data.length), newShape);
    }

    public static Tensor convolve2D(Tensor input, Tensor kernel, int stride, int padding) {
        if (input.getRank() != 2 || kernel.getRank() != 2) {
            throw new IllegalArgumentException("Both must be 2D tensors.");
        }
        int inputHeight = input.shape[0];
        int inputWidth = input.shape[1];
        int kernelHeight = kernel.shape[0];
        int kernelWidth = kernel.shape[1];
        int paddedHeight = inputHeight + 2 * padding;
        int paddedWidth = inputWidth + 2 * padding;

        Tensor padded = new Tensor(paddedHeight, paddedWidth);

        for (int a = 0; a < inputHeight; a++) {
            for (int b = 0; b < inputWidth; b++) {
                padded.setEntry(input.getEntry(a, b), a + padding, b + padding);
            }
        }

        int outputHeight = (paddedHeight - kernelHeight) / stride + 1;
        int outputWidth = (paddedWidth - kernelWidth) / stride + 1;

        Tensor output = new Tensor(outputHeight, outputWidth);
        for (int outputRow = 0; outputRow < outputHeight; outputRow++) {
            for (int outputColumn = 0; outputColumn < outputWidth; outputColumn++) {
                double sum = 0.0;
                for (int kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
                    for (int kernelColumn = 0; kernelColumn < kernelWidth; kernelColumn++) {
                        sum += padded.getEntry(outputRow * stride + kernelRow, outputColumn * stride + kernelColumn) * kernel.getEntry(kernelRow, kernelColumn);
                    }
                }
                output.setEntry(sum, outputRow, outputColumn);
            }
        }
        return output;
    }

    public static Tensor convolve2D(Tensor input, Tensor kernel, int stride, int padding, Tensor result) {
        if (input.getRank() != 2 || kernel.getRank() != 2) {
            throw new IllegalArgumentException("Both input and kernel must be 2D tensors.");
        }

        int inputHeight = input.getShape()[0];
        int inputWidth = input.getShape()[1];
        int kernelHeight = kernel.getShape()[0];
        int kernelWidth = kernel.getShape()[1];
        int paddedHeight = inputHeight + 2 * padding;
        int paddedWidth = inputWidth + 2 * padding;

        Tensor padded = new Tensor(paddedHeight, paddedWidth);

        for (int a = 0; a < inputHeight; a++) {
            for (int b = 0; b < inputWidth; b++) {
                padded.setEntry(input.getEntry(a, b), a + padding, b + padding);
            }
        }
        int H_padded = padded.getShape()[0];
        int W_padded = padded.getShape()[1];

        int H_out = (H_padded - kernelHeight) / stride + 1;
        int W_out = (W_padded - kernelWidth) / stride + 1;

        // Dimension validation for the result tensor
        if (result == null) {
            result = new Tensor(H_out, W_out);
        } else {
            int[] rShape = result.getShape();
            if (rShape.length != 2 || rShape[0] != H_out || rShape[1] != W_out) {
                throw new IllegalArgumentException(String.format("Result tensor shape mismatch. Expected [%d, %d] but got %s.", H_out, W_out, Arrays.toString(rShape)));
            }
        }

        // Perform convolution
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                double sum = 0.0;

                int h_start = h * stride;
                int w_start = w * stride;

                for (int i = 0; i < kernelHeight; i++) {
                    for (int j = 0; j < kernelWidth; j++) {
                        double inputVal = padded.getEntry(h_start + i, w_start + j);
                        double kernelVal = kernel.getEntry(i, j);
                        sum += inputVal * kernelVal;
                    }
                }

                result.setEntry(sum, h, w);
            }
        }

        return result;
    }

    public Tensor slice(int axis, int start, int end) {
        if (axis < 0 || axis >= this.getRank()) {
            throw new IllegalArgumentException(STR."Axis out of range: \{axis}");
        }
        if (start < 0 || end > this.shape[axis] || start >= end) {
            throw new IndexOutOfBoundsException(STR."Invalid slice range: \{start} to \{end}");
        }

        // Compute the new shape after slicing
        int[] newShape = Arrays.copyOf(this.shape, this.shape.length);
        newShape[axis] = end - start;

        // Compute stride product after the axis (how many elements per step along that axis)
        int sliceSize = 1;
        for (int i = axis + 1; i < this.shape.length; i++) {
            sliceSize *= this.shape[i];
        }

        // Compute stride product before the axis
        int beforeAxis = 1;
        for (int i = 0; i < axis; i++) {
            beforeAxis *= this.shape[i];
        }

        // Prepare result data buffer
        double[] newData = new double[Arrays.stream(newShape).reduce(1, (a, b) -> a * b)];
        int newIndex = 0;

        // Copy all slices along the selected range
        for (int b = 0; b < beforeAxis; b++) {
            int baseIndex = b * this.shape[axis] * sliceSize;
            for (int i = start; i < end; i++) {
                int from = baseIndex + i * sliceSize;
                System.arraycopy(this.data, from, newData, newIndex, sliceSize);
                newIndex += sliceSize;
            }
        }

        return new Tensor(newData, newShape);
    }

    @Override
    public String toString() {
        return STR."Tensor{rank=\{this.getRank()}, shape=\{Arrays.toString(shape)}, strides=\{Arrays.toString(strides)}, data=\{Arrays.toString(data)}}";
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Tensor tensor = (Tensor) o;
        return this.getRank() == tensor.getRank() && Objects.deepEquals(shape, tensor.shape) && Objects.deepEquals(strides, tensor.strides);
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.getRank(), Arrays.hashCode(shape), Arrays.hashCode(strides), Arrays.hashCode(data));
    }

}