package com.mkproductions.jnn.cpu.entity;

import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.Arrays;
import java.util.stream.IntStream;

public class Tensor {
    private IntArray shape;
    private IntArray strides;
    private DoubleArray data;
    private static final int PARALLEL_THRESHOLD = 10000;

    public Tensor(int... shape) {
        if (shape.length < 1) {
            throw new IllegalArgumentException("Shape must have at least one value.");
        }
        this.shape = IntArray.fromArray(shape);
        int dataLength = 1;
        for (int dimension : shape) {
            if (dimension < 1) {
                throw new IllegalArgumentException("Shape must have at least one value.");
            }
            dataLength *= dimension;
        }
        this.data = new DoubleArray(dataLength);
        // Precomputed multipliers.
        strides = new IntArray(shape.length);
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides.set(i, stride);
            stride *= shape[i];
        }
    }

    public Tensor(DoubleArray data, int... shape) {
        this(shape);
        this.data = data;
    }

    public static void validateTensors(Tensor tensor1, Tensor tensor2) {
        if (tensor1.getRank() != tensor2.getRank()) {
            System.err.println(STR."Ranks: (\{tensor1.getRank()}) != (\{tensor2.getRank()})");
            throw new IllegalArgumentException("Index length must be equal to rank.");
        }
        for (int i = 0; i < tensor1.shape.getSize(); i++) {
            if (tensor1.shape.get(i) != tensor2.shape.get(i)) {
                System.err.println(tensor1);
                System.err.println(tensor2);
                throw new IllegalArgumentException("Mismatch dimension.");
            }
        }
    }

    public static Tensor flip2D(Tensor input) {
        if (input.getRank() != 2) {
            throw new IllegalArgumentException("Flip2D only supports Rank-2 tensors (H x W).");
        }

        int H = input.getShape().toHeapArray()[0];
        int W = input.getShape().toHeapArray()[1];
        Tensor flipped = new Tensor(H, W);

        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                // Map original position (r, c) to flipped position (H - 1 - r, W - 1 - c)
                double value = input.getEntry(r, c);
                flipped.setEntry(value, H - 1 - r, W - 1 - c);
            }
        }
        return flipped;
    }

    public void copy(DoubleArray data, IntArray shape) {
        for (int i = 0; i < shape.getSize(); i++) {
            if (shape.get(i) != this.shape.get(i)) {
                throw new IllegalArgumentException("Shape mismatch.");
            }
        }
        this.shape = shape;
        int dataLength = 1;
        for (int i = 0; i < shape.getSize(); i++) {
            int dimension = shape.get(i);
            if (dimension < 1) {
                throw new IllegalArgumentException("Shape must have at least one value.");
            }
            dataLength *= dimension;
        }
        if (this.data.getSize() != data.getSize()) {
            throw new IllegalArgumentException("Data size must match tensor size.");
        }
        for (int i = 0; i < dataLength; i++) {
            this.data.set(i, data.get(i));
        }
        // Precomputed multipliers.
        this.strides = new IntArray(shape.getSize());
        int stride = 1;
        for (int i = shape.getSize() - 1; i >= 0; i--) {
            strides.set(i, stride);
            stride *= shape.get(i);
        }
    }

    public Tensor copy() {
        return new Tensor(this.data, this.shape.toHeapArray());
    }

    public Tensor copyShape() {
        return new Tensor(this.shape.toHeapArray());
    }

    public static Tensor copy(Tensor tensor) {
        return new Tensor(tensor.data, tensor.shape.toHeapArray());
    }

    public static Tensor copyShape(Tensor tensor) {
        return new Tensor(tensor.shape.toHeapArray());
    }

    public int getRank() {
        return this.shape.getSize();
    }

    public IntArray getShape() {
        return shape;
    }

    public static int getIndex(Tensor tensor, int[] shape) {
        if (shape.length != tensor.getRank()) {
            throw new IllegalArgumentException("Index length must be equal to rank.");
        }
        int index = 0;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0 || shape[i] >= tensor.shape.get(i)) {
                throw new IndexOutOfBoundsException(STR."Index out of bounds. Index: \{shape[i]} Shape: \{tensor.shape.get(i)}");
            }
            index += shape[i] * tensor.strides.get(i);
        }
        return index;
    }

    public void setEntry(double value, int... indices) {
        this.data.set(Tensor.getIndex(this, indices), value);
    }

    public double getEntry(int... indices) {
        return this.data.get(Tensor.getIndex(this, indices));
    }

    public DoubleArray getData() {
        return data;
    }

    public void setData(DoubleArray data) {
        this.data = data;
    }

    public int[] getStrides() {
        return strides.toHeapArray();
    }

    public void add(Tensor tensor) {
        validateTensors(this, tensor);
        for (int i = 0; i < this.data.getSize(); i++) {
            this.data.set(i, this.data.get(i) + tensor.data.get(i));
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
        for (int i = 0; i < this.data.getSize(); i++) {
            this.data.set(i, this.data.get(i) + tensor.data.get(i));
        }
    }

    public static Tensor subtract(Tensor tensor1, Tensor tensor2) {
        validateTensors(tensor1, tensor2);
        Tensor result = tensor1.copy();
        result.subtract(tensor2);
        return result;
    }

    public void scalarMultiply(double scale) {
        for (int i = 0; i < this.data.getSize(); i++) {
            this.data.set(i, this.data.get(i) * scale);
        }
    }

    public static Tensor scalarMultiply(Tensor tensor, double scale) {
        Tensor result = tensor.copy();
        result.scalarMultiply(scale);
        return result;
    }

    public void mapTensor(TensorMapAbleFunction function) {
        for (int i = 0; i < this.data.getSize(); i++) {
            this.data.set(i, function.map(i, this.data.get(i)));
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
        for (int i = 0; i < this.data.getSize(); i++) {
            resultSummation += this.data.get(i) * tensor.data.get(i);
        }
        return resultSummation;
    }

    public static double dotProduct(Tensor tensor1, Tensor tensor2) {
        validateTensors(tensor1, tensor2);
        return tensor1.dotProduct(tensor2);
    }

    public static Tensor matrixMultiplication(Tensor tensor1, Tensor tensor2) {
        if (tensor1.getRank() != 2 || tensor2.getRank() != 2) {
            throw new IllegalArgumentException("Matrix multiplication requires Rank 2 tensors.");
        }
        if (tensor1.shape.toHeapArray()[1] != tensor2.shape.toHeapArray()[0]) {
            throw new IllegalArgumentException(STR."Dimension mismatch: \{tensor1.shape.toHeapArray()[1]} != \{tensor2.shape.toHeapArray()[0]}");
        }

        int rowsA = tensor1.shape.toHeapArray()[0];
        int colsA = tensor1.shape.toHeapArray()[1]; // Same as rowsB
        int colsB = tensor2.shape.toHeapArray()[1];

        Tensor result = new Tensor(rowsA, colsB);

        double[] dataA = tensor1.data.toHeapArray();
        double[] dataB = tensor2.data.toHeapArray();
        double[] dataR = result.data.toHeapArray();

        for (int i = 0; i < rowsA; i++) {
            int rowOffsetA = i * colsA;
            int rowOffsetR = i * colsB;

            for (int k = 0; k < colsA; k++) {
                double valA = dataA[rowOffsetA + k];
                int rowOffsetB = k * colsB;

                for (int j = 0; j < colsB; j++) {
                    // R.get(i)[j] += A.get(i)[k] * B[k][j]
                    dataR[rowOffsetR + j] += valA * dataB[rowOffsetB + j];
                }
            }
        }
        return result;
    }

    public void elementWiseMultiplication(Tensor tensor) {
        validateTensors(this, tensor);
        for (int i = 0; i < this.data.getSize(); i++) {
            this.data.set(i, this.data.get(i) * tensor.data.get(i));
        }
    }

    public static Tensor elementWiseMultiplication(Tensor tensor1, Tensor tensor2) {
        validateTensors(tensor1, tensor2);
        Tensor result = tensor1.copy();
        result.elementWiseMultiplication(tensor2);
        return result;
    }

    public static Tensor transpose(Tensor tensor) {
        Tensor result = new Tensor(tensor.shape.toHeapArray()[1], tensor.shape.toHeapArray()[0]);
        for (int row = 0; row < tensor.shape.toHeapArray()[0]; row++) {
            for (int column = 0; column < tensor.shape.toHeapArray()[1]; column++) {
                result.data.set(column * result.getShape().toHeapArray()[1] + row, tensor.data.get(row * tensor.shape.toHeapArray()[1] + column));
            }
        }
        return result;
    }

    public static Tensor clip(Tensor predictions, double start, double end) {
        if (start >= end) {
            throw new IllegalArgumentException("Start must be less than end.");
        }
        return Tensor.tensorMapping(predictions, ((_, value) -> Math.max(Math.min(value, end), start)));
    }

    public Tensor reshape(int... newShape) {
        int totalOld = Arrays.stream(this.shape.toHeapArray()).reduce(1, (a, b) -> a * b);
        int totalNew = Arrays.stream(newShape).reduce(1, (a, b) -> a * b);
        if (totalOld != totalNew) {
            System.err.println(STR."Old: \{totalOld} New: \{totalNew}");
            throw new IllegalArgumentException("Reshape cannot change total size.");
        }
        return new Tensor(this.data, newShape);
    }

    public static Tensor convolve2D(Tensor input, Tensor kernel, int stride, int padding) {
        if (input.getRank() != 2 || kernel.getRank() != 2) {
            throw new IllegalArgumentException("Both must be 2D tensors.");
        }
        int inputHeight = input.shape.toHeapArray()[0];
        int inputWidth = input.shape.toHeapArray()[1];
        int kernelHeight = kernel.shape.toHeapArray()[0];
        int kernelWidth = kernel.shape.toHeapArray()[1];
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

    public void setSlice(Tensor slice, int... startIndices) {
        if (startIndices.length != this.shape.getSize()) {
            throw new IllegalArgumentException(STR."Start indices count (\{startIndices.length}) must match host rank (\{this.shape.getSize()}).");
        }
        int dimOffset = this.shape.getSize() - slice.shape.getSize();
        if (dimOffset < 0) {
            throw new IllegalArgumentException("Slice rank cannot be higher than Host rank.");
        }
        for (int i = 0; i < slice.shape.getSize(); i++) {
            int hostDimIndex = i + dimOffset;
            int requestedEnd = startIndices[hostDimIndex] + slice.shape.get(i);
            if (requestedEnd > this.shape.toHeapArray()[hostDimIndex]) {
                throw new IllegalArgumentException(STR."Slice overflow at dimension \{hostDimIndex}. Start: \{startIndices[hostDimIndex]}, Slice Dim: \{slice.shape.get(i)}, Host Dim: \{this.shape.toHeapArray()[hostDimIndex]}");
            }
        }
        int hostBaseOffset = 0;
        for (int i = 0; i < dimOffset; i++) {
            hostBaseOffset += startIndices[i] * this.strides.get(i);
        }
        setSliceRecursive(slice, 0, 0, hostBaseOffset, startIndices, dimOffset);
    }

    private void setSliceRecursive(Tensor slice, int currentSliceDim, int currentSliceOffset, int currentHostOffset, int[] startIndices, int dimOffset) {
        if (currentSliceDim == slice.shape.getSize() - 1) {
            slice.data = this.data;
            return;
        }
        int sliceDimSize = slice.shape.toHeapArray()[currentSliceDim];
        int hostDimIndex = currentSliceDim + dimOffset;
        for (int i = 0; i < sliceDimSize; i++) {
            int nextSliceOffset = currentSliceOffset + (i * slice.strides.toHeapArray()[currentSliceDim]);
            int nextHostOffset = currentHostOffset + ((startIndices[hostDimIndex] + i) * this.strides.toHeapArray()[hostDimIndex]);
            setSliceRecursive(slice, currentSliceDim + 1, nextSliceOffset, nextHostOffset, startIndices, dimOffset);
        }
    }

    public Tensor getSlice(int axis, int start, int end) {
        if (axis < 0 || axis >= this.getRank()) {
            throw new IllegalArgumentException(STR."Axis out of range: \{axis}");
        }
        if (start < 0 || end > this.shape.toHeapArray()[axis] || start >= end) {
            throw new IndexOutOfBoundsException(STR."Invalid slice range: \{start} to \{end}");
        }

        // Compute the new shape after slicing
        int[] newShape = Arrays.copyOf(this.shape.toHeapArray(), this.shape.getSize());
        newShape[axis] = end - start;

        // Compute stride product after the axis (how many elements per step along that axis)
        int sliceSize = 1;
        for (int i = axis + 1; i < this.shape.getSize(); i++) {
            sliceSize *= this.shape.get(i);
        }

        // Compute stride product before the axis
        int beforeAxis = 1;
        for (int i = 0; i < axis; i++) {
            beforeAxis *= this.shape.get(i);
        }

        // Prepare result data buffer
        DoubleArray newData = new DoubleArray(Arrays.stream(newShape).reduce(1, (a, b) -> a * b));
        int newIndex = 0;

        // Copy all slices along the selected range
        for (int b = 0; b < beforeAxis; b++) {
            int baseIndex = b * this.shape.toHeapArray()[axis] * sliceSize;
            for (int i = start; i < end; i++) {
                int from = baseIndex + i * sliceSize;
                for (int i1 = 0; i1 < newData.getSize(); i1++) {
                    this.data.set(i1 + from, newData.get(i1 + newIndex));
                }
                newIndex += sliceSize;
            }
        }
        return new Tensor(newData, newShape);
    }

    public static Tensor correlate2D(Tensor input, Tensor delta, int stride, int padding) {
        if (input.getRank() != 2 || delta.getRank() != 2) {
            throw new IllegalArgumentException("Input and delta must be 2D tensors.");
        }

        // Input dimensions
        int H_in = input.shape.toHeapArray()[0];
        int W_in = input.shape.toHeapArray()[1];

        // Delta dimensions (Output size from forward pass)
        int H_out = delta.shape.toHeapArray()[0];
        int W_out = delta.shape.toHeapArray()[1];

        // Derive the filter size (F) based on the forward pass formula: F = H_in + 2*P - S * (H_out - 1)
        int filterSizeH = H_in + 2 * padding - stride * (H_out - 1);
        int filterSizeW = W_in + 2 * padding - stride * (W_out - 1);

        if (filterSizeH != filterSizeW) {
            throw new IllegalArgumentException("Correlate2D derived non-square filter sizes.");
        }

        Tensor weightGradient = new Tensor(filterSizeH, filterSizeH);
        Tensor paddedInput = pad2D(input, padding); // Pad the input first

        // Direct Array Access for Optimization
        DoubleArray inputData = paddedInput.data;
        DoubleArray deltaData = delta.data;
        DoubleArray gradData = weightGradient.data;

        int paddedWidth = paddedInput.shape.toHeapArray()[1];
        int deltaWidth = delta.shape.toHeapArray()[1];

        // Parallelize the outer loop (Filter Rows)
        IntStream stream = IntStream.range(0, filterSizeH);
        if (filterSizeH * filterSizeH > PARALLEL_THRESHOLD) {
            stream = stream.parallel();
        }

        stream.forEach(fr -> {
            for (int fc = 0; fc < filterSizeH; fc++) {
                double sum = 0.0;
                // Loop over the Output Delta map
                for (int ho = 0; ho < H_out; ho++) {
                    int h_in_start = ho * stride + fr;
                    int rowOffsetInput = h_in_start * paddedWidth;
                    int rowOffsetDelta = ho * deltaWidth;

                    for (int wo = 0; wo < W_out; wo++) {
                        int w_in = wo * stride + fc;

                        // Calculation: A_prev[r, c] * Delta_Z[h_out, w_out]
                        double valI = inputData.get(rowOffsetInput + w_in);
                        double valD = deltaData.get(rowOffsetDelta + wo);

                        sum += valI * valD;
                    }
                }
                gradData.set(fr * filterSizeH + fc, sum);
            }
        });

        return weightGradient;
    }

    public static Tensor pad2D(Tensor input, int padding) {
        if (padding == 0) return input;

        int H = input.shape.toHeapArray()[0];
        int W = input.shape.toHeapArray()[1];
        int H_pad = H + 2 * padding;
        int W_pad = W + 2 * padding;

        Tensor padded = new Tensor(H_pad, W_pad);

        // Copy row by row using an optimized system call
        for (int r = 0; r < H; r++) {
            int srcPos = r * W;
            int destPos = (r + padding) * W_pad + padding;
            for (int i = 0; i < input.data.getSize(); i++) {
                input.data.set(destPos + i, input.data.get(srcPos + i));
            }
        }
        return padded;
    }

    public static boolean hasSameShape(Tensor tensor1, Tensor tensor2) {
        return Arrays.equals(tensor1.shape.toHeapArray(), tensor2.shape.toHeapArray());
    }

    @Override
    public String toString() {
        return STR."Tensor{rank=\{this.getRank()}, shape=\{Arrays.toString(shape.toHeapArray())}, strides=\{Arrays.toString(strides.toHeapArray())}, data=\{data}}";
    }
}