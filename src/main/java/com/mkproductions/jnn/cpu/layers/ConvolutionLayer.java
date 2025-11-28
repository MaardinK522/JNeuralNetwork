package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

public class ConvolutionLayer extends Layer {
    private final int filterSize;
    private final int numberOfFilters;
    private final int stride;
    private final int padding;

    public ConvolutionLayer(int filterSize, int numberOfFilters, int stride, int padding, ActivationFunction activation) {
        super("Convolution", activation);
        this.filterSize = filterSize;
        this.numberOfFilters = numberOfFilters;
        this.stride = stride;
        this.padding = padding;
    }

    public int getFilterSize() {
        return filterSize;
    }

    public int getNumberOfFilters() {
        return numberOfFilters;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    @Override
    public @NotNull String toString() {
        return STR."ConvolutionLayer{filterSize=\{filterSize}, activationFunction=\{getActivationFunction()}}";
    }

    @Override
    public Tensor forward(Tensor input) {
        int outputHeight = (input.getShape()[1] - filterSize + 2 * padding) / stride + 1;
        int outputWidth = (input.getShape()[2] - filterSize + 2 * padding) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0) {
            throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
        }
        Tensor output = new Tensor(numberOfFilters, outputHeight, outputWidth);
        for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
            Tensor filter = this.getWeights().slice(0, filterIndex, filterIndex + 1).reshape( // Reshaping
                    this.getWeights().getShape()[1], // Depths
                    this.getWeights().getShape()[2], // Rows
                    this.getWeights().getShape()[3] // Columns
            );
            // Accumulator for each convolution.
            Tensor accumulated = null;
            for (int a = 0; a < input.getShape()[0]; a++) {
                Tensor inputSlice = input.slice(0, a, a + 1).reshape(input.getShape()[1], input.getShape()[2]);
                Tensor filterSlice = filter.slice(0, a, a + 1).reshape(this.filterSize, this.filterSize);
                Tensor convo2D = Tensor.convolve2D(inputSlice, filterSlice, stride, padding);
                if (accumulated == null) {
                    accumulated = convo2D.copy();
                } else {
                    accumulated.add(convo2D);
                }
            }

            assert accumulated != null;
            for (int y = 0; y < outputHeight; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    output.setEntry(accumulated.getEntry(y, x), filterIndex, y, x);
                }
            }
        }
        addBiasToConvOutput(output, this.getBias(), output);
        return output;
    }

    @Override
    public Tensor[] backward(Tensor inputTensor, Tensor gradients) {
        Tensor outputTensor = forward(inputTensor);
        Tensor deactivatedOutputTensor = getDeactivatedActivationFunctionTensor(outputTensor, ActivationFunction.SOFTMAX);

        Tensor biases;
        if (this.getActivationFunction() == ActivationFunction.SOFTMAX) {
            this.setBias(Tensor.matrixMultiplication(deactivatedOutputTensor, gradients));
        } else {
            this.setBias(Tensor.elementWiseMultiplication(deactivatedOutputTensor, gradients));
        }
        return null;
    }

    private static void addBiasToConvOutput(Tensor output, Tensor bias, Tensor result) {
        // --- Shape validation ---
        if (output.getRank() != 3) {
            throw new IllegalArgumentException("Output tensor must be 3D (filters × height × width).");
        }
        if (bias.getRank() != 1) {
            throw new IllegalArgumentException("Bias tensor must be 1D (one bias per filter).");
        }
        if (result.getRank() != 3) {
            throw new IllegalArgumentException("Result tensor must be 3D (filters × height × width).");
        }

        int numFilters = output.getShape()[0];
        int outH = output.getShape()[1];
        int outW = output.getShape()[2];

        // --- Shape compatibility checks ---
        if (bias.getShape()[0] != numFilters) {
            throw new IllegalArgumentException(STR."Bias length (\{bias.getShape()[0]}) does not match number of filters (\{numFilters}).");
        }
        if (!Arrays.equals(output.getShape(), result.getShape())) {
            throw new IllegalArgumentException(STR."Result shape \{Arrays.toString(result.getShape())} does not match output shape \{Arrays.toString(output.getShape())}.");
        }

        // --- Bias addition ---
        for (int f = 0; f < numFilters; f++) {
            double biasVal = bias.getEntry(f);
            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {
                    double val = output.getEntry(f, y, x) + biasVal;
                    result.setEntry(val, f, y, x);
                }
            }
        }
    }

    public static Tensor getDeactivatedActivationFunctionTensor(Tensor activatedTensor, @NotNull ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (activatedTensor.getShape()[1] != 1) {
                throw new IllegalArgumentException("Softmax derivative expects a single vector output.");
            }
            int n = Math.max(activatedTensor.getShape()[0], activatedTensor.getShape()[1]);
            Tensor result = new Tensor(n, n);
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    double entry = a == b ? (activatedTensor.getEntry(a, b) * (1 - activatedTensor.getEntry(a, b))) : -activatedTensor.getEntry(a, b) * activatedTensor.getEntry(b, a);
                    result.setEntry(entry, a, b);
                }
            }
        }
        return Tensor.tensorMapping(activatedTensor, activationFunction.getDerivative());
    }
}