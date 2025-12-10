package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.networks.JSequential;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

public class ConvolutionLayer extends Layer {
    private final int filterSize;
    private final int numberOfFilters;
    private final int stride;
    private final int padding;
    private Tensor cachedZ;
    private Tensor cachedInputA;

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
        return Tensor.tensorMapping(this.forwardRawOutput(input), this.getActivationFunction().getEquation());
    }

    public Tensor forwardRawOutput(Tensor input) {
        this.cachedInputA = input.copyShape();
        int outputHeight = (input.getShape().toHeapArray()[1] - filterSize + 2 * padding) / stride + 1;
        int outputWidth = (input.getShape().toHeapArray()[2] - filterSize + 2 * padding) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0) {
            throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
        }
        Tensor output = new Tensor(numberOfFilters, outputHeight, outputWidth);
        for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
            Tensor filter = this.getWeights().getSlice(0, filterIndex, filterIndex + 1).reshape(this.getWeights().getShape().toHeapArray()[1], this.getWeights().getShape().toHeapArray()[2], this.getWeights().getShape().toHeapArray()[3]);

            Tensor accumulated = null;
            for (int a = 0; a < input.getShape().toHeapArray()[0]; a++) {
                Tensor inputSlice = input.getSlice(0, a, a + 1).reshape(input.getShape().toHeapArray()[1], input.getShape().toHeapArray()[2]);
                Tensor filterSlice = filter.getSlice(0, a, a + 1).reshape(this.filterSize, this.filterSize);
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
        // FIX: The bias should be added to the raw Z output.
        addBiasToConvOutput(output, this.getBias(), output);
        this.cachedZ = output.copyShape();
        return output; // This is Z
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
        Tensor inputA_prev = this.cachedInputA.copyShape();
        int inputDepth = inputA_prev.getShape().toHeapArray()[0];
        int inputHeight = inputA_prev.getShape().toHeapArray()[1];
        int inputWidth = inputA_prev.getShape().toHeapArray()[2];
        int outputHeight = this.cachedZ.getShape().toHeapArray()[1];
        int outputWidth = this.cachedZ.getShape().toHeapArray()[2];
        Tensor deltaZ;
        boolean isSoftmaxCCE = this.getActivationFunction().equals(ActivationFunction.SOFTMAX) && (gradients.getShape().toHeapArray()[0] == this.cachedZ.getShape().toHeapArray()[0]);
        if (isSoftmaxCCE) {
            deltaZ = gradients;
        } else {
            Tensor activationDerivative = JSequential.getDeactivatedTensor(this.cachedZ, this.getActivationFunction());
            deltaZ = Tensor.elementWiseMultiplication(activationDerivative, gradients);
        }
        Tensor deltaWeights = new Tensor(numberOfFilters, inputDepth, filterSize, filterSize);
        Tensor deltaBiases = new Tensor(numberOfFilters);
        Tensor propagatedGradient = new Tensor(inputDepth, inputHeight, inputWidth);
        for (int f = 0; f < numberOfFilters; f++) {
            Tensor deltaZ_map = deltaZ.getSlice(0, f, f + 1).reshape(outputHeight, outputWidth);
            double biasSum = 0;
            for (int r = 0; r < outputHeight; r++) {
                for (int c = 0; c < outputWidth; c++) {
                    biasSum += deltaZ_map.getEntry(r, c);
                }
            }
            deltaBiases.setEntry(biasSum, f);
            for (int c = 0; c < inputDepth; c++) {
                Tensor input_map = inputA_prev.getSlice(0, c, c + 1).reshape(inputHeight, inputWidth);
                Tensor weightGrad2D = Tensor.correlate2D(input_map, deltaZ_map, stride, padding);
                deltaWeights.setSlice(weightGrad2D, f, c, 0, 0);
                Tensor filter_slice = this.getWeights().getSlice(0, f, f + 1).getSlice(1, c, c + 1).reshape(filterSize, filterSize);
                Tensor flipped_filter = Tensor.flip2D(filter_slice);
                Tensor inputErrorGrad2D = Tensor.convolve2D(deltaZ_map, flipped_filter, 1, this.padding);
                Tensor currentChannelError = propagatedGradient.getSlice(0, c, c + 1); // [1, H_in, W_in]
                Tensor errorToAdd = inputErrorGrad2D.reshape(1, inputHeight, inputWidth);
                currentChannelError.add(errorToAdd);
                propagatedGradient.setSlice(currentChannelError, c, 0, 0);
            }
        }

        return new Tensor[]{deltaWeights, deltaBiases, propagatedGradient};
    }

    private static void addBiasToConvOutput(Tensor output, Tensor bias, Tensor result) {
        if (output.getRank() != 3) {
            throw new IllegalArgumentException("Output tensor must be 3D (filters × height × width).");
        }
        if (bias.getRank() != 1) {
            throw new IllegalArgumentException("Bias tensor must be 1D (one bias per filter).");
        }
        if (result.getRank() != 3) {
            throw new IllegalArgumentException("Result tensor must be 3D (filters × height × width).");
        }

        int numFilters = output.getShape().toHeapArray()[0];
        int outH = output.getShape().toHeapArray()[1];
        int outW = output.getShape().toHeapArray()[2];

        if (bias.getShape().toHeapArray()[0] != numFilters) {
            throw new IllegalArgumentException(STR."Bias length (\{bias.getShape().toHeapArray()[0]}) does not match number of filters (\{numFilters}).");
        }
        if (!Arrays.equals(output.getShape().toHeapArray(), result.getShape().toHeapArray())) {
            throw new IllegalArgumentException(STR."Result shape \{Arrays.toString(result.getShape().toHeapArray())} does not match output shape \{Arrays.toString(output.getShape().toHeapArray())}.");
        }

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
}