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
        int outputHeight = (input.getShape()[1] - filterSize + 2 * padding) / stride + 1;
        int outputWidth = (input.getShape()[2] - filterSize + 2 * padding) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0) {
            throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
        }
        Tensor output = new Tensor(numberOfFilters, outputHeight, outputWidth);
        for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
            // FIX: Use getWeights() once and then slice. Use getSlice instead of slice().
            Tensor filter = this.getWeights().getSlice(0, filterIndex, filterIndex + 1).reshape(this.getWeights().getShape()[1], this.getWeights().getShape()[2], this.getWeights().getShape()[3]);

            Tensor accumulated = null;
            for (int a = 0; a < input.getShape()[0]; a++) {
                Tensor inputSlice = input.getSlice(0, a, a + 1).reshape(input.getShape()[1], input.getShape()[2]);
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
        // Use cached dimensions and input A^(l-1)
        Tensor inputA_prev = this.cachedInputA.copyShape(); // Input volume

        int inputDepth = inputA_prev.getShape()[0];
        int inputHeight = inputA_prev.getShape()[1];
        int inputWidth = inputA_prev.getShape()[2];

        int outputHeight = this.cachedZ.getShape()[1];
        int outputWidth = this.cachedZ.getShape()[2];

        // --- 1. Calculate the Layer Error Delta (dL/dZ) ---
        Tensor deltaZ;

        // Check for the optimized Softmax/CCE case (dL/dZ^L passed directly)
        boolean isSoftmaxCCE = this.getActivationFunction().equals(ActivationFunction.SOFTMAX) && (gradients.getShape()[0] == this.cachedZ.getShape()[0]);
        if (isSoftmaxCCE) {
            deltaZ = gradients;
        } else {
            // FIX: Use cachedZ or cachedA for the derivative (assuming cachedA is correct)
            Tensor activationDerivative = JSequential.getDeactivatedTensor(this.cachedZ, this.getActivationFunction());
            deltaZ = Tensor.elementWiseMultiplication(activationDerivative, gradients);
        }

        // --- 2. Calculate Parameter Gradients (dL/dW and dL/dB) ---
        Tensor deltaWeights = new Tensor(numberOfFilters, inputDepth, filterSize, filterSize);
        Tensor deltaBiases = new Tensor(numberOfFilters);

        // dL/dA_prev: (InputDepth x InputHeight x InputWidth)
        Tensor propagatedGradient = new Tensor(inputDepth, inputHeight, inputWidth);

        // Loop over each filter (F)
        for (int f = 0; f < numberOfFilters; f++) {
            // Delta map for the current filter (H_out x W_out)
            Tensor deltaZ_map = deltaZ.getSlice(0, f, f + 1).reshape(outputHeight, outputWidth);
            double biasSum = 0;

            // 2a. Bias Gradient (dL/dB) -> Sum deltaZ map
            for (int r = 0; r < outputHeight; r++) {
                for (int c = 0; c < outputWidth; c++) {
                    biasSum += deltaZ_map.getEntry(r, c);
                }
            }
            deltaBiases.setEntry(biasSum, f);

            // Loop over each input channel (C_in)
            for (int c = 0; c < inputDepth; c++) {
                // Input map (H_in x W_in)
                Tensor input_map = inputA_prev.getSlice(0, c, c + 1).reshape(inputHeight, inputWidth);

                // 2b. Weight Gradient (dL/dW) -> Correlate A^(l-1) with deltaZ
                Tensor weightGrad2D = Tensor.correlate2D(input_map, deltaZ_map, stride, padding);
                // Set the 2D slice into the 4D weight gradient tensor
                deltaWeights.setSlice(weightGrad2D, f, c, 0, 0); // Assuming a 4D setSlice method

                // 3. Propagated Error (dL/dA^(l-1)) -> Conv deltaZ with Flipped W
                Tensor filter_slice = this.getWeights().getSlice(0, f, f + 1).getSlice(1, c, c + 1).reshape(filterSize, filterSize);

                Tensor flipped_filter = Tensor.flip2D(filter_slice);

                // Assuming convolve2D works for full/padded convolution
                // Note: Stride is 1, Padding must be carefully set to ensure H_in x W_in output
                Tensor inputErrorGrad2D = Tensor.convolve2D(deltaZ_map, flipped_filter, 1, this.padding);
                // Sum the error across all filters (f) for the current input channel (c)
                // FIX: Get the existing channel slice as a mutable 3D object [1, H, W]
                Tensor currentChannelError = propagatedGradient.getSlice(0, c, c + 1); // [1, H_in, W_in]

                // Add the 2D error map to the 3D slice (reshaping necessary if add is 3D only)
                // Assuming add can handle broadcasting or element-wise addition if dimensions match [1, H, W]
                // We add the 2D error map to the H_in x W_in plane of the 3D tensor

                // CRITICAL FIX FOR DIMENSION MISMATCH:
                // We reshape the inputErrorGrad2D to [1, H_in, W_in] for 3D addition/setting.
                // We then call ADD on the current slice and SET it back.

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

        int numFilters = output.getShape()[0];
        int outH = output.getShape()[1];
        int outW = output.getShape()[2];

        if (bias.getShape()[0] != numFilters) {
            throw new IllegalArgumentException(STR."Bias length (\{bias.getShape()[0]}) does not match number of filters (\{numFilters}).");
        }
        if (!Arrays.equals(output.getShape(), result.getShape())) {
            throw new IllegalArgumentException(STR."Result shape \{Arrays.toString(result.getShape())} does not match output shape \{Arrays.toString(output.getShape())}.");
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