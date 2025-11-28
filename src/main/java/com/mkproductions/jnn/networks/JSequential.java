package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.layers.ConvolutionLayer;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.layers.FlattenLayer;
import com.mkproductions.jnn.cpu.layers.Layer;
import com.mkproductions.jnn.cpu.layers.PoolingLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;

import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Random;

import static java.lang.StringTemplate.STR;


public class JSequential {
    private final int inputDepth;
    private final int inputWidth;
    private final int inputHeight;
    private final Layer[] networkLayers;
    private final Tensor[] kernelTensors;
    private final Tensor[] biasTensors;
    private final Tensor[] outputTensors;
    private final Tensor[] rawOutputTensors;
    private final Tensor[] errorTensors;
    private final Tensor[] deltaWeightsTensor;
    private final Tensor[] deltaBiasesTensor;
    private final int[][][] maxPoolMasks;
    private static final Random random = new SecureRandom();
    private final LossFunction lossFunction;
    private final JNetworkOptimizer optimizer;
    private double learningRate;

    public JSequential(int[] inputShape, LossFunction lossFunction, JNetworkOptimizer optimizer, Layer... networkLayers) {
        this.inputDepth = inputShape[0];
        this.inputWidth = inputShape[1];
        this.inputHeight = inputShape[2];
        this.networkLayers = networkLayers;
        this.kernelTensors = new Tensor[this.networkLayers.length];
        this.biasTensors = new Tensor[this.networkLayers.length];
        this.outputTensors = new Tensor[this.networkLayers.length];
        this.errorTensors = new Tensor[this.networkLayers.length];
        this.rawOutputTensors = new Tensor[this.networkLayers.length];
        this.deltaWeightsTensor = new Tensor[this.networkLayers.length];
        this.deltaBiasesTensor = new Tensor[this.networkLayers.length];
        this.maxPoolMasks = new int[this.networkLayers.length][][];
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.learningRate = 0.01;
        int currentInputDepth = inputDepth;
        int currentInputWidth = inputWidth;
        int currentInputHeight = inputHeight;

        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            switch (this.networkLayers[layerIndex]) {
                case ConvolutionLayer convolutionLayer -> {

                    int outputHeight = (currentInputHeight + 2 * convolutionLayer.getPadding() - convolutionLayer.getFilterSize()) / convolutionLayer.getStride() + 1;
                    int outputWidth = (currentInputWidth + 2 * convolutionLayer.getPadding() - convolutionLayer.getFilterSize()) / convolutionLayer.getStride() + 1;

                    if (outputWidth < 1 || outputHeight < 1) {
                        throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
                    }

                    Tensor kernel = new Tensor(convolutionLayer.getNumberOfFilters(), currentInputDepth, convolutionLayer.getFilterSize(), convolutionLayer.getFilterSize());
                    Tensor bias = new Tensor(convolutionLayer.getNumberOfFilters());

                    randomize(bias);
                    randomize(kernel);
                    this.kernelTensors[layerIndex] = kernel;
                    this.biasTensors[layerIndex] = bias;
                    this.outputTensors[layerIndex] = new Tensor(convolutionLayer.getNumberOfFilters(), outputHeight, outputWidth);
                    currentInputDepth = convolutionLayer.getNumberOfFilters();
                    currentInputWidth = outputWidth;
                    currentInputHeight = outputHeight;
                }
                case PoolingLayer poolingLayer -> {
                    int outputHeight = (currentInputHeight - poolingLayer.getPoolSize()) / poolingLayer.getStride() + 1;
                    int outputWidth = (currentInputWidth - poolingLayer.getPoolSize()) / poolingLayer.getStride() + 1;

                    if (outputWidth < 1 || outputHeight < 1) {
                        throw new IllegalArgumentException("Pooling resulted in zero or negative dimensions.");
                    }
                    this.outputTensors[layerIndex] = new Tensor(currentInputDepth, outputHeight, outputWidth);
                    currentInputWidth = outputWidth;
                    currentInputHeight = outputHeight;
                }
                case DenseLayer denseLayer -> {
                    boolean flattenFound = false;
                    for (int i = layerIndex - 1; i >= 0; i--) {
                        if (this.networkLayers[i] instanceof FlattenLayer) {
                            flattenFound = true;
                            break;
                        }
                        if (this.networkLayers[i] instanceof ConvolutionLayer || this.networkLayers[i] instanceof PoolingLayer) {
                            break; // No flatten found before another feature-map-producing layer
                        }
                    }
                    if (!flattenFound) {
                        throw new IllegalArgumentException("Dense layer must be placed after a Flatten layer in the network architecture.");
                    }
                    Tensor weight = new Tensor(currentInputWidth * currentInputHeight * currentInputDepth, denseLayer.getNumberOfNodes());
                    randomize(weight);
                    this.kernelTensors[layerIndex] = weight;
                    currentInputWidth = denseLayer.getNumberOfNodes();
                    this.biasTensors[layerIndex] = new Tensor(1, denseLayer.getNumberOfNodes());
                    this.outputTensors[layerIndex] = this.biasTensors[layerIndex].copyShape();
                }
                case FlattenLayer _ -> {
                    this.outputTensors[layerIndex] = new Tensor(1, currentInputWidth * currentInputHeight * currentInputDepth);
                    currentInputWidth = currentInputWidth * currentInputHeight * currentInputDepth;
                    currentInputHeight = 1;
                    currentInputDepth = 1;
                }
                case null, default -> {
                    System.err.println(this.networkLayers[layerIndex]);
                    throw new IllegalArgumentException("Unsupported layer type detected in network architecture.");
                }
            }
        }
    }

    public void forwardPropagation(Tensor input) {
        if (input.getRank() != 3) {
            throw new IllegalArgumentException("Input must be 3D tensor");
        } else if (this.inputDepth != input.getShape()[0] || this.inputHeight != input.getShape()[1] || this.inputWidth != input.getShape()[2]) {
            throw new IllegalArgumentException("Input dimensions do not match network architecture");
        }

        Tensor currentInput = input.copy();
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            if (this.networkLayers[layerIndex] instanceof ConvolutionLayer convolutionLayer) {
                Tensor kernel = this.kernelTensors[layerIndex];

                int outputHeight = (currentInput.getShape()[1] - convolutionLayer.getFilterSize() + 2 * convolutionLayer.getPadding()) / convolutionLayer.getStride() + 1;
                int outputWidth = (currentInput.getShape()[2] - convolutionLayer.getFilterSize() + 2 * convolutionLayer.getPadding()) / convolutionLayer.getStride() + 1;

                if (outputHeight < 1 || outputWidth < 1) {
                    throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
                }
                Tensor output = new Tensor(convolutionLayer.getNumberOfFilters(), outputHeight, outputWidth);
                for (int filterNumber = 0; filterNumber < convolutionLayer.getNumberOfFilters(); filterNumber++) {
                    Tensor filter = kernel.slice(0, filterNumber, filterNumber + 1).reshape( // Reshaping
                            kernel.getShape()[1], // Depth,
                            kernel.getShape()[2], // Rows,
                            kernel.getShape()[3]  // Columns
                    );
                    // Accumulator for each convolution.
                    Tensor accumulated = null;
                    for (int d = 0; d < currentInput.getShape()[0]; d++) {
                        Tensor inputSlice = currentInput.slice(0, d, d + 1).reshape(currentInput.getShape()[1], currentInput.getShape()[2]);
                        Tensor filterSlice = filter.slice(0, d, d + 1).reshape(convolutionLayer.getFilterSize(), convolutionLayer.getFilterSize());
                        Tensor conv2D = Tensor.convolve2D(inputSlice, filterSlice, convolutionLayer.getStride(), convolutionLayer.getPadding());
                        if (accumulated == null) {
                            accumulated = conv2D.copy();
                        } else {
                            accumulated.add(conv2D);
                        }
                    }
                    assert accumulated != null;
                    for (int y = 0; y < outputHeight; y++) {
                        for (int x = 0; x < outputWidth; x++) {
                            output.setEntry(accumulated.getEntry(y, x), filterNumber, y, x);
                        }
                    }
                }
                addBiasToConvOutput(output, this.biasTensors[layerIndex], this.outputTensors[layerIndex]);
                this.rawOutputTensors[layerIndex] = output;
                this.outputTensors[layerIndex].mapTensor(this.networkLayers[layerIndex].getActivationFunction().getEquation());
                currentInput = this.outputTensors[layerIndex];
            } else if (this.networkLayers[layerIndex] instanceof PoolingLayer poolingLayer) {
                if (poolingLayer.getPoolingLayerType() == PoolingLayer.PoolingLayerType.MAX) {
                    this.outputTensors[layerIndex] = maxPool(currentInput, poolingLayer.getPoolSize(), poolingLayer.getStride());
                } else if (poolingLayer.getPoolingLayerType() == PoolingLayer.PoolingLayerType.AVG) {
                    this.outputTensors[layerIndex] = averagePool(currentInput, poolingLayer.getPoolSize(), poolingLayer.getStride());
                }
                currentInput = this.outputTensors[layerIndex];
            } else if (this.networkLayers[layerIndex] instanceof FlattenLayer) {
                this.outputTensors[layerIndex] = currentInput.reshape(currentInput.getData().length);
                currentInput = this.outputTensors[layerIndex];
            } else if (this.networkLayers[layerIndex] instanceof DenseLayer denseLayer) {
                int inputNode = currentInput.getShape()[0];
                this.outputTensors[layerIndex] = Tensor.matrixMultiplication(this.outputTensors[layerIndex - 1].reshape(1, inputNode), this.kernelTensors[layerIndex]);
                this.outputTensors[layerIndex] = Tensor.add(this.outputTensors[layerIndex], this.biasTensors[layerIndex].reshape(1, this.biasTensors[layerIndex].getShape()[1]));
                this.rawOutputTensors[layerIndex] = this.outputTensors[layerIndex];
                this.outputTensors[layerIndex] = getAppliedActivationTensors(this.outputTensors[layerIndex], denseLayer.getActivationFunction());
                currentInput = this.outputTensors[layerIndex];
            }
            System.out.println(
                    STR."\{this.networkLayers[layerIndex].getName()} Layer index: \{layerIndex},\n\tOutput shape: \{this.outputTensors[layerIndex]},\n\t Layer name:\{this.networkLayers[layerIndex].getActivationFunction()
                            .name()}");
        }
        //        return this.outputTensors[this.networkLayers.length - 1];
    }

    private void backPropagation(Tensor input, Tensor target) {
        Tensor currentInput = input.copy();
        Tensor outputTensor = this.outputTensors[this.outputTensors.length - 1];
        Tensor errorTensor = this.lossFunction.getDerivativeTensor(outputTensor, target);
    }

    /**
     *
     */
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

    private static Tensor maxPool(Tensor input, int poolSize, int stride) {
        if (input.getRank() != 3) {
            throw new IllegalArgumentException("Input for pooling must be a Rank-3 Tensor (C, H, W).");
        }

        int C = input.getShape()[0];
        int H_in = input.getShape()[1];
        int W_in = input.getShape()[2];

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

    private static Tensor averagePool(Tensor input, int poolSize, int stride) {
        if (input.getRank() != 3) {
            throw new IllegalArgumentException("Input for pooling must be a Rank-3 Tensor (C, H, W).");
        }

        int C = input.getShape()[0];
        int H_in = input.getShape()[1];
        int W_in = input.getShape()[2];

        int H_out = (H_in - poolSize) / stride + 1;
        int W_out = (W_in - poolSize) / stride + 1;

        if (H_out <= 0 || W_out <= 0) {
            throw new IllegalArgumentException("Pooling resulted in zero or negative dimensions.");
        }

        Tensor output = new Tensor(C, H_out, W_out);

        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {

                    double sum = 0.0;
                    int count = 0;

                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    for (int p_h = 0; p_h < poolSize; p_h++) {
                        for (int p_w = 0; p_w < poolSize; p_w++) {
                            double entry = input.getEntry(c, h_start + p_h, w_start + p_w);
                            sum += entry;
                            count++;
                        }
                    }

                    double avg = sum / count;
                    output.setEntry(avg, c, h_out, w_out);
                }
            }
        }

        return output;
    }

    private Tensor getAppliedActivationTensors(Tensor matrix, ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            return null;
        }
        return Tensor.tensorMapping(matrix, activationFunction.getEquation());
    }

    private Tensor getDeactivatedTensor(Tensor activatedTensor, ActivationFunction activationFunction) {
        if (activationFunction.equals(ActivationFunction.SOFTMAX)) {
            throw new IllegalArgumentException("Softmax yet not implemented");
        }
        return Tensor.tensorMapping(activatedTensor, activationFunction.getDerivative());
    }

    public static void randomize(Tensor tensor) {
        for (int i = 0; i < tensor.getData().length; i++) {
            double value = -1 + (random.nextDouble() * 2);
            //            System.out.println(value);
            tensor.getData()[i] = value;
        }
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}