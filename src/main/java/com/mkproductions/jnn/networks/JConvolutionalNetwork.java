package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.layers.ConvolutionLayer;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.layers.FlattenLayer;
import com.mkproductions.jnn.cpu.layers.Layer;
import com.mkproductions.jnn.cpu.layers.PoolingLayer;

import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Random;

public class JConvolutionalNetwork {
    private final int inputDepth;
    private final int inputWidth;
    private final int inputHeight;
    private final Tensor[] kernelTensors;
    private final Tensor[] biasTensors;
    private final Tensor[] outputTensors;
    private final Layer[] networkLayers;
    private final Random random;

    public JConvolutionalNetwork(int inputDepth, int inputWidth, int inputHeight, ConvolutionLayer... networkLayers) {
        this.inputDepth = inputDepth;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.networkLayers = networkLayers;
        this.random = new SecureRandom();
        this.kernelTensors = new Tensor[this.networkLayers.length];
        this.biasTensors = new Tensor[this.networkLayers.length];
        this.outputTensors = new Tensor[this.networkLayers.length];
        int currentInputDepth = inputDepth;
        int currentInputWidth = inputWidth;
        int currentInputHeight = inputHeight;

        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            if (this.networkLayers[layerIndex] instanceof ConvolutionLayer convolutionLayer) {
                Tensor kernel = new Tensor(convolutionLayer.getFilterSize(), convolutionLayer.getFilterSize(), currentInputDepth);
                Tensor bias = new Tensor(convolutionLayer.getPadding());
                int outputHeight = (currentInputHeight + 2 * convolutionLayer.getPadding() - convolutionLayer.getPadding()) / convolutionLayer.getPadding() + 1;
                int outputWidth = (currentInputWidth + 2 * convolutionLayer.getPadding() - convolutionLayer.getPadding()) / convolutionLayer.getPadding() + 1;

                if (outputWidth < 1 || outputHeight < 1) {
                    throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
                }
                currentInputDepth = convolutionLayer.getStride();
                currentInputWidth = outputWidth;
                currentInputHeight = outputHeight;

                randomize(bias);
                randomize(kernel);

                this.kernelTensors[layerIndex] = kernel;
                this.biasTensors[layerIndex] = bias;
            } else if (this.networkLayers[layerIndex] instanceof PoolingLayer poolingLayer) {
                int outputHeight = (currentInputHeight - poolingLayer.getStride()) / poolingLayer.getStride() + 1;
                int outputWidth = (currentInputWidth - poolingLayer.getPoolSize()) / poolingLayer.getStride() + 1;

                if (outputWidth < 1 || outputHeight < 1) {
                    throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
                }
                currentInputWidth = outputWidth;
                currentInputHeight = outputHeight;
            } else if (this.networkLayers[layerIndex] instanceof DenseLayer denseLayer) {
                Tensor weight = new Tensor(currentInputWidth, denseLayer.getNumberOfNodes());
                randomize(weight);
                this.kernelTensors[layerIndex] = weight;
                currentInputWidth = denseLayer.getNumberOfNodes();
            } else {
                throw new IllegalArgumentException("Invalid layer type.");
            }
        }
    }

    private Tensor forwardPropagation(Tensor input) {
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            if (this.networkLayers[layerIndex] instanceof ConvolutionLayer convolutionLayer) {
                Tensor prototypeKernel = this.kernelTensors[layerIndex];
                Tensor prototypeBias = this.biasTensors[layerIndex];
                Tensor[] filters = new Tensor[convolutionLayer.getNumberOfFilters()];
                for (int a = 0; a < filters.length; a++) {
                    Tensor filter = Tensor.copyShape(prototypeKernel);
                    randomize(filter);
                    filters[a] = filter;
                }
                this.outputTensors[layerIndex] = Tensor.convolve(input, filters, convolutionLayer.getPadding(), convolutionLayer.getStride());
                this.outputTensors[layerIndex] = Tensor.add(this.outputTensors[layerIndex], prototypeBias);
                this.outputTensors[layerIndex] = Tensor.tensorMapping(this.outputTensors[layerIndex], this.networkLayers[layerIndex].getActivationFunction().getEquation());
            } else if (this.networkLayers[layerIndex] instanceof PoolingLayer poolingLayer) {
                this.outputTensors[layerIndex] = Tensor.maxPool(this.outputTensors[layerIndex - 1], poolingLayer.getPoolSize(), poolingLayer.getStride());
            } else if (this.networkLayers[layerIndex] instanceof FlattenLayer) {
                this.outputTensors[layerIndex] = this.outputTensors[layerIndex - 1].reshape(this.outputTensors[layerIndex - 1].getData().length);
            } else if (this.networkLayers[layerIndex] instanceof DenseLayer denseLayer) {
                int inputNode = this.outputTensors[layerIndex - 1].getShape()[0];
                this.outputTensors[layerIndex] = Tensor.matrixMultiplication(this.outputTensors[layerIndex - 1].reshape(inputNode), this.kernelTensors[layerIndex]);
                this.outputTensors[layerIndex] = Tensor.add(this.outputTensors[layerIndex], this.biasTensors[layerIndex].reshape(1, this.biasTensors[layerIndex].getShape()[0]));
                this.outputTensors[layerIndex] = getAppliedActivationMatrices(this.outputTensors[layerIndex], denseLayer.getActivationFunction());
            }
        }
        return this.outputTensors[this.outputTensors.length - 1];
    }

    private Tensor getAppliedActivationMatrices(Tensor matrix, ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            return null;
        }
        return Tensor.tensorMapping(matrix, activationFunction.getEquation());
    }

    private void randomize(Tensor matrix) {
        Tensor.tensorMapping(matrix, (flatIndex, value) -> -1 + (random.nextDouble() * 2));
    }
}