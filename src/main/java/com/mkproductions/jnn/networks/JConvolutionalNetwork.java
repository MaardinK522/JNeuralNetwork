package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.activationFunctions.ActivationFunction3D;
import com.mkproductions.jnn.cpu.layers.ConvolutionLayer;
import com.mkproductions.jnn.cpu.layers.Layer;

import java.security.SecureRandom;
import java.util.Random;

public class JConvolutionalNetwork {
    //    private final int inputDepth;
    //    private final int inputWidth;
    //    private final int inputHeight;
    //    private final Layer networkLayers[];
    //
    //    private final Matrix3D[] kernelMatrices;
    //    private final Matrix3D[] biasesMatrices;
    //    private final Matrix3D[] outputMatrices;
    //    private Random random;
    //
    //    public JConvolutionalNetwork(int inputDepth, int inputWidth, int inputHeight, ConvolutionLayer... networkLayers) {
    //        this.inputDepth = inputDepth;
    //        this.inputWidth = inputWidth;
    //        this.inputHeight = inputHeight;
    //        this.networkLayers = networkLayers;
    //        this.kernelMatrices = new Matrix3D[this.networkLayers.length];
    //        this.biasesMatrices = new Matrix3D[this.networkLayers.length];
    //        this.outputMatrices = new Matrix3D[this.networkLayers.length];
    //        this.random = new SecureRandom();
    //
    //        int currentInputDepth = inputDepth;
    //        int currentInputHeight = inputHeight;
    //        int currentInputWidth = inputWidth;
    //
    //        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
    //            Layer layer = this.networkLayers[layerIndex];
    //            if (layer instanceof ConvolutionLayer currentLayer) {
    //                int filterSize = currentLayer.filterSize();
    //                int outputSize = currentLayer.numberOfFilters();
    //                this.kernelMatrices[layerIndex] = new Matrix3D(filterSize, filterSize, currentInputDepth);
    //                randomize(this.kernelMatrices[layerIndex]);
    //
    //                this.biasesMatrices[layerIndex] = new Matrix3D(1, 1, outputSize);
    //                randomize(this.biasesMatrices[layerIndex]);
    //
    //                int outputHeight = (currentInputHeight + 2 * currentLayer.padding() - filterSize) / currentLayer.stride() + 1;
    //                int outputWidth = (currentInputWidth + 2 * currentLayer.padding() - filterSize) / currentLayer.stride() + 1;
    //                if (outputHeight < 1 || outputWidth < 1) {
    //                    throw new IllegalArgumentException(STR."Convolution layer resulted in zero or negative dimensions. Layer index: \{layerIndex}");
    //                }
    //                this.outputMatrices[layerIndex] = new Matrix3D(outputSize, outputHeight, outputWidth);
    //
    //                currentInputDepth = outputSize;
    //                currentInputHeight = outputHeight;
    //                currentInputWidth = outputWidth;
    //            }
    //        }
    //        System.out.println("--- Initialization Complete ---");
    //    }
    //
    //    private Matrix3D getAppliedActivationMatrices(Matrix3D matrix, ActivationFunction3D activationFunction) {
    //        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
    //            return null;
    //        }
    //        return Matrix3D.Matrix3DMapping(matrix, activationFunction.equation);
    //    }
    //
    //    private void randomize(Matrix3D matrix) {
    //        Matrix3D.mapToMatrix3D(matrix, (a, b, c, value) -> -1 + (random.nextDouble() * 2));
    //    }
}