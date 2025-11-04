package com.mkproductions.jnn.network;

import com.mkproductions.jnn.cpu.layers.ConvolutionLayer;
import com.mkproductions.jnn.cpu.entity.Matrix3D;

import java.util.Random;

public class JConvolutionalNetwork {
    private final int numberOfInputsNodes;
    private final ConvolutionLayer networkLayers[];

    private final Matrix3D[] kernels;
    private final Matrix3D[] outputMatrices;
    private Random random;

    public JConvolutionalNetwork(int numberOfInputsNodes, ConvolutionLayer... networkLayers) {
        this.numberOfInputsNodes = numberOfInputsNodes;
        this.networkLayers = networkLayers;
        this.kernels = new Matrix3D[this.networkLayers.length];
        this.outputMatrices = new Matrix3D[this.networkLayers.length];
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            // What will be the size of the kernel matrix?
            this.kernels[layerIndex] = new Matrix3D(1, 1, 1);
        }
        this.random = new Random();
    }

    private void randomize(Matrix3D matrix) {
        Matrix3D.mapToMatrix3D(matrix, (a, b, c, value) -> -1 + (random.nextDouble() * 2));
    }
}