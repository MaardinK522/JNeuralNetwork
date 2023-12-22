package com.mkproductions.jnn.network;

import com.mkproductions.jnn.entity.*;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.sql.SQLOutput;
import java.util.Arrays;
import java.util.Random;

public class JNeuralNetwork implements Serializable {
    private final int numberOfInputNode;
    private final Layer[] netWorkLayers;
    private final Matrix[] weightsMatrices;
    private final Matrix[] biasesMatrices;
    private double learningRate;

    public JNeuralNetwork(int numberOfInputNode, Layer... netWorkLayers) {
        // Storing the design of the Neural Network
        this.learningRate = 0.01;
        this.numberOfInputNode = numberOfInputNode;
        this.netWorkLayers = netWorkLayers;
        // Initializing the arrays
        this.weightsMatrices = new Matrix[netWorkLayers.length];
        this.biasesMatrices = new Matrix[netWorkLayers.length];
        // Assign weights and biases to matrix arrays
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                this.weightsMatrices[a] = new Matrix(this.netWorkLayers[a].numberOfNodes(), this.numberOfInputNode);
            } else {
                this.weightsMatrices[a] = new Matrix(this.netWorkLayers[a].numberOfNodes(), this.netWorkLayers[a - 1].numberOfNodes());
            }
            this.biasesMatrices[a] = new Matrix(this.netWorkLayers[a].numberOfNodes(), 1);
            // Randomizing the weights and bias
            this.weightsMatrices[a].randomize();
            this.biasesMatrices[a].randomize();
        }
    }

    /**
     * Process inputs and produces outputs as per the network schema.
     *
     * @param inputs A double array to predict the output.
     * @return double array of output predicted by the network.
     */
    public double[] processInputs(double @NotNull [] inputs) {
        if (inputs.length != this.numberOfInputNode)
            throw new RuntimeException("Mismatch length of inputs to the network.");
        Matrix inputMatrix = Matrix.fromArray(inputs).transpose();

        Matrix[] outputMatrices = new Matrix[this.netWorkLayers.length];
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                Matrix weightedInput = Matrix.matrixMultiplication(this.weightsMatrices[a], inputMatrix);
                weightedInput.add(this.biasesMatrices[a]);
                outputMatrices[a] = Matrix.matrixMapping(weightedInput, this.netWorkLayers[a].activationFunction().equation);
                continue;
            }
            Matrix weightedInput = Matrix.matrixMultiplication(this.weightsMatrices[a], outputMatrices[a - 1]);
            weightedInput.add(this.biasesMatrices[a]);
            outputMatrices[a] = Matrix.matrixMapping(weightedInput, this.netWorkLayers[a].activationFunction().equation);
        }
        return outputMatrices[outputMatrices.length - 1].transpose().getRowCount(0);
    }

    /**
     * Function to train inputs with targets.
     * Random indexing is used to create a better model.
     *
     * @param input  2D array of inputs to be learned by network.
     * @param target 2D array to train the network as per inputs index.
     */
    private void backPropagate(double[] input, double[] target) {
        if (input.length != this.numberOfInputNode)
            throw new RuntimeException("Mismatch length of inputs to the network.");
        Matrix inputMatrix = Matrix.fromArray(input).transpose();

        Matrix[] outputMatrices = new Matrix[this.netWorkLayers.length];
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            Matrix weightedInput;
            if (a == 0) weightedInput = Matrix.matrixMultiplication(this.weightsMatrices[a], inputMatrix);
            else weightedInput = Matrix.matrixMultiplication(this.weightsMatrices[a], outputMatrices[a - 1]);

            weightedInput.add(this.biasesMatrices[a]);

            outputMatrices[a] = Matrix.matrixMapping(weightedInput, this.netWorkLayers[a].activationFunction().equation);
        }

        Matrix outputMatrix = outputMatrices[outputMatrices.length - 1];

        Matrix targetMatrix = new Matrix(new double[][]{target}).transpose();

        // Output error matrix.
        Matrix errorMatrix = Matrix.scalarMultiply(Matrix.subtract(outputMatrix, targetMatrix), -1);

        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {

            // Calculating gradients
            Matrix gradientMatrix = Matrix.matrixMapping(outputMatrices[layerIndex], this.netWorkLayers[layerIndex].activationFunction().derivative);
            gradientMatrix.elementWiseMultiply(errorMatrix);
            gradientMatrix.scalarMultiply(-this.learningRate);
            
            errorMatrix = Matrix.matrixMultiplication(this.weightsMatrices[layerIndex].transpose(), errorMatrix);

            // Getting the inputs of the current layer.
            Matrix previousOutputMatrixTranspose;
            if (layerIndex == 0) previousOutputMatrixTranspose = inputMatrix.transpose();
            else previousOutputMatrixTranspose = outputMatrices[layerIndex - 1].transpose();
            Matrix deltaWeightsMatrix = Matrix.matrixMultiplication(gradientMatrix, previousOutputMatrixTranspose);

            this.weightsMatrices[layerIndex].subtract(deltaWeightsMatrix);
            this.biasesMatrices[layerIndex].subtract(gradientMatrix);
        }
    }

    /**
     * Function to train inputs with targets.
     * Random indexing is used to create a better model.
     *
     * @param epochs         number of iterations for taken to perform back-propagation.
     * @param trainingInputs 2D array of inputs to be learned by network.
     * @param targetOutputs  2D array to train the network as per the random input index.
     */
    public void train(double[][] trainingInputs, double[][] targetOutputs, int epochs) {
        if (trainingInputs[0].length != this.numberOfInputNode || targetOutputs[0].length != this.netWorkLayers[this.netWorkLayers.length - 1].numberOfNodes())
            throw new RuntimeException("Mismatch inputs or outputs size.");
        for (int a = 0; a < epochs; a++) {
            int randomIndex = new Random().nextInt(0, trainingInputs.length);
            double[] training_input = trainingInputs[randomIndex];
            double[] targets = targetOutputs[randomIndex];
            System.out.println("Training input: " + Arrays.toString(training_input));
            System.out.println("Training output: " + Arrays.toString(targets));
            backPropagate(training_input, targets);
        }
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}