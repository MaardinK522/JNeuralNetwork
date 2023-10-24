package org.mkproductions.jnn.network;

import com.sun.security.jgss.GSSUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public class JNeuralNetwork {
    private final int numberOfInputNode;
    private final int numberOfHiddenLayers;
    private final int[] hiddenNodeSchema;
    private final int numberOfOutputNode;

    RealMatrix[] weightsMatrices;
    RealMatrix[] biasesMatrices;
    RealMatrix[] outputMatrices;

    private static double learningRate = 0.1;

    public JNeuralNetwork(int numberOfInputNode, int[] hiddenNodeSchema, int numberOfOutputNode) {
        // Storing the design of the Neural Network
        this.numberOfInputNode = numberOfInputNode;
        this.numberOfHiddenLayers = hiddenNodeSchema.length;
        this.hiddenNodeSchema = hiddenNodeSchema;
        this.numberOfOutputNode = numberOfOutputNode;

        // Initializing the arrays
        int networkSchemaLength = hiddenNodeSchema.length + 1;
        this.weightsMatrices = new RealMatrix[networkSchemaLength];
        this.biasesMatrices = new RealMatrix[networkSchemaLength];

        // Assign weights and biases to matrices arrays
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                this.weightsMatrices[a] = MatrixUtils.createRealMatrix(this.hiddenNodeSchema[a], this.numberOfInputNode);
                this.biasesMatrices[a] = MatrixUtils.createRealMatrix(this.hiddenNodeSchema[a], 1);
            } else if (a == this.weightsMatrices.length - 1) {
                this.weightsMatrices[a] = MatrixUtils.createRealMatrix(this.numberOfOutputNode, this.hiddenNodeSchema[a - 1]);
                this.biasesMatrices[a] = MatrixUtils.createRealMatrix(this.numberOfOutputNode, 1);
            } else {
                this.weightsMatrices[a] = MatrixUtils.createRealMatrix(this.hiddenNodeSchema[a], this.hiddenNodeSchema[a - 1]);
                this.biasesMatrices[a] = MatrixUtils.createRealMatrix(this.hiddenNodeSchema[a], 1);
            }
            // Randomizing the weights and bias
            randomizeMatrix(this.weightsMatrices[a]);
            randomizeMatrix(this.biasesMatrices[a]);
        }
    }

    public double[] processInputs(double[] inputs) throws Exception {
        if (inputs.length != this.numberOfInputNode) throw new Exception("Mismatch length of inputs to the network.");
        RealMatrix inputMatrix = MatrixUtils.createRealMatrix(new double[][]{inputs}).transpose();
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                RealMatrix weightedInput = this.weightsMatrices[a].multiply(inputMatrix);
                weightedInput.add(this.biasesMatrices[a]);
                outputMatrices[a] = activateOutput(weightedInput, ActivationFunction.SIGMOID);
                continue;
            }
            RealMatrix weightedInput = this.weightsMatrices[a].multiply(this.outputMatrices[a - 1]);
            weightedInput.add(this.biasesMatrices[a]);
            outputMatrices[a] = activateOutput(weightedInput, ActivationFunction.SIGMOID);
        }
        return outputMatrices[this.outputMatrices.length - 1].transpose().getRow(0);
    }

    private void backPropagate(double[] input, double[] target) throws Exception {
        // Function to perform backward-propagation in the neural network.
    }

    public void train(double[][] inputs, double[][] targets, int epochs) throws Exception {
        // Function to train inputs with targets.
        // Random indexing is ussd to create a better model.
    }

    private void randomizeMatrix(RealMatrix matrix) {
        Random random = new Random();
        for (int a = 0; a < matrix.getRowDimension(); a++)
            for (int b = 0; b < matrix.getColumnDimension(); b++)
                matrix.setEntry(a, b, random.nextDouble() * 2 - 1);
    }

    private RealMatrix activateOutput(RealMatrix matrix, int activationFunction) {
        RealMatrix newMatrix = MatrixUtils.createRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int a = 0; a < matrix.getColumnDimension(); a++)
            for (int b = 0; b < matrix.getRowDimension(); b++)
                newMatrix.setEntry(b, a, ActivationFunction.activate(activationFunction, matrix.getEntry(b, a)));
        return newMatrix;
    }

    private RealMatrix deactivateOutput(RealMatrix matrix, int activationFunction) {
        RealMatrix newMatrix = MatrixUtils.createRealMatrix(matrix.getData());
        for (int a = 0; a < newMatrix.getColumnDimension(); a++)
            for (int b = 0; b < newMatrix.getRowDimension(); b++)
                newMatrix.setEntry(b, a, ActivationFunction.deactivate(activationFunction, matrix.getEntry(b, a)));
        return newMatrix;
    }

    public int getNumberOfInputNode() {
        return numberOfInputNode;
    }

    public int getNumberOfHiddenLayers() {
        return numberOfHiddenLayers;
    }

    public int[] getHiddenNodeSchema() {
        return hiddenNodeSchema;
    }

    public int getNumberOfOutputNode() {
        return numberOfOutputNode;
    }

    public static double getLearningRate() {
        return learningRate;
    }

    public static void setLearningRate(double learningRate) {
        JNeuralNetwork.learningRate = learningRate;
    }

    private void printRealMatrix(RealMatrix matrix) {
        System.out.println("Matrix(rows: " + matrix.getRowDimension() + ",cols: " + matrix.getColumnDimension() + ")");
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("{\n");
        for (int a = 0; a < matrix.getRowDimension(); a++) {
            stringBuilder.append("\t");
            for (int b = 0; b < matrix.getColumnDimension(); b++) {
                stringBuilder.append(matrix.getEntry(a, b)).append(" ");
            }
            stringBuilder.append("\n");
        }
        stringBuilder.append("}\n");
        System.out.println(stringBuilder);
    }
}