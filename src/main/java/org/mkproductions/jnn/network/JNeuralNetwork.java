package org.mkproductions.jnn.network;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

public class JNeuralNetwork {
    private final int numberOfInputNode;
    private final int[] hiddenNodeSchema;
    private final int numberOfOutputNode;
    private final RealMatrix[] weightsMatrices;
    private final RealMatrix[] biasesMatrices;
    private final RealMatrix[] outputMatrices;
    private final RealMatrix[] outputDeltaMatrices;
    private final RealMatrix[] outputGradientMatrices;
    private static double learningRate = 0.1;

    public JNeuralNetwork(int numberOfInputNode, int[] hiddenNodeSchema, int numberOfOutputNode) {
        // Storing the design of the Neural Network
        this.numberOfInputNode = numberOfInputNode;
        this.hiddenNodeSchema = hiddenNodeSchema;
        this.numberOfOutputNode = numberOfOutputNode;

        // Initializing the arrays
        int networkSchemaLength = hiddenNodeSchema.length + 1;
        this.weightsMatrices = new RealMatrix[networkSchemaLength];
        this.biasesMatrices = new RealMatrix[networkSchemaLength];
        this.outputMatrices = new RealMatrix[networkSchemaLength];
        this.outputDeltaMatrices = new RealMatrix[networkSchemaLength];
        this.outputGradientMatrices = new RealMatrix[networkSchemaLength];
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

    /**
     * @param inputs A double array to predict the output.
     * @return A double array of output predicted by the network.
     */
    public double[] processInputs(double @NotNull [] inputs) throws Exception {
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

    /**
     * Function to train inputs with targets.
     * Random indexing is ussd to create a better model.
     *
     * @param input  2D array of inputs to be learned by network.
     * @param target 2D array to train the network as per inputs index.
     */
    // Function to perform backward-propagation in the neural network.
    private void backPropagate(double[] input, double[] target) throws Exception {
        if (input.length != this.numberOfInputNode) throw new Exception("Mismatch length of inputs to the network.");
        RealMatrix outputError = MatrixUtils.createRealMatrix(new double[][]{target}).transpose().subtract(this.outputMatrices[this.outputMatrices.length - 1]);
        for (int layerIndex = this.weightsMatrices.length - 1; layerIndex > 0; layerIndex--) {
            RealMatrix outputMatrix = outputMatrices[layerIndex];
            RealMatrix previousOutputMatrix = outputMatrices[layerIndex - 1];

            RealMatrix gradientMatrix = deactivateOutput(outputMatrix, ActivationFunction.SIGMOID);
            gradientMatrix = elementWiseMultiply(gradientMatrix, outputError);
            gradientMatrix = gradientMatrix.scalarMultiply(learningRate);

            outputDeltaMatrices[layerIndex] = gradientMatrix.multiply(previousOutputMatrix.transpose());

            RealMatrix weightTranspose = this.weightsMatrices[layerIndex].transpose();
            outputError = weightTranspose.multiply(gradientMatrix);
            this.outputGradientMatrices[layerIndex] = gradientMatrix;
        }

        for (int a = 1; a < this.weightsMatrices.length; a++) {
            this.weightsMatrices[a] = this.weightsMatrices[a].add(outputDeltaMatrices[a]);
            this.biasesMatrices[a] = this.biasesMatrices[a].add(outputGradientMatrices[a]);
        }
    }

    /**
     * Function to train inputs with targets.
     * Random indexing is ussd to create a better model.
     *
     * @param epochs  number of iteration for taken to perform back-propagation.
     * @param inputs  2D array of inputs to be learned by network.
     * @param targets 2D array to train the network as per the random input index.
     */
    public void train(double[][] inputs, double[][] targets, int epochs) throws Exception {
        if (inputs[0].length != this.numberOfInputNode || targets[0].length != this.numberOfOutputNode)
            throw new Exception("Mismatch inputs or outputs size.");
        Random random = new Random();
        for (int a = 0; a < epochs; a++) {
            System.out.println("Epoch: " + (a + 1));
            int randomIndex = random.nextInt(inputs.length);
            backPropagate(inputs[randomIndex], targets[randomIndex]);
            double[] output = processInputs(inputs[randomIndex]);
            double[] target = targets[randomIndex];
            for (int b = 0; b < output.length; b++) {
                System.out.println("Error: " + (target[b] - output[b]));
            }
        }
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

    private static RealMatrix elementWiseMultiply(RealMatrix matrix1, RealMatrix matrix2) throws Exception {
        if (matrix1.getRowDimension() != matrix2.getRowDimension() || matrix1.getColumnDimension() != matrix2.getColumnDimension()) {
            throw new Exception("Matrices must have the same dimensions for element-wise multiplication.");
        }
        int numRows = matrix1.getRowDimension();
        int numCols = matrix1.getColumnDimension();
        RealMatrix result = MatrixUtils.createRealMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result.setEntry(i, j, matrix1.getEntry(i, j) * matrix2.getEntry(i, j));
        return result;
    }


    public int getNumberOfInputNode() {
        return numberOfInputNode;
    }

    public int getHiddenNodeCount(int a) {
        return this.hiddenNodeSchema[a];
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