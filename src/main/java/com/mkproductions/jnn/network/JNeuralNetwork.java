package com.mkproductions.jnn.network;


import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.Activator;
import com.mkproductions.jnn.entity.Layer;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

public class JNeuralNetwork implements Serializable {
    private boolean DEBUG = true;
    private final int numberOfInputNode;
    private final Layer[] hiddenNodeSchema;
    private final BlockRealMatrix[] weightsMatrices;
    private final BlockRealMatrix[] biasesMatrices;
    private final BlockRealMatrix[] outputMatrices;
    private final BlockRealMatrix[] outputDeltaMatrices;
    private final BlockRealMatrix[] outputGradientMatrices;
    private double learningRate = 0.0001;

    public JNeuralNetwork(double learningRate, int numberOfInputNode, Layer... layers) {
        // Storing the design of the Neural Network
        this.learningRate = learningRate;
        this.numberOfInputNode = numberOfInputNode;
        this.hiddenNodeSchema = layers;
        // Initializing the arrays
        this.weightsMatrices = new BlockRealMatrix[layers.length];
        this.biasesMatrices = new BlockRealMatrix[layers.length];
        this.outputMatrices = new BlockRealMatrix[layers.length];
        this.outputDeltaMatrices = new BlockRealMatrix[layers.length];
        this.outputGradientMatrices = new BlockRealMatrix[layers.length];
        // Assign weights and biases to matrices arrays
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                this.weightsMatrices[a] = new BlockRealMatrix(this.hiddenNodeSchema[a].getNumberOfNodes(), this.numberOfInputNode);
                this.biasesMatrices[a] = new BlockRealMatrix(this.hiddenNodeSchema[a].getNumberOfNodes(), 1);
            } else {
                this.weightsMatrices[a] = new BlockRealMatrix(this.hiddenNodeSchema[a].getNumberOfNodes(), this.hiddenNodeSchema[a - 1].getNumberOfNodes());
                this.biasesMatrices[a] = new BlockRealMatrix(this.hiddenNodeSchema[a].getNumberOfNodes(), 1);
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
        BlockRealMatrix inputMatrix = new BlockRealMatrix(new double[][]{inputs}).transpose();
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                BlockRealMatrix weightedInput = this.weightsMatrices[a].multiply(inputMatrix);
                weightedInput.add(this.biasesMatrices[a]);
                outputMatrices[a] = activateOutput(weightedInput, ActivationFunction.SIGMOID);
                continue;
            }
            BlockRealMatrix weightedInput = this.weightsMatrices[a].multiply(this.outputMatrices[a - 1]);
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
        BlockRealMatrix outputError = new BlockRealMatrix(new double[][]{target}).transpose().subtract(this.outputMatrices[this.outputMatrices.length - 1]);
        for (int layerIndex = this.weightsMatrices.length - 1; layerIndex > 0; layerIndex--) {
            BlockRealMatrix outputMatrix = outputMatrices[layerIndex];
            BlockRealMatrix previousOutputMatrix = outputMatrices[layerIndex - 1];

            BlockRealMatrix gradientMatrix = deactivateOutput(outputMatrix, ActivationFunction.SIGMOID);
            gradientMatrix = elementWiseMultiply(gradientMatrix, outputError);
            gradientMatrix = (BlockRealMatrix) gradientMatrix.scalarMultiply(learningRate);

            outputDeltaMatrices[layerIndex] = gradientMatrix.multiply(previousOutputMatrix.transpose());

            BlockRealMatrix weightTranspose = this.weightsMatrices[layerIndex].transpose();
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
        if (inputs[0].length != this.numberOfInputNode || targets[0].length != this.hiddenNodeSchema[this.hiddenNodeSchema.length - 1].getNumberOfNodes())
            throw new Exception("Mismatch inputs or outputs size.");
        Random random = new Random();
        int lastProgress = 0;

        for (int a = 0; a < epochs; a++) {
            int randomIndex = random.nextInt(inputs.length);
            double[] input = inputs[randomIndex];
            double[] target = targets[randomIndex];
            backPropagate(input, target);
            int progress = ((a + 1) * 100) / epochs;
            StringBuilder stringBuilder = new StringBuilder();
            if (!this.DEBUG) if (lastProgress != progress) {
                stringBuilder.append("Progress: ").append(progress).append(" ");
                stringBuilder.append("|");
                for (int b = 0; b < 100; b++) {
                    if ((b + 1) <= progress) stringBuilder.append("=");
                    else stringBuilder.append(" ");
                }
                stringBuilder.append("|");
                System.out.println(stringBuilder);
            } else {
                double[] output = processInputs(inputs[randomIndex]);
                System.out.println("Error: ");
                for (int b = 0; b < output.length; b++) {
                    System.out.println(output[a] - target[a]);
                }
            }
            lastProgress = progress;
        }
    }

    private void randomizeMatrix(BlockRealMatrix matrix) {
        Random random = new Random();
        for (int a = 0; a < matrix.getRowDimension(); a++)
            for (int b = 0; b < matrix.getColumnDimension(); b++)
                matrix.setEntry(a, b, random.nextDouble() * 2 - 1);
    }

    private BlockRealMatrix activateOutput(BlockRealMatrix matrix, ActivationFunction activationFunction) {
        BlockRealMatrix newMatrix = new BlockRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int a = 0; a < matrix.getColumnDimension(); a++)
            for (int b = 0; b < matrix.getRowDimension(); b++)
                newMatrix.setEntry(b, a, Activator.activate(activationFunction, matrix.getEntry(b, a)));
        return newMatrix;
    }

    private BlockRealMatrix deactivateOutput(BlockRealMatrix matrix, ActivationFunction activationFunction) {
        BlockRealMatrix newMatrix = new BlockRealMatrix(matrix.getData());
        for (int a = 0; a < newMatrix.getColumnDimension(); a++)
            for (int b = 0; b < newMatrix.getRowDimension(); b++)
                newMatrix.setEntry(b, a, Activator.deactivate(activationFunction, matrix.getEntry(b, a)));
        return newMatrix;
    }

    private static BlockRealMatrix elementWiseMultiply(BlockRealMatrix matrix1, BlockRealMatrix matrix2) throws Exception {
        if (matrix1.getRowDimension() != matrix2.getRowDimension() || matrix1.getColumnDimension() != matrix2.getColumnDimension()) {
            throw new Exception("Matrices must have the same dimensions for element-wise multiplication.");
        }
        int numRows = matrix1.getRowDimension();
        int numCols = matrix1.getColumnDimension();
        BlockRealMatrix result = new BlockRealMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result.setEntry(i, j, matrix1.getEntry(i, j) * matrix2.getEntry(i, j));
        return result;
    }


    public int getNumberOfInputNode() {
        return numberOfInputNode;
    }

    public int getHiddenNodeCount(int a) {
        return this.hiddenNodeSchema[a].getNumberOfNodes();
    }

    public int getNumberOfOutputNode() {
        return this.hiddenNodeSchema[this.hiddenNodeSchema.length - 1].getNumberOfNodes();
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public boolean isDEBUG() {
        return DEBUG;
    }

    public void setDEBUG(boolean DEBUG) {
        this.DEBUG = DEBUG;
    }

    private void printRealMatrix(BlockRealMatrix matrix) {
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

    public void printNetwork() {
        ArrayList<Integer> schema = new ArrayList<>();
        schema.add(this.numberOfInputNode);
        for (var i : this.hiddenNodeSchema) {
            schema.add(i.getNumberOfNodes());
        }
        System.out.println(schema);
    }

    public void saveModel(String fileName) throws IOException {
        new ObjectOutputStream(new FileOutputStream(fileName)).writeObject(this);
    }

    public static JNeuralNetwork loadModel(String filePath) throws Exception {
        return (JNeuralNetwork) new ObjectInputStream(new FileInputStream(filePath)).readObject();
    }
}
