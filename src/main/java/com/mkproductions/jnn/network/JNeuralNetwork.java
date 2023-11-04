package com.mkproductions.jnn.network;

import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.Mapper;
import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.entity.Matrix;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.Random;

public class JNeuralNetwork implements Serializable {
    private final int numberOfInputNode;
    private final Layer[] hiddenNodeSchema;
    private final Matrix[] weightsMatrices;
    private final Matrix[] biasesMatrices;
    private final Matrix[] outputMatrices;
    private final Matrix[] outputDeltaMatrices;
    private final Matrix[] outputGradientMatrices;
    private double learningRate;

    public JNeuralNetwork(int numberOfInputNode, Layer... layers) {
        // Storing the design of the Neural Network
        this.learningRate = 0.01;
        this.numberOfInputNode = numberOfInputNode;
        this.hiddenNodeSchema = layers;
        // Initializing the arrays
        this.weightsMatrices = new Matrix[layers.length];
        this.biasesMatrices = new Matrix[layers.length];
        this.outputMatrices = new Matrix[layers.length];
        this.outputDeltaMatrices = new Matrix[layers.length];
        this.outputGradientMatrices = new Matrix[layers.length];
        // Assign weights and biases to matrix arrays
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                this.weightsMatrices[a] = new Matrix(this.hiddenNodeSchema[a].numberOfNodes(), this.numberOfInputNode);
            } else {
                this.weightsMatrices[a] = new Matrix(this.hiddenNodeSchema[a].numberOfNodes(), this.hiddenNodeSchema[a - 1].numberOfNodes());
            }
            this.biasesMatrices[a] = new Matrix(this.hiddenNodeSchema[a].numberOfNodes(), 1);
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

        Matrix inputMatrix = Matrix.fromArray(inputs);
        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                Matrix weightedInput = this.weightsMatrices[a].multiply(inputMatrix);
                weightedInput.add(this.biasesMatrices[a]);
                outputMatrices[a] = applyActivationToMatrix(weightedInput, this.hiddenNodeSchema[a].activationFunction());
                continue;
            }
            Matrix weightedInput = this.weightsMatrices[a].multiply(this.outputMatrices[a - 1]);
            weightedInput.add(this.biasesMatrices[a]);
            outputMatrices[a] = applyActivationToMatrix(weightedInput, this.hiddenNodeSchema[a].activationFunction());
        }
        return outputMatrices[this.outputMatrices.length - 1].transpose().getRow(0);
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
        Matrix targetMatrix = new Matrix(new double[][]{target}).transpose();
        Matrix outputError = this.outputMatrices[this.outputMatrices.length - 1].subtract(targetMatrix).scalarMultiply(2);
        for (int layerIndex = this.hiddenNodeSchema.length - 1; layerIndex >= 0; layerIndex--) {
            // Getting the inputs of the current layer.
            Matrix previousOutputMatrix;
            if (layerIndex == 0) previousOutputMatrix = new Matrix(new double[][]{input}).transpose();
            else previousOutputMatrix = this.outputMatrices[layerIndex - 1];

            // Getting the output of the layer.
            Matrix outputMatrix = this.outputMatrices[layerIndex];

            // Calculating gradient.
            // Calculating partial derivative here.
            Matrix gradientMatrix;
//            if (layerIndex == this.outputMatrices.length - 1)
//                gradientMatrix = outputError;
//            else
                gradientMatrix = deactivateOutput(outputMatrix, this.hiddenNodeSchema[layerIndex].activationFunction());

            gradientMatrix = Matrix.elementWiseMultiply(outputError, gradientMatrix);
            gradientMatrix = gradientMatrix.scalarMultiply(this.learningRate);

            // Calculating deltas
            this.outputDeltaMatrices[layerIndex] = gradientMatrix.multiply(previousOutputMatrix.transpose());

            Matrix weightTranspose = this.weightsMatrices[layerIndex].transpose();
            outputError = weightTranspose.multiply(outputError);
            this.outputGradientMatrices[layerIndex] = gradientMatrix;
        }

        for (int a = 0; a < this.outputGradientMatrices.length; a++) {
            this.weightsMatrices[a] = this.weightsMatrices[a].add(outputDeltaMatrices[a]);
            this.biasesMatrices[a] = this.biasesMatrices[a].add(outputGradientMatrices[a]);
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
    public void train(double[][] trainingInputs, double[][] targetOutputs, int epochs) throws Exception {
        if (trainingInputs[0].length != this.numberOfInputNode || targetOutputs[0].length != this.hiddenNodeSchema[this.hiddenNodeSchema.length - 1].numberOfNodes())
            throw new RuntimeException("Mismatch inputs or outputs size.");
//        int lastProgress = 0;
        for (int a = 0; a < epochs; a++) {
            int randomIndex = new Random().nextInt(0, trainingInputs.length);
            double[] inputs = trainingInputs[randomIndex];
            double[] targets = targetOutputs[randomIndex];
            double[] outputs = processInputs(inputs);

            backPropagate(inputs, targets);

            double networkCost = networkCost(outputs, targets);
            System.out.println("Cost of the network: " + networkCost);
        }
    }

    private double networkCost(double[] outputs, double[] targets) {
        double cost = 0;
        for (int a = 0; a < outputs.length; a++) {
            double diff = targets[a] - outputs[a];
            cost += diff * diff;
        }
        double n = 1.0 / outputs.length;
        return cost * n;
    }

    private Matrix applyActivationToMatrix(Matrix matrix, ActivationFunction activationFunction) {
        Matrix newMatrix = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int a = 0; a < matrix.getRow(); a++)
            for (int b = 0; b < matrix.getColumn(); b++)
                newMatrix.setEntry(a, b, Mapper.mapActivation(activationFunction, matrix.getEntry(a, b)));
        return newMatrix;
    }

    private Matrix deactivateOutput(Matrix matrix, ActivationFunction activationFunction) {
        Matrix newMatrix = new Matrix(matrix.getData());
        for (int a = 0; a < newMatrix.getRow(); a++)
            for (int b = 0; b < newMatrix.getColumn(); b++) {
                double derivative = Mapper.mapDeactivation(activationFunction, matrix.getEntry(a, b));
                newMatrix.setEntry(a, b, derivative);
            }
        return newMatrix;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}

//    private void printRealMatrix(Matrix matrix) {
//        System.out.println("Matrix(rows: " + matrix.getRow() + ",cols: " + matrix.getColumn() + ")");
//        StringBuilder stringBuilder = new StringBuilder();
//        stringBuilder.append("{\n");
//        for (int a = 0; a < matrix.getRow(); a++) {
//            stringBuilder.append("\t");
//            for (int b = 0; b < matrix.getColumn(); b++) {
//                stringBuilder.append(matrix.getEntry(a, b)).append(" ");
//            }
//            stringBuilder.append("\n");
//        }
//        stringBuilder.append("}\n");
//        System.out.println(stringBuilder);
//    }
//
//    public void printNetwork() {
//        ArrayList<Integer> schema = new ArrayList<>();
//        schema.add(this.numberOfInputNode);
//        for (var i : this.hiddenNodeSchema) {
//            schema.add(i.numberOfNodes());
//        }
//        System.out.println(schema);
//    }
//
//    public void saveModel(String fileName) throws IOException {
//        new ObjectOutputStream(new FileOutputStream(fileName)).writeObject(this);
//    }
//
//    public static JNeuralNetwork loadModel(String filePath) throws Exception {
//        return (JNeuralNetwork) new ObjectInputStream(new FileInputStream(filePath)).readObject();
//    }
