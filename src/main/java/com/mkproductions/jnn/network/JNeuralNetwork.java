package com.mkproductions.jnn.network;

import com.mkproductions.jnn.entity.*;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.Random;

public class JNeuralNetwork implements Serializable {
    private final int numberOfInputNode;
    private final Layer[] netWorkLayers;
    private final Matrix[] weightsMatrices;
    private final Matrix[] biasesMatrices;
    private double learningRate;
    private int networkAccuracy = 0;
    private final LossFunction lossFunction;

    public JNeuralNetwork(LossFunction lossFunction, int numberOfInputNode, Layer... netWorkLayers) {
        this.lossFunction = lossFunction;
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

    public JNeuralNetwork(JNeuralNetwork jNeuralNetwork) {
        this.numberOfInputNode = jNeuralNetwork.numberOfInputNode;
        this.netWorkLayers = jNeuralNetwork.netWorkLayers;
        this.learningRate = 0.01;
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
                outputMatrices[a] = Matrix.matrixMapping(Matrix.add(Matrix.matrixMultiplication(this.weightsMatrices[a], inputMatrix), this.biasesMatrices[a]), this.netWorkLayers[a].activationFunction().equation);
                continue;
            }
            outputMatrices[a] = Matrix.matrixMapping(Matrix.add(Matrix.matrixMultiplication(this.weightsMatrices[a], outputMatrices[a - 1]), this.biasesMatrices[a]), this.netWorkLayers[a].activationFunction().equation);
        }
        return outputMatrices[outputMatrices.length - 1].getColumn(0);
    }

    // TODO: Implement (Adam, SGD, XGBoost)
    // TODO: Loss function (Categorical cross entropy, Spars Categorical cross entropy)

    /**
     * Function to perform back-propagate and adjusts weights and biases as per the given inputs with targets.
     *
     * @param input        2D array of inputs to be learned by network.
     * @param targetOutput 2D array to train the network as per inputs index.
     */
    private void backPropagateSGD(double[] input, double[] targetOutput) {
        if (input.length != this.numberOfInputNode)
            throw new RuntimeException("Mismatch length of inputs to the network.");

        Matrix inputMatrix = Matrix.fromArray(input).transpose();
        Matrix[] outputMatrices = new Matrix[this.netWorkLayers.length];

        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                outputMatrices[a] = Matrix.matrixMultiplication(this.weightsMatrices[a], inputMatrix);
                outputMatrices[a] = Matrix.add(outputMatrices[a], this.biasesMatrices[a]);
                outputMatrices[a] = Matrix.matrixMapping(outputMatrices[a], this.netWorkLayers[a].activationFunction().equation);
                continue;
            }
            outputMatrices[a] = Matrix.matrixMultiplication(this.weightsMatrices[a], outputMatrices[a - 1]);
            outputMatrices[a] = Matrix.add(outputMatrices[a], this.biasesMatrices[a]);
            outputMatrices[a] = Matrix.matrixMapping(outputMatrices[a], this.netWorkLayers[a].activationFunction().equation);
        }

        Matrix outputMatrix = outputMatrices[outputMatrices.length - 1];
        Matrix targetMatrix = new Matrix(new double[][]{targetOutput}).transpose();

        // Output error matrix.
        Matrix errorMatrix = Matrix.scalarMultiply(Matrix.subtract(outputMatrix, targetMatrix), -1);

        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            // Calculating gradients
            Matrix gradientMatrix = Matrix.matrixMapping(outputMatrices[layerIndex], this.netWorkLayers[layerIndex].activationFunction().derivative);
            gradientMatrix = Matrix.elementWiseMultiply(gradientMatrix, errorMatrix);
            gradientMatrix = Matrix.scalarMultiply(gradientMatrix, -this.learningRate);

            // Calculating error
            errorMatrix = Matrix.matrixMultiplication(this.weightsMatrices[layerIndex].transpose(), errorMatrix);

            // Getting the inputs of the current layer.
            Matrix previousOutputMatrixTranspose = (layerIndex == 0) ? inputMatrix.transpose() : outputMatrices[layerIndex - 1].transpose();

            // Calculating the change in weights matrix for each layer of the network.
            Matrix deltaWeightsMatrix = Matrix.matrixMultiplication(gradientMatrix, previousOutputMatrixTranspose);

            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(deltaWeightsMatrix);
            this.biasesMatrices[layerIndex].subtract(gradientMatrix);
        }
    }

    private void backPropagateAdam(double[] inputs, double[] targetOutput) {
        Matrix inputMatrix = Matrix.fromArray(inputs).transpose();
        Matrix[] outputMatrices = new Matrix[this.netWorkLayers.length];

        for (int a = 0; a < this.weightsMatrices.length; a++) {
            if (a == 0) {
                outputMatrices[a] = Matrix.matrixMultiplication(this.weightsMatrices[a], inputMatrix);
                outputMatrices[a] = Matrix.add(outputMatrices[a], this.biasesMatrices[a]);
                outputMatrices[a] = Matrix.matrixMapping(outputMatrices[a], this.netWorkLayers[a].activationFunction().equation);
                continue;
            }
            outputMatrices[a] = Matrix.matrixMultiplication(this.weightsMatrices[a], outputMatrices[a - 1]);
            outputMatrices[a] = Matrix.add(outputMatrices[a], this.biasesMatrices[a]);
            outputMatrices[a] = Matrix.matrixMapping(outputMatrices[a], this.netWorkLayers[a].activationFunction().equation);
        }

        Matrix outputMatrix = outputMatrices[outputMatrices.length - 1];
        Matrix targetMatrix = new Matrix(new double[][]{targetOutput}).transpose();

        // Output error matrix.
        Matrix errorMatrix = Matrix.scalarMultiply(Matrix.subtract(outputMatrix, targetMatrix), -1);
    }

    /**
     * Function to train model for mass amount of training inputs and outputs with random samples.
     *
     * @param epochCount      number of back-propagation iterations performed by the model.
     * @param trainingInputs  2D array of inputs for training the model.
     * @param trainingOutputs 2D array of outputs for training the model.
     */
    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochCount) {
        if (trainingInputs[0].length != this.numberOfInputNode) throw new RuntimeException("Mismatch inputs size.");
        if (trainingOutputs[0].length != this.netWorkLayers[this.netWorkLayers.length - 1].numberOfNodes())
            throw new RuntimeException("Mismatch outputs size.");
        int progress;
        int lastProgress = 0;
        for (int epoch = 0; epoch < epochCount; epoch++) {
            // Random index for training random data from the training data set.
            progress = (epoch * 100) / epochCount;
            int randomIndex = new Random().nextInt(0, trainingInputs.length);
            double[] trainingInput = trainingInputs[randomIndex];
            double[] trainingOutput = trainingOutputs[randomIndex];
            backPropagateSGD(trainingInput, trainingOutput);
            if (progress != lastProgress) {
                lastProgress = progress;
                int a;
                System.out.print("Training progress: " + progress + " [");
                for (a = 0; a < progress + 1; a++) {
                    System.out.print("#");
                }
                for (int b = 0; b < 100 - a; b++) {
                    System.out.print(" ");
                }
                System.out.print("]");
                System.out.println();
            }
        }
    }

    public double calculateAccuracy(double[][] testingInputs, double[][] testingOutputs) {
        System.out.println("Calculating network Accuracy");
        double accuracy;
        Random random = new Random();
        double correctCount = 0;
        int progress;
        int lastProgress = 0;
        for (int i = 0; i < testingInputs.length; i++) {
            progress = (int) (((double) i / testingInputs.length) * 100);
            int randomIndex = random.nextInt(testingInputs.length);
            double[] prediction = this.processInputs(testingInputs[randomIndex]);
            double[] testingOutput = testingOutputs[randomIndex];
            int correctPredictionCount = 0;
            for (int a = 0; a < prediction.length; a++) {
                if (Math.abs(prediction[a]) == Math.abs(testingOutput[a])) correctPredictionCount++;
            }
            if (correctPredictionCount < 10) correctCount++;
            if (progress != lastProgress) {
                lastProgress = progress;
                int a;
                System.out.print("Testing progress: " + progress + " [");
                for (a = 0; a < progress + 1; a++) {
                    System.out.print("#");
                }
                for (int b = 0; b < 100 - a; b++) {
                    System.out.print(" ");
                }
                System.out.print("]");
                System.out.println();
            }
        }
        accuracy = correctCount / testingInputs.length;
        System.out.println("Thanks for waiting.");
        return accuracy;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

}