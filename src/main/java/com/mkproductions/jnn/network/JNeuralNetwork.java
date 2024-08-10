package com.mkproductions.jnn.network;

import com.mkproductions.jnn.entity.*;
import com.mkproductions.jnn.entity.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.entity.lossFunctions.ClassificationLossFunction;
import com.mkproductions.jnn.entity.optimzers.JNeuralNetworkOptimizer;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

public class JNeuralNetwork implements Serializable {
    private final int numberOfInputNode;
    private final Layer[] netWorkLayers;
    private final Matrix[] weightsMatrices;
    private final Matrix[] biasesMatrices;
    private double learningRate;
    private int networkAccuracy = 0;
    private final LossFunction lossFunction;
    private final JNeuralNetworkOptimizer jNeuralNetworkOptimizer;
    private final double sgdMomentumFactor;
    private boolean debugMode = false;

    public JNeuralNetwork(LossFunction lossFunction, JNeuralNetworkOptimizer jNeuralNetworkOptimizer, int numberOfInputNode, Layer... netWorkLayers) {
        this.lossFunction = lossFunction;
        this.jNeuralNetworkOptimizer = jNeuralNetworkOptimizer;
        // Storing the design of the Neural Network
        this.learningRate = 0.01;
        this.numberOfInputNode = numberOfInputNode;
        this.netWorkLayers = netWorkLayers;
        // Initializing the arrays
        this.weightsMatrices = new Matrix[netWorkLayers.length];
        this.biasesMatrices = new Matrix[netWorkLayers.length];
        this.sgdMomentumFactor = 0.5;
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
        this.networkAccuracy = jNeuralNetwork.networkAccuracy;
        this.lossFunction = jNeuralNetwork.lossFunction;
        this.jNeuralNetworkOptimizer = jNeuralNetwork.jNeuralNetworkOptimizer;
        this.sgdMomentumFactor = 1;
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
                outputMatrices[a] = Matrix.matrixMultiplication(this.weightsMatrices[a], inputMatrix);
                outputMatrices[a] = Matrix.add(outputMatrices[a], this.biasesMatrices[a]);
                outputMatrices[a] = JNeuralNetwork.getAppliedActivationFunctionMatrix(outputMatrices[a], this.netWorkLayers[a].activationFunction());
                continue;
            }
            outputMatrices[a] = Matrix.matrixMultiplication(this.weightsMatrices[a], outputMatrices[a - 1]);
            outputMatrices[a] = Matrix.add(outputMatrices[a], this.biasesMatrices[a]);
            outputMatrices[a] = JNeuralNetwork.getAppliedActivationFunctionMatrix(outputMatrices[a], this.netWorkLayers[a].activationFunction());
        }
        return outputMatrices[outputMatrices.length - 1].getColumn(0);
    }

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
        //        Matrix errorMatrix = Matrix.scalarMultiply(Matrix.subtract(outputMatrix, targetMatrix), -1);
        Matrix errorMatrix = this.lossFunction.getLossFunctionMatrix(outputMatrix, targetMatrix);
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

    private void backPropagateSGDWithMomentum(double[] input, double[] targetOutput) {
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
        Matrix errorMatrix = this.lossFunction.getLossFunctionMatrix(outputMatrix, targetMatrix);
        Matrix[] velocityWeightsMatrices = new Matrix[this.weightsMatrices.length];
        Matrix[] velocityBiasesMatrices = new Matrix[this.biasesMatrices.length];

        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            // Shaping the velocity matrices.
            velocityWeightsMatrices[layerIndex] = new Matrix(this.weightsMatrices[layerIndex].getRowCount(), this.weightsMatrices[layerIndex].getColumnCount());
            velocityBiasesMatrices[layerIndex] = new Matrix(this.biasesMatrices[layerIndex].getRowCount(), this.biasesMatrices[layerIndex].getColumnCount());

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

            // Calculating the velocities of the weights and biases
            velocityWeightsMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(velocityWeightsMatrices[layerIndex], this.sgdMomentumFactor), deltaWeightsMatrix);
            velocityBiasesMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(velocityBiasesMatrices[layerIndex], this.sgdMomentumFactor), gradientMatrix);

            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(velocityWeightsMatrices[layerIndex]);
            this.biasesMatrices[layerIndex].subtract(velocityBiasesMatrices[layerIndex]);
        }
    }

    /**
     * Function to train model for mass amount of training inputs and outputs with random samples.
     *
     * @param epochCount      number of back-propagation iterations performed by the model.
     * @param trainingInputs  2D array of inputs for training the model.
     * @param trainingOutputs 2D array of outputs for training the model.
     */
    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochCount) {
        int outputCount = 0;
        for (int i = 0; i < trainingOutputs.length; i++) {
            for (int j = 0; j < trainingInputs[i].length; j++) {
                outputCount++;
            }
        }
        if (outputCount == 1 && this.lossFunction instanceof ClassificationLossFunction) {
            throw new IllegalArgumentException("Loss function is not supported. Please proceed again to use a Linear regression loss function.");
        }
        if (trainingInputs[0].length != this.numberOfInputNode)
            throw new IllegalArgumentException("Mismatch inputs size.");
        if (trainingOutputs[0].length != this.netWorkLayers[this.netWorkLayers.length - 1].numberOfNodes())
            throw new IllegalArgumentException("Mismatch outputs size.");
        int progress;
        int lastProgress = 0;
        for (int epoch = 0; epoch < epochCount; epoch++) {
            // Random index for training random data from the training data set.
            progress = (epoch * 100) / epochCount;
            int randomIndex = new Random().nextInt(0, trainingInputs.length);
            double[] trainingInput = trainingInputs[randomIndex];
            double[] trainingOutput = trainingOutputs[randomIndex];
            switch (this.jNeuralNetworkOptimizer) {
                case SGD -> backPropagateSGD(trainingInput, trainingOutput);
                case SGD_MOMENTUM -> backPropagateSGDWithMomentum(trainingInput, trainingOutput);
            }
            if (this.debugMode && progress != lastProgress) {
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

    public static Matrix getAppliedActivationFunctionMatrix(Matrix matrix, ActivationFunction activationFunction) {
        Matrix result = new Matrix(matrix.getData());
//        if (activationFunction.name().equals("softmax")) {
//            result.matrixMapping((r, c, value) -> Math.exp(value));
//            for (int layerIndex = 0; layerIndex < result.getColumnCount(); layerIndex++) {
//                double[] column = result.getColumn(layerIndex);
//                Double sum = 0.0;
//                for (Double value : column) sum += value;
//                for (int rowIndex = 0; rowIndex < column.length; rowIndex++)
//                    result.setEntry(rowIndex, layerIndex, result.getEntry(rowIndex, layerIndex) / sum);
//            }
//        } else {
        result.matrixMapping(activationFunction.equation);
//        }
        return result;
    }

    public static Matrix getDactivatedActivationFunctionMatrix(Matrix activatedMatrix, ActivationFunction activationFunction) {
        Matrix result = new Matrix(activatedMatrix.getData());
//        if (!activationFunction.name().equals("softmax")) {
        result.matrixMapping(activationFunction.derivative);
//        } else {
//            // Softmax derivative implementation
//            result.matrixMapping((rowIndex, columnIndex, value) -> {
//                double sum = 0.0;
//                for (int j = 0; j < result.getColumnCount(); j++) {
//                    sum += Math.exp(activatedMatrix.getEntry(rowIndex, j));
//                }
//                double softmax_j = Math.exp(value) / sum;
//                if (columnIndex == rowIndex) {
//                    return softmax_j * (1 - softmax_j); // Diagonal element (y_i * (1 - y_i))
//                } else {
//                    return softmax_j * -Math.exp(activatedMatrix.getEntry(rowIndex, columnIndex)) / sum; // Off-diagonal element (y_i * -y_j)
//                }
//            });
//        }
        return result;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public boolean isDebugMode() {
        return debugMode;
    }

    public void setDebugMode(boolean debugMode) {
        this.debugMode = debugMode;
    }
}