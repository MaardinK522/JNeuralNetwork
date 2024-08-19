package com.mkproductions.jnn.network;

import com.mkproductions.jnn.entity.*;
import com.mkproductions.jnn.entity.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.entity.optimzers.JNeuralNetworkOptimizer;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.Random;

public class JNeuralNetwork implements Serializable {
    private int networkAccuracy = 0;
    private final int numberOfInputNode;

    private final Layer[] netWorkLayers;
    private final Matrix[] weightsMatrices;
    private final Matrix[] biasesMatrices;
    private final Matrix[] velocityWeightsMatrices;
    private final Matrix[] velocityBiasesMatrices;
    private final Matrix[] momentumWeightsMatrices;
    private final Matrix[] momentumBiasesMatrices;
    private final LossFunctionAble lossFunctionable;

    private final JNeuralNetworkOptimizer jNeuralNetworkOptimizer;

    private boolean debugMode = false;

    private double learningRate;
    private double momentumFactorBeta1;
    private double momentumFactorBeta2;
    private final double epsilonRMSProp = Math.pow(10, -8);

    public JNeuralNetwork(LossFunctionAble lossFunctionable, JNeuralNetworkOptimizer jNeuralNetworkOptimizer, int numberOfInputNode, Layer... netWorkLayers) {
        this.lossFunctionable = lossFunctionable;
        this.jNeuralNetworkOptimizer = jNeuralNetworkOptimizer;
        // Storing the design of the Neural Network
        this.learningRate = 0.01;
        this.numberOfInputNode = numberOfInputNode;
        this.netWorkLayers = netWorkLayers;
        // Initializing the arrays
        this.weightsMatrices = new Matrix[netWorkLayers.length];
        this.biasesMatrices = new Matrix[netWorkLayers.length];
        this.velocityWeightsMatrices = new Matrix[this.weightsMatrices.length];
        this.velocityBiasesMatrices = new Matrix[this.weightsMatrices.length];
        this.momentumWeightsMatrices = new Matrix[this.weightsMatrices.length];
        this.momentumBiasesMatrices = new Matrix[this.weightsMatrices.length];
        this.momentumFactorBeta1 = 0.9;
        // Assign weights and biases and velocity to matrix arrays
        for (int layerIndex = 0; layerIndex < this.weightsMatrices.length; layerIndex++) {
            if (layerIndex == 0) {
                this.weightsMatrices[layerIndex] = new Matrix(this.netWorkLayers[layerIndex].numberOfNodes(), this.numberOfInputNode);
            } else {
                this.weightsMatrices[layerIndex] = new Matrix(this.netWorkLayers[layerIndex].numberOfNodes(), this.netWorkLayers[layerIndex - 1].numberOfNodes());
            }
            this.biasesMatrices[layerIndex] = new Matrix(this.netWorkLayers[layerIndex].numberOfNodes(), 1);
            // Randomizing the weights and bias
            this.weightsMatrices[layerIndex].randomize();
            this.biasesMatrices[layerIndex].randomize();
            // Initializing the velocity matrices.
            if (this.jNeuralNetworkOptimizer != JNeuralNetworkOptimizer.SGD) {
                this.velocityWeightsMatrices[layerIndex] = new Matrix(this.weightsMatrices[layerIndex].getRowCount(), this.weightsMatrices[layerIndex].getColumnCount());
                this.velocityBiasesMatrices[layerIndex] = new Matrix(this.biasesMatrices[layerIndex].getRowCount(), this.biasesMatrices[layerIndex].getColumnCount());
            }
            if (this.jNeuralNetworkOptimizer == JNeuralNetworkOptimizer.ADAM) {
                this.momentumWeightsMatrices[layerIndex] = new Matrix(this.weightsMatrices[layerIndex].getRowCount(), this.weightsMatrices[layerIndex].getColumnCount());
                this.momentumBiasesMatrices[layerIndex] = new Matrix(this.biasesMatrices[layerIndex].getRowCount(), this.biasesMatrices[layerIndex].getColumnCount());
            }
        }
    }

    public JNeuralNetwork(JNeuralNetwork jNeuralNetwork) {
        // TODO: Copy all the parameters from the given neural network.
        this.numberOfInputNode = jNeuralNetwork.numberOfInputNode;
        this.netWorkLayers = jNeuralNetwork.netWorkLayers;
        this.learningRate = jNeuralNetwork.getLearningRate();
        // Initializing the arrays
        this.weightsMatrices = new Matrix[netWorkLayers.length];
        this.biasesMatrices = new Matrix[netWorkLayers.length];
        this.velocityWeightsMatrices = new Matrix[jNeuralNetwork.weightsMatrices.length];
        this.velocityBiasesMatrices = new Matrix[jNeuralNetwork.weightsMatrices.length];
        this.momentumWeightsMatrices = new Matrix[jNeuralNetwork.weightsMatrices.length];
        this.momentumBiasesMatrices = new Matrix[jNeuralNetwork.weightsMatrices.length];
        this.networkAccuracy = jNeuralNetwork.networkAccuracy;
        this.lossFunctionable = jNeuralNetwork.lossFunctionable;
        this.jNeuralNetworkOptimizer = jNeuralNetwork.jNeuralNetworkOptimizer;
        this.momentumFactorBeta1 = jNeuralNetwork.getMomentumFactorBeta1();
        this.momentumFactorBeta2 = jNeuralNetwork.getMomentumFactorBeta2();
        this.debugMode = jNeuralNetwork.debugMode;
        // Assign weights and biases and velocity to matrix arrays
        for (int layerIndex = 0; layerIndex < jNeuralNetwork.weightsMatrices.length; layerIndex++) {
            if (layerIndex == 0) {
                jNeuralNetwork.weightsMatrices[layerIndex] = new Matrix(jNeuralNetwork.netWorkLayers[layerIndex].numberOfNodes(), jNeuralNetwork.numberOfInputNode);
            } else {
                jNeuralNetwork.weightsMatrices[layerIndex] = new Matrix(jNeuralNetwork.netWorkLayers[layerIndex].numberOfNodes(), jNeuralNetwork.netWorkLayers[layerIndex - 1].numberOfNodes());
            }
            jNeuralNetwork.biasesMatrices[layerIndex] = new Matrix(jNeuralNetwork.netWorkLayers[layerIndex].numberOfNodes(), 1);
            // Randomizing the weights and bias
            jNeuralNetwork.weightsMatrices[layerIndex].randomize();
            jNeuralNetwork.biasesMatrices[layerIndex].randomize();
            // Initializing the velocity matrices.
            if (jNeuralNetwork.jNeuralNetworkOptimizer != JNeuralNetworkOptimizer.SGD && jNeuralNetwork.jNeuralNetworkOptimizer != JNeuralNetworkOptimizer.SGD_MOMENTUM) {
                jNeuralNetwork.velocityWeightsMatrices[layerIndex] = new Matrix(jNeuralNetwork.weightsMatrices[layerIndex].getRowCount(), jNeuralNetwork.weightsMatrices[layerIndex].getColumnCount());
                jNeuralNetwork.velocityBiasesMatrices[layerIndex] = new Matrix(jNeuralNetwork.biasesMatrices[layerIndex].getRowCount(), jNeuralNetwork.biasesMatrices[layerIndex].getColumnCount());
            }
            if (jNeuralNetwork.jNeuralNetworkOptimizer == JNeuralNetworkOptimizer.ADAM) {
                jNeuralNetwork.momentumWeightsMatrices[layerIndex] = new Matrix(jNeuralNetwork.weightsMatrices[layerIndex].getRowCount(), jNeuralNetwork.weightsMatrices[layerIndex].getColumnCount());
                jNeuralNetwork.momentumBiasesMatrices[layerIndex] = new Matrix(jNeuralNetwork.biasesMatrices[layerIndex].getRowCount(), jNeuralNetwork.biasesMatrices[layerIndex].getColumnCount());
            }
        }
    }

    private void generateIfInvalidParametersExceptionGenerates(int length) {
        if (length != this.numberOfInputNode) throw new RuntimeException("Mismatch length of inputs to the network.");
    }

    /**
     * Process inputs and produces outputs as per the network schema.
     *
     * @param inputs A double array to predict the output.
     * @return double array of output predicted by the network.
     */
    public double[] processInputs(double @NotNull [] inputs) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
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
        return outputMatrices[outputMatrices.length - 1].getColumn(0);
    }

    /**
     * Function to perform back-propagate and adjusts weights and biases as per the given inputs with targets.
     *
     * @param inputs       2D array of inputs to be learned by network.
     * @param targetOutput 2D array to train the network as per inputs index.
     */
    private void backPropagateSGD(double[] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);

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
//        Matrix errorMatrix = Matrix.subtract(targetMatrix, outputMatrix);
//        System.out.println("Error: " + errorMatrix.getEntry(0, 0));
        Matrix errorMatrix = this.lossFunctionable.getLossFunctionMatrix(outputMatrix, targetMatrix);
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

    private void backPropagateSGDWithMomentum(double[] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);

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
        Matrix errorMatrix = this.lossFunctionable.getLossFunctionMatrix(outputMatrix, targetMatrix);

//        // Initializing the velocity matrices.
//        for (int layerIndex = 0; layerIndex < this.weightsMatrices.length; layerIndex++) {
//            velocityWeightsMatrices[layerIndex] = new Matrix(this.weightsMatrices[layerIndex].getRowCount(), this.weightsMatrices[layerIndex].getColumnCount());
//            velocityBiasesMatrices[layerIndex] = new Matrix(this.biasesMatrices[layerIndex].getRowCount(), this.biasesMatrices[layerIndex].getColumnCount());
//        }
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

            // Calculating the velocities of the weights and biases
            this.velocityWeightsMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta1), Matrix.scalarMultiply(deltaWeightsMatrix, 1 - this.momentumFactorBeta1));
            this.velocityBiasesMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta1), Matrix.scalarMultiply(gradientMatrix, 1 - this.momentumFactorBeta1));

            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(this.velocityWeightsMatrices[layerIndex]);
            this.biasesMatrices[layerIndex].subtract(this.velocityBiasesMatrices[layerIndex]);
        }
    }

    private void backPropagateRMSProp(double[] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);

        final Matrix inputMatrix = Matrix.fromArray(inputs).transpose();
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

        final Matrix outputMatrix = outputMatrices[outputMatrices.length - 1];
        final Matrix targetMatrix = new Matrix(new double[][]{targetOutput}).transpose();

        // Output error matrix.
        Matrix errorMatrix = this.lossFunctionable.getLossFunctionMatrix(outputMatrix, targetMatrix);

        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            // Calculating gradients
            Matrix gradientMatrix = Matrix.matrixMapping(outputMatrices[layerIndex], this.netWorkLayers[layerIndex].activationFunction().derivative);
            gradientMatrix = Matrix.elementWiseMultiply(gradientMatrix, errorMatrix);
            gradientMatrix = Matrix.scalarMultiply(gradientMatrix, -this.learningRate);

            // Calculating error
            errorMatrix = Matrix.matrixMultiplication(this.weightsMatrices[layerIndex].transpose(), errorMatrix);

            // Getting the inputs of the current layer.
            Matrix previousOutputTransposeMatrix = (layerIndex == 0) ? inputMatrix.transpose() : outputMatrices[layerIndex - 1].transpose();

            // Calculating the change in weights matrix for each layer of the network.
            Matrix deltaWeightsMatrix = Matrix.matrixMultiplication(gradientMatrix, previousOutputTransposeMatrix);

            // Calculating the velocities of the weights and biases
            // weights
            Matrix squaredDeltaWeightsMatrix = Matrix.matrixMapping(deltaWeightsMatrix, (r, c, val) -> Math.pow(val, 2));
            this.velocityWeightsMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta1), Matrix.scalarMultiply(squaredDeltaWeightsMatrix, 1 - this.momentumFactorBeta1));
            Matrix newWeightsVelocityMatrix = this.velocityWeightsMatrices[layerIndex];
            Matrix rootWithVelocityWegihtsMatrix = Matrix.matrixMapping(deltaWeightsMatrix, (r, c, deltaWeight) -> this.learningRate * deltaWeight / Math.sqrt(newWeightsVelocityMatrix.getEntry(r, c) + this.epsilonRMSProp));

            // biases
            Matrix squaredGradientsMatrix = Matrix.matrixMapping(gradientMatrix, (r, c, gradient) -> Math.pow(gradient, 2));
            this.velocityBiasesMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta1), Matrix.scalarMultiply(squaredGradientsMatrix, 1 - this.momentumFactorBeta1));
            Matrix newBiasesVelocityMatrix = this.velocityBiasesMatrices[layerIndex];
            Matrix rootWithVelocityBiasesMatrix = Matrix.matrixMapping(gradientMatrix, (r, c, gradient) -> this.learningRate * gradient / Math.sqrt(newBiasesVelocityMatrix.getEntry(r, c) + this.epsilonRMSProp));

            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(rootWithVelocityWegihtsMatrix);
            this.biasesMatrices[layerIndex].subtract(rootWithVelocityBiasesMatrix);
        }
    }

    private void backPropagateAdaGrad(double[] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        final Matrix inputMatrix = Matrix.fromArray(inputs).transpose();
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

        final Matrix outputMatrix = outputMatrices[outputMatrices.length - 1];
        final Matrix targetMatrix = new Matrix(new double[][]{targetOutput}).transpose();

        // Output error matrix.
        Matrix errorMatrix = this.lossFunctionable.getLossFunctionMatrix(outputMatrix, targetMatrix);

        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            // Calculating gradients
            Matrix gradientMatrix = Matrix.matrixMapping(outputMatrices[layerIndex], this.netWorkLayers[layerIndex].activationFunction().derivative);
            gradientMatrix = Matrix.elementWiseMultiply(gradientMatrix, errorMatrix);
            gradientMatrix = Matrix.scalarMultiply(gradientMatrix, -this.learningRate);

            // Calculating error
            errorMatrix = Matrix.matrixMultiplication(this.weightsMatrices[layerIndex].transpose(), errorMatrix);

            // Getting the inputs of the current layer.
            Matrix previousOutputTransposeMatrix = (layerIndex == 0) ? inputMatrix.transpose() : outputMatrices[layerIndex - 1].transpose();

            // Calculating the change in weights matrix for each layer of the network.
            Matrix deltaWeightsMatrix = Matrix.matrixMultiplication(gradientMatrix, previousOutputTransposeMatrix);

            Matrix squaredDeltaWeightsMatrix = Matrix.matrixMapping(deltaWeightsMatrix, (r, c, deltaWeight) -> Math.pow(deltaWeight, 2));
            Matrix squaredGradientsMatrix = Matrix.matrixMapping(gradientMatrix, (r, c, gradient) -> Math.pow(gradient, 2));

            // Calculating the velocities of the weights and biases
            // weights
            this.velocityWeightsMatrices[layerIndex] = Matrix.add(this.velocityWeightsMatrices[layerIndex], squaredDeltaWeightsMatrix);
            this.velocityBiasesMatrices[layerIndex] = Matrix.add(this.velocityBiasesMatrices[layerIndex], squaredGradientsMatrix);
            Matrix currentWeightVelocity = this.velocityWeightsMatrices[layerIndex];
            Matrix currentBiasesVelocity = this.velocityBiasesMatrices[layerIndex];

            Matrix rootWithVelocityWegihtsMatrix = Matrix.matrixMapping(deltaWeightsMatrix, (row, column, deltaWeight) -> this.learningRate * deltaWeight / Math.sqrt(currentWeightVelocity.getEntry(row, column) + this.epsilonRMSProp));
            Matrix rootWithVelocityBiasesMatrix = Matrix.matrixMapping(gradientMatrix, (row, column, gradient) -> this.learningRate * gradient / Math.sqrt(currentBiasesVelocity.getEntry(row, column) + this.epsilonRMSProp));

            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(rootWithVelocityWegihtsMatrix);
            this.biasesMatrices[layerIndex].subtract(rootWithVelocityBiasesMatrix);
        }
    }

    private void backPropagateAdam(double[] inputs, double[] targetOutput, int iterationCount) {
        // TODO: Implement back propagation with Adam (Adaptive Momentum).
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        final Matrix inputMatrix = Matrix.fromArray(inputs).transpose();
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

        final Matrix outputMatrix = outputMatrices[outputMatrices.length - 1];
        final Matrix targetMatrix = new Matrix(new double[][]{targetOutput}).transpose();

        // Output error matrix.
        Matrix errorMatrix = this.lossFunctionable.getLossFunctionMatrix(outputMatrix, targetMatrix);

        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            // Calculating gradients
            Matrix gradientMatrix = Matrix.matrixMapping(outputMatrices[layerIndex], this.netWorkLayers[layerIndex].activationFunction().derivative);
            gradientMatrix = Matrix.elementWiseMultiply(gradientMatrix, errorMatrix);
            gradientMatrix = Matrix.scalarMultiply(gradientMatrix, -this.learningRate);

            // Calculating error
            errorMatrix = Matrix.matrixMultiplication(this.weightsMatrices[layerIndex].transpose(), errorMatrix);

            // Getting the inputs of the current layer.
            Matrix previousOutputTransposeMatrix = (layerIndex == 0) ? inputMatrix.transpose() : outputMatrices[layerIndex - 1].transpose();

            // Calculating the change in weights matrix for each layer of the network.
            Matrix deltaWeightsMatrix = Matrix.matrixMultiplication(gradientMatrix, previousOutputTransposeMatrix);

            // Calculating the velocity and momentum of the weights and biases.
            this.momentumWeightsMatrices[layerIndex] = Matrix.add(
                    Matrix.scalarMultiply(this.momentumWeightsMatrices[layerIndex], this.momentumFactorBeta1),
                    Matrix.scalarMultiply(deltaWeightsMatrix, 1 - this.momentumFactorBeta1)
            );
            this.momentumBiasesMatrices[layerIndex] = Matrix.add(
                    Matrix.scalarMultiply(this.momentumBiasesMatrices[layerIndex], this.momentumFactorBeta1),
                    Matrix.scalarMultiply(gradientMatrix, 1 - this.momentumFactorBeta1)
            );

            Matrix squaredDeltaWeightsMatrix = Matrix.matrixMapping(deltaWeightsMatrix, (r, c, deltaWeight) -> Math.pow(deltaWeight, 2));
            Matrix squaredGradientsMatrix = Matrix.matrixMapping(gradientMatrix, (r, c, gradient) -> Math.pow(gradient, 2));

            this.velocityWeightsMatrices[layerIndex] = Matrix.add(
                    Matrix.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta2),
                    Matrix.scalarMultiply(squaredDeltaWeightsMatrix, 1 - this.momentumFactorBeta2)
            );
            this.velocityBiasesMatrices[layerIndex] = Matrix.add(
                    Matrix.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta2),
                    Matrix.scalarMultiply(squaredGradientsMatrix, 1 - this.momentumFactorBeta2)
            );

            double beta1_t = Math.pow(this.momentumFactorBeta1, iterationCount + 1);
            double beta2_t = Math.pow(this.momentumFactorBeta2, iterationCount + 1);

            Matrix momentumWeightsHatMatrix = Matrix.matrixMapping(this.momentumWeightsMatrices[layerIndex], ((row, column, weightMomentum) -> weightMomentum / (1 - beta1_t)));
            Matrix momentumBiasesHatMatrix = Matrix.matrixMapping(this.momentumBiasesMatrices[layerIndex], ((row, column, biasMomentum) -> biasMomentum / (1 - beta1_t)));

            Matrix velocityWeightsHatMatrix = Matrix.matrixMapping(this.velocityWeightsMatrices[layerIndex], ((row, column, weightVelocity) -> weightVelocity / (1 - beta2_t)));
            Matrix velocityBiasesHatMatrix = Matrix.matrixMapping(this.velocityBiasesMatrices[layerIndex], ((row, column, biasVelocity) -> biasVelocity / (1 - beta2_t)));

            Matrix mometumRootWithVelocityWegihtsMatrix = Matrix.matrixMapping(momentumWeightsHatMatrix, (row, column, weightsMomentum) -> this.learningRate * weightsMomentum / (Math.sqrt(velocityWeightsHatMatrix.getEntry(row, column)) + this.epsilonRMSProp));
            Matrix mometumRootWithVelocityBiasesMatrix = Matrix.matrixMapping(momentumBiasesHatMatrix, (row, column, biasMomentum) -> this.learningRate * biasMomentum / (Math.sqrt(velocityBiasesHatMatrix.getEntry(row, column)) + this.epsilonRMSProp));

            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(mometumRootWithVelocityWegihtsMatrix);
            this.biasesMatrices[layerIndex].subtract(mometumRootWithVelocityBiasesMatrix);
        }
    }

    /**
     * Function to train model for mass amount of training inputs and outputs with random samples.
     *
     * @param epochs          number of back-propagation iterations performed by the model.
     * @param trainingInputs  2D array of inputs for training the model.
     * @param trainingOutputs 2D array of outputs for training the model.
     */
    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs) {
        if (trainingInputs[0].length != this.numberOfInputNode)
            throw new IllegalArgumentException("Mismatch inputs size.");
        if (trainingOutputs[0].length != this.netWorkLayers[this.netWorkLayers.length - 1].numberOfNodes())
            throw new IllegalArgumentException("Mismatch outputs size.");
        int progress;
        int lastProgress = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Random index for training random data from the training data set.
            progress = (epoch * 100) / epochs;
            int randomIndex = new Random().nextInt(0, trainingInputs.length);
            double[] trainingInput = trainingInputs[randomIndex];
            double[] trainingOutput = trainingOutputs[randomIndex];
            switch (this.jNeuralNetworkOptimizer) {
                case SGD -> backPropagateSGD(trainingInput, trainingOutput);
                case SGD_MOMENTUM -> backPropagateSGDWithMomentum(trainingInput, trainingOutput);
                case RMS_PROP -> backPropagateRMSProp(trainingInput, trainingOutput);
                case ADA_GARD -> backPropagateAdaGrad(trainingInput, trainingOutput);
                case ADAM -> backPropagateAdam(trainingInput, trainingOutput, epoch);
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
            boolean hasRegistered = false;
            for (int a = 0; a < prediction.length; a++) {
                if (Math.ceil(prediction[a]) == Math.ceil(testingOutput[a]) && !hasRegistered) {
                    hasRegistered = true;
                    correctPredictionCount++;
                }
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

    public void setMomentumFactorBeta2(double momentumFactorBeta2) {
        this.momentumFactorBeta2 = momentumFactorBeta2;
    }

    public double getEpsilonRMSProp() {
        return epsilonRMSProp;
    }

    public LossFunctionAble getLossFunctionable() {
        return lossFunctionable;
    }

    public JNeuralNetworkOptimizer getjNeuralNetworkOptimizer() {
        return jNeuralNetworkOptimizer;
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

    public double getMomentumFactorBeta1() {
        return this.momentumFactorBeta1;
    }

    public void setMomentumFactorBeta1(double momentumFactorBeta1) {
        this.momentumFactorBeta1 = momentumFactorBeta1;
    }

    public double getMomentumFactorBeta2() {
        return momentumFactorBeta2;
    }


}