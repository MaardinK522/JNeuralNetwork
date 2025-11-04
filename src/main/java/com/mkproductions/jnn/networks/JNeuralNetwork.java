package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.entity.LossFunctionAble;
import com.mkproductions.jnn.cpu.entity.Matrix;
import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNeuralNetworkOptimizer;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.security.SecureRandom;
import java.util.Random;
import java.util.stream.IntStream;

public class JNeuralNetwork implements Serializable {
    private final int numberOfInputNode;
    private final DenseLayer[] netWorkDenseLayers;
    private final Matrix[] weightsMatrices;
    private final Matrix[] biasesMatrices;
    private final Matrix[] velocityWeightsMatrices;
    private final Matrix[] velocityBiasesMatrices;
    private final Matrix[] momentumWeightsMatrices;
    private final Matrix[] momentumBiasesMatrices;
    private final Matrix[] outputMatrices;
    private LossFunctionAble lossFunctionable;

    private JNeuralNetworkOptimizer jNeuralNetworkOptimizer;

    private boolean debugMode = false;

    private double learningRate;
    private double momentumFactorBeta1;
    private double momentumFactorBeta2;
    private final double epsilonRMSProp = Math.pow(10, -8);
    private int epochCount = 0;
    private int adamSteps = 0;
    private final Random random = new SecureRandom();

    public JNeuralNetwork(LossFunctionAble lossFunctionable, JNeuralNetworkOptimizer jNeuralNetworkOptimizer, int numberOfInputNode, DenseLayer... netWorkDenseLayers) {
        this.lossFunctionable = lossFunctionable;
        this.jNeuralNetworkOptimizer = jNeuralNetworkOptimizer;
        // Storing the design of the Neural Network
        this.learningRate = 0.01;
        this.numberOfInputNode = numberOfInputNode;
        this.netWorkDenseLayers = netWorkDenseLayers;
        // Initializing the arrays
        this.weightsMatrices = new Matrix[netWorkDenseLayers.length];
        this.biasesMatrices = new Matrix[netWorkDenseLayers.length];
        this.outputMatrices = new Matrix[this.netWorkDenseLayers.length];
        this.velocityWeightsMatrices = new Matrix[this.weightsMatrices.length];
        this.velocityBiasesMatrices = new Matrix[this.weightsMatrices.length];
        this.momentumWeightsMatrices = new Matrix[this.weightsMatrices.length];
        this.momentumBiasesMatrices = new Matrix[this.weightsMatrices.length];
        this.momentumFactorBeta1 = 0.9;
        this.momentumFactorBeta2 = 0.999;
        // Assign weights and biases and velocity to matrix arrays
        for (int layerIndex = 0; layerIndex < this.weightsMatrices.length; layerIndex++) {
            if (layerIndex == 0) {
                this.weightsMatrices[layerIndex] = new Matrix(this.netWorkDenseLayers[layerIndex].numberOfNodes(), this.numberOfInputNode);
            } else {
                this.weightsMatrices[layerIndex] = new Matrix(this.netWorkDenseLayers[layerIndex].numberOfNodes(), this.netWorkDenseLayers[layerIndex - 1].numberOfNodes());
            }
            this.outputMatrices[layerIndex] = new Matrix(this.netWorkDenseLayers[layerIndex].numberOfNodes(), 1);
            this.biasesMatrices[layerIndex] = new Matrix(this.outputMatrices[layerIndex].getRowCount(), this.outputMatrices[layerIndex].getColumnCount());

            // Randomizing the weights and bias
            randomize(this.weightsMatrices[layerIndex]);
            randomize(this.biasesMatrices[layerIndex]);
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
        this.numberOfInputNode = jNeuralNetwork.numberOfInputNode;
        this.netWorkDenseLayers = jNeuralNetwork.netWorkDenseLayers;
        this.learningRate = jNeuralNetwork.getLearningRate();
        this.weightsMatrices = jNeuralNetwork.weightsMatrices;
        this.biasesMatrices = jNeuralNetwork.biasesMatrices;
        this.outputMatrices = jNeuralNetwork.outputMatrices;
        this.velocityWeightsMatrices = jNeuralNetwork.velocityWeightsMatrices;
        this.velocityBiasesMatrices = jNeuralNetwork.velocityBiasesMatrices;
        this.momentumWeightsMatrices = jNeuralNetwork.momentumWeightsMatrices;
        this.momentumBiasesMatrices = jNeuralNetwork.momentumBiasesMatrices;
        this.lossFunctionable = jNeuralNetwork.lossFunctionable;
        this.jNeuralNetworkOptimizer = jNeuralNetwork.jNeuralNetworkOptimizer;
        this.momentumFactorBeta1 = jNeuralNetwork.getMomentumFactorBeta1();
        this.momentumFactorBeta2 = jNeuralNetwork.getMomentumFactorBeta2();
        this.debugMode = jNeuralNetwork.debugMode;
        this.epochCount = jNeuralNetwork.epochCount;
        this.adamSteps = jNeuralNetwork.adamSteps;
        for (int layerIndex = 0; layerIndex < jNeuralNetwork.weightsMatrices.length; layerIndex++) {
            this.weightsMatrices[layerIndex] = jNeuralNetwork.weightsMatrices[layerIndex];
            this.biasesMatrices[layerIndex] = jNeuralNetwork.biasesMatrices[layerIndex];
            this.velocityWeightsMatrices[layerIndex] = jNeuralNetwork.velocityWeightsMatrices[layerIndex];
            this.velocityBiasesMatrices[layerIndex] = jNeuralNetwork.velocityBiasesMatrices[layerIndex];
            this.momentumWeightsMatrices[layerIndex] = jNeuralNetwork.momentumWeightsMatrices[layerIndex];
            this.momentumBiasesMatrices[layerIndex] = jNeuralNetwork.momentumBiasesMatrices[layerIndex];
        }
    }

    private void generateIfInvalidParametersExceptionGenerates(int length) {
        if (length != this.numberOfInputNode) {
            throw new IllegalArgumentException("Mismatch length of inputs to the network.");
        }
    }

    private Matrix[] forwardPropagation(double[] inputs) {
        final Matrix inputMatrix = Matrix.fromArray(inputs).transpose(); //new Matrix(new double[][] { inputs });
        for (int layerIndex = 0; layerIndex < this.weightsMatrices.length; layerIndex++) {
            if (layerIndex == 0) {
                this.outputMatrices[layerIndex] = Matrix.add(Matrix.matrixMultiplication(this.weightsMatrices[layerIndex], inputMatrix), this.biasesMatrices[layerIndex]);
            } else {
                this.outputMatrices[layerIndex] = Matrix.add(Matrix.matrixMultiplication(this.weightsMatrices[layerIndex], this.outputMatrices[layerIndex - 1]), this.biasesMatrices[layerIndex]);
            }
            this.outputMatrices[layerIndex] = getAppliedActivationFunctionMatrix(this.outputMatrices[layerIndex], this.netWorkDenseLayers[layerIndex].activationFunction());
        }
        return this.outputMatrices;
    }

    /**
     * Process inputs and produces outputs as per the network schema.
     *
     * @param inputs
     *         A double array to predict the output.
     * @return double array of output predicted by the network.
     */
    public double[] processInputs(double @NotNull [] inputs) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Matrix[] outputMatrices = this.forwardPropagation(inputs);
        double[] outputs = new double[this.netWorkDenseLayers[this.netWorkDenseLayers.length - 1].numberOfNodes()];
        for (int a = 0; a < outputs.length; a++) {
            outputs[a] = outputMatrices[this.netWorkDenseLayers.length - 1].getEntry(a, 0);
        }
        return outputs;
    }

    @Contract("_, _ -> new")
    private Matrix[][] backPropagation(double[] inputs, double[] targets) {
        Matrix[] biasesGradients = new Matrix[this.netWorkDenseLayers.length];
        Matrix[] weightsGradients = new Matrix[this.netWorkDenseLayers.length];
        Matrix targetMatrix = Matrix.fromArray(targets);
        Matrix[] outputMatrices = forwardPropagation(inputs);
        Matrix outputMatrix = outputMatrices[netWorkDenseLayers.length - 1];

        // Initialize error matrix (will be set in first iteration)
        Matrix errorMatrix = null;

        // Layer loop
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {

            // 2. Calculate data (dL / dz) per index.
            if (layerIndex == this.netWorkDenseLayers.length - 1 && lossFunctionable.equals(LossFunction.CATEGORICAL_CROSS_ENTROPY) && this.netWorkDenseLayers[layerIndex].activationFunction()
                    .equals(ActivationFunction.SOFTMAX)) {
                biasesGradients[layerIndex] = Matrix.subtract(outputMatrix, targetMatrix.transpose());

            } else {
                if (layerIndex == this.netWorkDenseLayers.length - 1) {
                    errorMatrix = this.lossFunctionable.getDerivativeMatrix(outputMatrix, targetMatrix.transpose());
                }
                Matrix activationDerivative = getDactivatedActivationFunctionMatrix(outputMatrices[layerIndex], this.netWorkDenseLayers[layerIndex].activationFunction());
                //  Handle Softmax Jacobian with matrix multiplication
                assert errorMatrix != null;
                if (this.netWorkDenseLayers[layerIndex].activationFunction().equals(ActivationFunction.SOFTMAX)) {
                    biasesGradients[layerIndex] = Matrix.matrixMultiplication(activationDerivative, errorMatrix);
                } else {
                    biasesGradients[layerIndex] = Matrix.elementWiseMultiply(activationDerivative, errorMatrix);
                }
            }

            // 3. Propagate error backwards (for the next layer in the loop)
            if (layerIndex > 0) {
                errorMatrix = Matrix.matrixMultiplication(this.weightsMatrices[layerIndex].transpose(), biasesGradients[layerIndex]);
            }

            // 4. Calculate the weight gradients
            Matrix previousOutputMatrix = (layerIndex == 0) ? Matrix.fromArray(inputs) : outputMatrices[layerIndex - 1].transpose();
            weightsGradients[layerIndex] = Matrix.matrixMultiplication(biasesGradients[layerIndex], previousOutputMatrix);
        }
        return new Matrix[][] { biasesGradients, weightsGradients };
    }

    /**
     * Function to perform back-propagate and adjusts weights and biases as per the given inputs with targets.
     *
     * @param inputs
     *         2D array of inputs to be learned by network.
     * @param targetOutput
     *         2D array to train the network as per inputs index.
     */
    private void backPropagateSGD(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        // Gradients are now RAW: gradients[0] = dL/db, gradients[1] = dL/dW
        Matrix[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix[] deltaBiasesMatrix = gradients[0];
        Matrix[] deltaWeightsMatrix = gradients[1];
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            // Apply learning rate.
            deltaBiasesMatrix[layerIndex] = Matrix.scalarMultiply(deltaBiasesMatrix[layerIndex], -this.learningRate);
            deltaWeightsMatrix[layerIndex] = Matrix.scalarMultiply(deltaWeightsMatrix[layerIndex], -this.learningRate);
            // Apply change: W_new = W_old - delta
            this.biasesMatrices[layerIndex].subtract(deltaBiasesMatrix[layerIndex]);
            this.weightsMatrices[layerIndex].subtract(deltaWeightsMatrix[layerIndex]);
        }
    }

    private void backPropagateSGDWithMomentum(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Matrix[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix[] deltaBiasesMatrix = gradients[0];
        Matrix[] deltaWeightsMatrix = gradients[1];
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            deltaBiasesMatrix[layerIndex] = Matrix.scalarMultiply(deltaBiasesMatrix[layerIndex], -this.learningRate);
            deltaWeightsMatrix[layerIndex] = Matrix.scalarMultiply(deltaWeightsMatrix[layerIndex], -this.learningRate);
            // Calculating the velocities of the weights and biases
            this.velocityWeightsMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta1),
                    Matrix.scalarMultiply(deltaWeightsMatrix[layerIndex], 1 - this.momentumFactorBeta1));
            this.velocityBiasesMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta1),
                    Matrix.scalarMultiply(deltaBiasesMatrix[layerIndex], 1 - this.momentumFactorBeta1));
            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(this.velocityWeightsMatrices[layerIndex]);
            this.biasesMatrices[layerIndex].subtract(this.velocityBiasesMatrices[layerIndex]);
        }
    }

    private void backPropagateRMSProp(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Matrix[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix[] deltaBiasesMatrix = gradients[0];
        Matrix[] deltaWeightsMatrix = gradients[1];
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            deltaBiasesMatrix[layerIndex] = Matrix.scalarMultiply(deltaBiasesMatrix[layerIndex], -this.learningRate);
            deltaWeightsMatrix[layerIndex] = Matrix.scalarMultiply(deltaWeightsMatrix[layerIndex], -this.learningRate);
            // Weights
            Matrix squaredDeltaWeightsMatrix = Matrix.matrixMapping(deltaWeightsMatrix[layerIndex], (_, _, val) -> Math.pow(val, 2));
            this.velocityWeightsMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta1),
                    Matrix.scalarMultiply(squaredDeltaWeightsMatrix, 1 - this.momentumFactorBeta1));
            Matrix newWeightsVelocityMatrix = this.velocityWeightsMatrices[layerIndex];
            Matrix rootWithVelocityWegihtsMatrix = Matrix.matrixMapping(deltaWeightsMatrix[layerIndex],
                    (r, c, deltaWeight) -> this.learningRate * deltaWeight / Math.sqrt(newWeightsVelocityMatrix.getEntry(r, c) + this.epsilonRMSProp));
            // Biases
            Matrix squaredGradientsMatrix = Matrix.matrixMapping(deltaBiasesMatrix[layerIndex], (_, _, gradient) -> Math.pow(gradient, 2));
            this.velocityBiasesMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta1),
                    Matrix.scalarMultiply(squaredGradientsMatrix, 1 - this.momentumFactorBeta1));
            Matrix newBiasesVelocityMatrix = this.velocityBiasesMatrices[layerIndex];
            Matrix rootWithVelocityBiasesMatrix = Matrix.matrixMapping(deltaBiasesMatrix[layerIndex],
                    (r, c, gradient) -> this.learningRate * gradient / Math.sqrt(newBiasesVelocityMatrix.getEntry(r, c) + this.epsilonRMSProp));
            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(rootWithVelocityWegihtsMatrix);
            this.biasesMatrices[layerIndex].subtract(rootWithVelocityBiasesMatrix);
        }
    }

    private void backPropagateAdaGrad(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Matrix[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix[] deltaBiasesMatrix = gradients[0];
        Matrix[] deltaWeightsMatrix = gradients[1];
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            deltaBiasesMatrix[layerIndex] = Matrix.scalarMultiply(deltaBiasesMatrix[layerIndex], -this.learningRate);
            deltaWeightsMatrix[layerIndex] = Matrix.scalarMultiply(deltaWeightsMatrix[layerIndex], -this.learningRate);
            // Calculating square of delta weights and biases.
            Matrix squaredDeltaWeightsMatrix = Matrix.matrixMapping(deltaWeightsMatrix[layerIndex], (_, _, deltaWeight) -> Math.pow(deltaWeight, 2));
            Matrix squaredGradientsMatrix = Matrix.matrixMapping(deltaBiasesMatrix[layerIndex], (_, _, gradient) -> Math.pow(gradient, 2));
            // Weights
            this.velocityWeightsMatrices[layerIndex] = Matrix.add(this.velocityWeightsMatrices[layerIndex], squaredDeltaWeightsMatrix);
            Matrix currentWeightVelocity = this.velocityWeightsMatrices[layerIndex];
            Matrix rootWithVelocityWegihtsMatrix = Matrix.matrixMapping(deltaWeightsMatrix[layerIndex],
                    (row, column, deltaWeight) -> this.learningRate * deltaWeight / Math.sqrt(currentWeightVelocity.getEntry(row, column) + this.epsilonRMSProp));
            this.weightsMatrices[layerIndex].subtract(rootWithVelocityWegihtsMatrix);
            // Biases
            this.velocityBiasesMatrices[layerIndex] = Matrix.add(this.velocityBiasesMatrices[layerIndex], squaredGradientsMatrix);
            Matrix currentBiasesVelocity = this.velocityBiasesMatrices[layerIndex];
            Matrix rootWithVelocityBiasesMatrix = Matrix.matrixMapping(deltaBiasesMatrix[layerIndex],
                    (row, column, gradient) -> this.learningRate * gradient / Math.sqrt(currentBiasesVelocity.getEntry(row, column) + this.epsilonRMSProp));
            this.biasesMatrices[layerIndex].subtract(rootWithVelocityBiasesMatrix);
        }
    }

    private void backPropagateAdam(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        this.adamSteps++;

        Matrix[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix[] deltaBiasesMatrix = gradients[0];
        Matrix[] deltaWeightsMatrix = gradients[1];

        final double beta_1_t = Math.pow(this.momentumFactorBeta1, this.adamSteps);
        final double beta_2_t = Math.pow(this.momentumFactorBeta2, this.adamSteps);

        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            deltaBiasesMatrix[layerIndex] = Matrix.scalarMultiply(deltaBiasesMatrix[layerIndex], -this.learningRate);
            deltaWeightsMatrix[layerIndex] = Matrix.scalarMultiply(deltaWeightsMatrix[layerIndex], -this.learningRate);

            this.momentumWeightsMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.momentumWeightsMatrices[layerIndex], this.momentumFactorBeta1),
                    Matrix.scalarMultiply(deltaWeightsMatrix[layerIndex], 1 - this.momentumFactorBeta1));
            this.momentumBiasesMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.momentumBiasesMatrices[layerIndex], this.momentumFactorBeta1),
                    Matrix.scalarMultiply(deltaBiasesMatrix[layerIndex], 1 - this.momentumFactorBeta1));

            Matrix squaredGradientsWeights = Matrix.matrixMapping(deltaWeightsMatrix[layerIndex], (_, _, grad) -> Math.pow(grad, 2));
            Matrix squaredGradientsBiases = Matrix.matrixMapping(deltaBiasesMatrix[layerIndex], (_, _, grad) -> Math.pow(grad, 2));

            this.velocityWeightsMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta2),
                    Matrix.scalarMultiply(squaredGradientsWeights, 1 - this.momentumFactorBeta2));
            this.velocityBiasesMatrices[layerIndex] = Matrix.add(Matrix.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta2),
                    Matrix.scalarMultiply(squaredGradientsBiases, 1 - this.momentumFactorBeta2));

            Matrix momentumWeightsHat = Matrix.matrixMapping(this.momentumWeightsMatrices[layerIndex], (_, _, m) -> m / (1 - beta_1_t));
            Matrix momentumBiasesHat = Matrix.matrixMapping(this.momentumBiasesMatrices[layerIndex], (_, _, m) -> m / (1 - beta_1_t));

            Matrix velocityWeightsHat = Matrix.matrixMapping(this.velocityWeightsMatrices[layerIndex], (_, _, v) -> v / (1 - beta_2_t));
            Matrix velocityBiasesHat = Matrix.matrixMapping(this.velocityBiasesMatrices[layerIndex], (_, _, v) -> v / (1 - beta_2_t));

            Matrix updateWeights = Matrix.matrixMapping(momentumWeightsHat, (row, col, m) -> this.learningRate * m / (Math.sqrt(velocityWeightsHat.getEntry(row, col)) + this.epsilonRMSProp));
            Matrix updateBiases = Matrix.matrixMapping(momentumBiasesHat, (row, col, m) -> this.learningRate * m / (Math.sqrt(velocityBiasesHat.getEntry(row, col)) + this.epsilonRMSProp));

            this.weightsMatrices[layerIndex].subtract(updateWeights);
            this.biasesMatrices[layerIndex].subtract(updateBiases);
        }
    }

    /**
     * Function to train model for mass amount of training inputs and outputs with random samples.
     *
     * @param epochs
     *         number of back-propagation iterations performed by the model.
     * @param trainingInputs
     *         2D array of inputs for training the model.
     * @param trainingOutputs
     *         2D array of outputs for training the model.
     */
    public void train(double[] @NotNull [] trainingInputs, double[][] trainingOutputs, int epochs) {
        if (trainingInputs[0].length != this.numberOfInputNode) {
            throw new IllegalArgumentException("Mismatch inputs size.");
        }
        if (trainingOutputs[0].length != this.netWorkDenseLayers[this.netWorkDenseLayers.length - 1].numberOfNodes()) {
            throw new IllegalArgumentException("Mismatch outputs size.");
        }
        int progress;
        int lastProgress = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Random index for training random data from the training data set.
            progress = (epoch * 100) / epochs;
            int randomIndex = this.random.nextInt(0, trainingInputs.length);
            //            System.out.println("Random index: " + randomIndex);
            double[] trainingInput = trainingInputs[randomIndex];
            double[] trainingOutput = trainingOutputs[randomIndex];
            //            System.out.println("Training inputs: " + Arrays.toString(trainingInput));
            //            System.out.println("Training outputs: " + Arrays.toString(trainingOutput));
            switch (this.jNeuralNetworkOptimizer) {
                case SGD -> backPropagateSGD(trainingInput, trainingOutput);
                case SGD_MOMENTUM -> backPropagateSGDWithMomentum(trainingInput, trainingOutput);
                case RMS_PROP -> backPropagateRMSProp(trainingInput, trainingOutput);
                case ADA_GARD -> backPropagateAdaGrad(trainingInput, trainingOutput);
                case ADAM -> backPropagateAdam(trainingInput, trainingOutput);
            }
            if (this.debugMode && progress != lastProgress) {
                lastProgress = progress;
                int filled = progress / 2;
                int empty = 50 - filled;
                String filledBar = "#".repeat(filled);
                String emptyBar = " ".repeat(empty);
                System.out.printf("\rTraining progress: %d%% [%s%s]", progress, filledBar, emptyBar);
                if (progress == 100) {
                    System.out.println();
                }
            }
            epochCount++;
        }
    }

    private int argmax(double @NotNull [] array) {
        if (array.length == 0) {
            return -1;
        }
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public double calculateAccuracy(double[][] testingInputs, double[][] testingOutputs) {
        if (testingInputs == null || testingInputs.length == 0) {
            return 0.0;
        }
        System.out.println();
        double correctCount = 0;
        int dataSetSize = testingInputs.length;

        // Tracking for console output
        int lastProgress = -1;

        for (int i = 0; i < dataSetSize; i++) {
            // --- 1. Get Prediction and True Label ---
            double[] prediction = this.processInputs(testingInputs[i]);
            double[] testingOutput = testingOutputs[i];

            //            boolean isSame = true;
            //            for (int a = 0; a < prediction.length; a++) {
            //                // Check if the predicted class index matches the true class index
            //                if (prediction[a] != testingOutput[a]) {
            //                    isSame = false;
            //                    break;
            //                }
            //            }
            //            if (isSame) {
            //                correctCount++;
            //            }

            // --- 2. Argmax/Classification Check (Correct Logic) ---
            // Find the index of the highest activation in the prediction array
            int predictedClass = argmax(prediction);

            // Find the index of the true class (index of 1.0 in testingOutput)
            int trueClass = argmax(testingOutput);

            // Check if the predicted class index matches the true class index
            if (predictedClass == trueClass) {
                correctCount++;
            }

            // --- 3. Progress Bar (Optional, improved) ---
            int progress = (int) (((double) (i + 1) / dataSetSize) * 100);
            if (progress != lastProgress) {
                lastProgress = progress;
                // Use '\r' to overwrite the current line for a non-scrolling progress bar
                System.out.printf("\rTesting progress: %d%% [%s%s]", progress, "#".repeat(progress / 2), // Example: print a '#' for every 2%
                        " ".repeat(50 - (progress / 2)) // Fill remaining space
                );
                // Ensure the line is only ended when 100% is reached
                if (progress == 100) {
                    System.out.println(); // Newline after 100%
                }
            }
        }

        double accuracy = (correctCount / dataSetSize) * 100.0;
        // In case the progress bar didn't finish with a newline
        if (lastProgress != 100) {
            System.out.println();
        }
        System.out.printf("Accuracy: %.2f%% (%d/%d correct)%n", accuracy, (int) correctCount, dataSetSize);

        return accuracy;
    }

    public static Matrix getAppliedActivationFunctionMatrix(Matrix matrix, @NotNull ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (matrix.getColumnCount() != 1) {
                throw new IllegalArgumentException("Unable to apply softmax due to more column existing in the given matrix.");
            }
            //            System.out.println("Printing non applied matrix: ");
            //            matrix.printMatrix();
            Matrix eRasiedMatrix = Matrix.matrixMapping(matrix, (a, b, value) -> Math.exp(matrix.getEntry(a, b)));
            //            System.out.println("Printing powered by e matrix: ");
            //            eRasiedMatrix.printMatrix();
            double sum = IntStream.range(0, eRasiedMatrix.getRowCount()).mapToDouble(a -> eRasiedMatrix.getEntry(a, 0)).sum();
            Matrix result = Matrix.matrixMapping(eRasiedMatrix, (a, b, value) -> eRasiedMatrix.getEntry(a, b) / sum);
            //            System.out.println("After getting averaged by the sum: ");
            //            result.printMatrix();
            return result;
        }
        return Matrix.matrixMapping(matrix, activationFunction.equation);
    }

    public static Matrix getDactivatedActivationFunctionMatrix(Matrix activatedMatrix, @NotNull ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (activatedMatrix.getRowCount() != 1 && activatedMatrix.getColumnCount() != 1) {
                throw new IllegalArgumentException("Softmax derivative expects a single vector output.");
            }
            int n = Math.max(activatedMatrix.getRowCount(), activatedMatrix.getColumnCount());
            double[] values = Matrix.toFlatArray(activatedMatrix);
            return Matrix.matrixMapping(new Matrix(n, n), (a, b, value) -> {
                if (a == b) {
                    return values[a] * (1 - values[a]);
                } else {
                    return -values[a] * values[b];
                }
            });
        }
        return Matrix.matrixMapping(activatedMatrix, activationFunction.derivative);
    }

    public double getEpsilonRMSProp() {
        return epsilonRMSProp;
    }

    public double getMomentumFactorBeta2() {
        return momentumFactorBeta2;
    }

    public JNeuralNetwork setMomentumFactorBeta2(double momentumFactorBeta2) {
        this.momentumFactorBeta2 = momentumFactorBeta2;
        return this;
    }

    public double getMomentumFactorBeta1() {
        return momentumFactorBeta1;
    }

    public JNeuralNetwork setMomentumFactorBeta1(double momentumFactorBeta1) {
        this.momentumFactorBeta1 = momentumFactorBeta1;
        return this;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public JNeuralNetwork setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public boolean isDebugMode() {
        return debugMode;
    }

    public JNeuralNetwork setDebugMode(boolean debugMode) {
        this.debugMode = debugMode;
        return this;
    }

    public JNeuralNetworkOptimizer getjNeuralNetworkOptimizer() {
        return jNeuralNetworkOptimizer;
    }

    public JNeuralNetwork setLossFunctionAble(LossFunctionAble lossFunctionAble) {
        this.lossFunctionable = lossFunctionAble;
        return this;
    }

    public LossFunctionAble getLossFunctionable() {
        return lossFunctionable;
    }

    public int getNumberOfInputNode() {
        return numberOfInputNode;
    }

    public JNeuralNetwork setJNeuralNetworkOptimizer(JNeuralNetworkOptimizer jNeuralNetworkOptimizer) {
        this.jNeuralNetworkOptimizer = jNeuralNetworkOptimizer;
        return this;
    }

    private void randomize(@NotNull Matrix matrix2DDouble) {
        double stdDev = Math.sqrt(2.0 / matrix2DDouble.getColumnCount());
        for (int a = 0; a < matrix2DDouble.getRowCount(); a++) {
            for (int i = 0; i < matrix2DDouble.getColumnCount(); i++) {
                double u1 = random.nextDouble();
                double u2 = random.nextDouble();
                double gaussian = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                matrix2DDouble.setEntry(a, i, gaussian * stdDev);
            }
        }
    }
}