package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.entity.LossFunctionAble;
import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.security.SecureRandom;
import java.util.Random;
import java.util.stream.IntStream;

public class JNeuralNetwork implements Serializable {
    private final int numberOfInputNode;
    private final DenseLayer[] netWorkDenseLayers;
    private final Tensor[] weightsMatrices;
    private final Tensor[] biasesMatrices;
    private final Tensor[] velocityWeightsMatrices;
    private final Tensor[] velocityBiasesMatrices;
    private final Tensor[] momentumWeightsMatrices;
    private final Tensor[] momentumBiasesMatrices;
    private final Tensor[] outputMatrices;
    private LossFunctionAble lossFunctionable;

    private JNetworkOptimizer jNetworkOptimizer;

    private boolean debugMode = false;

    private double learningRate;
    private double momentumFactorBeta1;
    private double momentumFactorBeta2;
    private final double epsilon = Math.pow(10, -8);
    private int epochCount = 0;
    private int adamSteps = 0;
    private final Random random = new SecureRandom();

    public JNeuralNetwork(LossFunctionAble lossFunctionable, JNetworkOptimizer jNetworkOptimizer, int numberOfInputNode, DenseLayer... netWorkDenseLayers) {
        this.lossFunctionable = lossFunctionable;
        this.jNetworkOptimizer = jNetworkOptimizer;
        // Storing the design of the Neural Network
        this.learningRate = 0.01;
        this.numberOfInputNode = numberOfInputNode;
        this.netWorkDenseLayers = netWorkDenseLayers;
        // Initializing the arrays
        this.weightsMatrices = new Tensor[netWorkDenseLayers.length];
        this.biasesMatrices = new Tensor[netWorkDenseLayers.length];
        this.outputMatrices = new Tensor[this.netWorkDenseLayers.length];
        this.velocityWeightsMatrices = new Tensor[this.weightsMatrices.length];
        this.velocityBiasesMatrices = new Tensor[this.weightsMatrices.length];
        this.momentumWeightsMatrices = new Tensor[this.weightsMatrices.length];
        this.momentumBiasesMatrices = new Tensor[this.weightsMatrices.length];
        this.momentumFactorBeta1 = 0.9;
        this.momentumFactorBeta2 = 0.999;
        // Assign weights and biases and velocity to Tensor arrays
        for (int layerIndex = 0; layerIndex < this.weightsMatrices.length; layerIndex++) {
            this.weightsMatrices[layerIndex] = new Tensor(this.netWorkDenseLayers[layerIndex].getNumberOfNodes(), layerIndex == 0 ? this.numberOfInputNode : this.netWorkDenseLayers[layerIndex - 1].getNumberOfNodes());
            this.outputMatrices[layerIndex] = new Tensor(this.netWorkDenseLayers[layerIndex].getNumberOfNodes(), 1);
            this.biasesMatrices[layerIndex] = new Tensor(this.outputMatrices[layerIndex].getShape()[0], this.outputMatrices[layerIndex].getShape()[1]);
            // Randomizing the weights and bias
            randomize(this.weightsMatrices[layerIndex]);
            randomize(this.biasesMatrices[layerIndex]);
            // Initializing the velocity matrices.
            this.velocityWeightsMatrices[layerIndex] = new Tensor(this.weightsMatrices[layerIndex].getShape()[0], this.weightsMatrices[layerIndex].getShape()[1]);
            this.velocityBiasesMatrices[layerIndex] = new Tensor(this.biasesMatrices[layerIndex].getShape()[0], this.biasesMatrices[layerIndex].getShape()[1]);
            this.momentumWeightsMatrices[layerIndex] = new Tensor(this.weightsMatrices[layerIndex].getShape()[0], this.weightsMatrices[layerIndex].getShape()[1]);
            this.momentumBiasesMatrices[layerIndex] = new Tensor(this.biasesMatrices[layerIndex].getShape()[0], this.biasesMatrices[layerIndex].getShape()[1]);
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
        this.jNetworkOptimizer = jNeuralNetwork.jNetworkOptimizer;
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

    private void forwardPropagation(double[] inputs) {
        Tensor inputTensor = new Tensor(inputs, inputs.length, 1); //new Tensor(new double[][] { inputs });
        for (int layerIndex = 0; layerIndex < this.weightsMatrices.length; layerIndex++) {
            if (layerIndex == 0) {
                this.outputMatrices[layerIndex] = Tensor.add(Tensor.matrixMultiplication(this.weightsMatrices[layerIndex], inputTensor), this.biasesMatrices[layerIndex]);
            } else {
                this.outputMatrices[layerIndex] = Tensor.add(Tensor.matrixMultiplication(this.weightsMatrices[layerIndex], this.outputMatrices[layerIndex - 1]), this.biasesMatrices[layerIndex]);
            }
            this.outputMatrices[layerIndex] = getAppliedActivationFunctionTensor(this.outputMatrices[layerIndex], this.netWorkDenseLayers[layerIndex].getActivationFunction());
        }
    }

    /**
     * Process inputs and produces outputs as per the network schema.
     *
     * @param inputs A double array to predict the output.
     * @return double array of output predicted by the network.
     */
    public double[] processInputs(double[] inputs) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        this.forwardPropagation(inputs);
        double[] outputs = new double[this.netWorkDenseLayers[this.netWorkDenseLayers.length - 1].getNumberOfNodes()];
        for (int a = 0; a < outputs.length; a++) {
            outputs[a] = outputMatrices[this.netWorkDenseLayers.length - 1].getEntry(a, 0);
        }
        return outputs;
    }

    private Tensor[][] backPropagation(double[] inputs, double[] targets) {
        Tensor[] biasesGradients = new Tensor[this.netWorkDenseLayers.length];
        Tensor[] weightsGradients = new Tensor[this.netWorkDenseLayers.length];
        Tensor targetTensor = new Tensor(targets, 1, targets.length);
        forwardPropagation(inputs);
        Tensor outputTensor = outputMatrices[netWorkDenseLayers.length - 1];

        // Initialize error Tensor (will be set in the first iteration)
        Tensor gradients = null;

        // Layer loop
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {

            // 2. Calculate data (dL / dz) per index.
            if (layerIndex == this.netWorkDenseLayers.length - 1 && lossFunctionable.equals(LossFunction.CATEGORICAL_CROSS_ENTROPY) && this.netWorkDenseLayers[layerIndex].getActivationFunction()
                    .equals(ActivationFunction.SOFTMAX)) {
                biasesGradients[layerIndex] = Tensor.subtract(outputTensor, Tensor.transpose(targetTensor));
            } else {
                if (layerIndex == this.netWorkDenseLayers.length - 1) {
                    gradients = this.lossFunctionable.getDerivativeTensor(outputTensor, Tensor.transpose(targetTensor));
                }
                Tensor activationDerivative = getDeactivatedActivationFunctionTensor(outputMatrices[layerIndex], this.netWorkDenseLayers[layerIndex].getActivationFunction());
                //  Handle Softmax Jacobian with Tensor multiplication
                assert gradients != null;
                if (this.netWorkDenseLayers[layerIndex].getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
                    biasesGradients[layerIndex] = Tensor.matrixMultiplication(activationDerivative, gradients);
                } else {
                    biasesGradients[layerIndex] = Tensor.elementWiseMultiplication(activationDerivative, gradients);
                }
            }

            // 3. Propagate error backwards (for the next layer in the loop)
            if (layerIndex > 0) {
                gradients = Tensor.matrixMultiplication(Tensor.transpose(this.weightsMatrices[layerIndex]), biasesGradients[layerIndex]);
            }

            // 4. Calculate the weight gradients
            Tensor previousOutputTensor = (layerIndex == 0) ? new Tensor(inputs, 1, inputs.length) : Tensor.transpose(outputMatrices[layerIndex - 1]);
            weightsGradients[layerIndex] = Tensor.matrixMultiplication(biasesGradients[layerIndex], previousOutputTensor);
        }
        return new Tensor[][]{biasesGradients, weightsGradients};
    }

    /**
     * Function to perform back-propagate and adjusts weights and biases as per the given inputs with targets.
     *
     * @param inputs       2D array of inputs to be learned by network.
     * @param targetOutput 2D array to train the network as per inputs index.
     */
    private void backPropagateSGD(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        // Gradients are now RAW: gradients[0] = dL/db, gradients[1] = dL/dW
        Tensor[][] gradients = this.backPropagation(inputs, targetOutput);
        Tensor[] deltaBiasesTensor = gradients[0];
        Tensor[] deltaWeightsTensor = gradients[1];
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            // Apply learning rate.
            deltaBiasesTensor[layerIndex] = Tensor.scalarMultiply(deltaBiasesTensor[layerIndex], -this.learningRate);
            deltaWeightsTensor[layerIndex] = Tensor.scalarMultiply(deltaWeightsTensor[layerIndex], -this.learningRate);
            // Apply change: W_new = W_old - delta
            this.biasesMatrices[layerIndex].subtract(deltaBiasesTensor[layerIndex]);
            this.weightsMatrices[layerIndex].subtract(deltaWeightsTensor[layerIndex]);
        }
    }

    private void backPropagateSGDWithMomentum(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Tensor[][] gradients = this.backPropagation(inputs, targetOutput);
        Tensor[] deltaBiasesTensor = gradients[0];
        Tensor[] deltaWeightsTensor = gradients[1];
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            deltaBiasesTensor[layerIndex] = Tensor.scalarMultiply(deltaBiasesTensor[layerIndex], -this.learningRate);
            deltaWeightsTensor[layerIndex] = Tensor.scalarMultiply(deltaWeightsTensor[layerIndex], -this.learningRate);
            // Calculating the velocities of the weights and biases
            this.velocityWeightsMatrices[layerIndex] = Tensor.add(Tensor.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta1),
                    Tensor.scalarMultiply(deltaWeightsTensor[layerIndex], 1 - this.momentumFactorBeta1));
            this.velocityBiasesMatrices[layerIndex] = Tensor.add(Tensor.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta1),
                    Tensor.scalarMultiply(deltaBiasesTensor[layerIndex], 1 - this.momentumFactorBeta1));
            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(this.velocityWeightsMatrices[layerIndex]);
            this.biasesMatrices[layerIndex].subtract(this.velocityBiasesMatrices[layerIndex]);
        }
    }

    private void backPropagateRMSProp(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Tensor[][] gradients = this.backPropagation(inputs, targetOutput);
        Tensor[] deltaBiasesTensor = gradients[0];
        Tensor[] deltaWeightsTensor = gradients[1];
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            deltaBiasesTensor[layerIndex] = Tensor.scalarMultiply(deltaBiasesTensor[layerIndex], -this.learningRate);
            deltaWeightsTensor[layerIndex] = Tensor.scalarMultiply(deltaWeightsTensor[layerIndex], -this.learningRate);
            // Weights
            Tensor squaredDeltaWeightsTensor = Tensor.tensorMapping(deltaWeightsTensor[layerIndex], (_, val) -> Math.pow(val, 2));

            this.velocityWeightsMatrices[layerIndex] = Tensor.add(Tensor.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta1),
                    Tensor.scalarMultiply(squaredDeltaWeightsTensor, 1 - this.momentumFactorBeta1));

            Tensor newWeightsVelocityTensor = this.velocityWeightsMatrices[layerIndex];
            Tensor rootWithVelocityWeightsTensor = Tensor.tensorMapping(deltaWeightsTensor[layerIndex],
                    ((flatIndex, value) -> this.learningRate * value / Math.sqrt(newWeightsVelocityTensor.getData()[flatIndex] + this.epsilon)));

            // Biases
            Tensor squaredGradientsTensor = Tensor.tensorMapping(deltaBiasesTensor[layerIndex], (_, gradient) -> Math.pow(gradient, 2));
            this.velocityBiasesMatrices[layerIndex] = Tensor.add(Tensor.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta1),
                    Tensor.scalarMultiply(squaredGradientsTensor, 1 - this.momentumFactorBeta1));

            Tensor newBiasesVelocityTensor = this.velocityBiasesMatrices[layerIndex];
            Tensor rootWithVelocityBiasesTensor = Tensor.tensorMapping(deltaBiasesTensor[layerIndex],
                    (flatIndex, value) -> this.learningRate * value / Math.sqrt(newBiasesVelocityTensor.getData()[flatIndex] + this.epsilon));

            //            Tensor rootWithVelocityBiasesTensor = Tensor.TensorMapping(deltaBiasesTensor[layerIndex],
            //                    (r, c, gradient) -> this.learningRate * gradient / Math.sqrt(newBiasesVelocityTensor.getEntry(r, c) + this.epsilonRMSProp));
            // Applying the change of weights in the current weights of the network.
            this.weightsMatrices[layerIndex].subtract(rootWithVelocityWeightsTensor);
            this.biasesMatrices[layerIndex].subtract(rootWithVelocityBiasesTensor);
        }
    }

    private void backPropagateAdaGrad(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Tensor[][] gradients = this.backPropagation(inputs, targetOutput);
        Tensor[] deltaBiasesTensor = gradients[0];
        Tensor[] deltaWeightsTensor = gradients[1];
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            deltaBiasesTensor[layerIndex] = Tensor.scalarMultiply(deltaBiasesTensor[layerIndex], -this.learningRate);
            deltaWeightsTensor[layerIndex] = Tensor.scalarMultiply(deltaWeightsTensor[layerIndex], -this.learningRate);
            // Calculating square of delta weights and biases.
            Tensor squaredDeltaWeightsTensor = Tensor.tensorMapping(deltaWeightsTensor[layerIndex], (_, deltaWeight) -> Math.pow(deltaWeight, 2));
            Tensor squaredGradientsTensor = Tensor.tensorMapping(deltaBiasesTensor[layerIndex], (_, gradient) -> Math.pow(gradient, 2));
            // Weights
            this.velocityWeightsMatrices[layerIndex] = Tensor.add(this.velocityWeightsMatrices[layerIndex], squaredDeltaWeightsTensor);
            Tensor currentWeightVelocity = this.velocityWeightsMatrices[layerIndex];
            Tensor rootWithVelocityWegihtsTensor = Tensor.tensorMapping(deltaWeightsTensor[layerIndex],
                    (flatIndex, deltaWeight) -> this.learningRate * deltaWeight / Math.sqrt(currentWeightVelocity.getData()[flatIndex] + this.epsilon));
            this.weightsMatrices[layerIndex].subtract(rootWithVelocityWegihtsTensor);
            // Biases
            this.velocityBiasesMatrices[layerIndex] = Tensor.add(this.velocityBiasesMatrices[layerIndex], squaredGradientsTensor);
            Tensor currentBiasesVelocity = this.velocityBiasesMatrices[layerIndex];
            Tensor rootWithVelocityBiasesTensor = Tensor.tensorMapping(deltaBiasesTensor[layerIndex],
                    (flatIndex, gradient) -> this.learningRate * gradient / Math.sqrt(currentBiasesVelocity.getData()[flatIndex] + this.epsilon));
            this.biasesMatrices[layerIndex].subtract(rootWithVelocityBiasesTensor);
        }
    }

    private void backPropagateAdam(double @NotNull [] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        this.adamSteps++;

        Tensor[][] gradients = this.backPropagation(inputs, targetOutput);
        Tensor[] deltaBiasesTensor = gradients[0];
        Tensor[] deltaWeightsTensor = gradients[1];

        final double beta_1_t = Math.pow(this.momentumFactorBeta1, this.adamSteps);
        final double beta_2_t = Math.pow(this.momentumFactorBeta2, this.adamSteps);

        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            deltaBiasesTensor[layerIndex] = Tensor.scalarMultiply(deltaBiasesTensor[layerIndex], -this.learningRate);
            deltaWeightsTensor[layerIndex] = Tensor.scalarMultiply(deltaWeightsTensor[layerIndex], -this.learningRate);

            this.momentumWeightsMatrices[layerIndex] = Tensor.add(Tensor.scalarMultiply(this.momentumWeightsMatrices[layerIndex], this.momentumFactorBeta1),
                    Tensor.scalarMultiply(deltaWeightsTensor[layerIndex], 1 - this.momentumFactorBeta1));
            this.momentumBiasesMatrices[layerIndex] = Tensor.add(Tensor.scalarMultiply(this.momentumBiasesMatrices[layerIndex], this.momentumFactorBeta1),
                    Tensor.scalarMultiply(deltaBiasesTensor[layerIndex], 1 - this.momentumFactorBeta1));

            Tensor squaredGradientsWeights = Tensor.tensorMapping(deltaWeightsTensor[layerIndex], (_, grad) -> Math.pow(grad, 2));
            Tensor squaredGradientsBiases = Tensor.tensorMapping(deltaBiasesTensor[layerIndex], (_, grad) -> Math.pow(grad, 2));

            this.velocityWeightsMatrices[layerIndex] = Tensor.add(Tensor.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta2),
                    Tensor.scalarMultiply(squaredGradientsWeights, 1 - this.momentumFactorBeta2));
            this.velocityBiasesMatrices[layerIndex] = Tensor.add(Tensor.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta2),
                    Tensor.scalarMultiply(squaredGradientsBiases, 1 - this.momentumFactorBeta2));

            Tensor momentumWeightsHat = Tensor.tensorMapping(this.momentumWeightsMatrices[layerIndex], (_, m) -> m / (1 - beta_1_t));
            Tensor momentumBiasesHat = Tensor.tensorMapping(this.momentumBiasesMatrices[layerIndex], (_, m) -> m / (1 - beta_1_t));

            Tensor velocityWeightsHat = Tensor.tensorMapping(this.velocityWeightsMatrices[layerIndex], (_, v) -> v / (1 - beta_2_t));
            Tensor velocityBiasesHat = Tensor.tensorMapping(this.velocityBiasesMatrices[layerIndex], (_, v) -> v / (1 - beta_2_t));

            Tensor updateWeights = Tensor.tensorMapping(momentumWeightsHat, (flatIndex, m) -> this.learningRate * m / (Math.sqrt(velocityWeightsHat.getData()[flatIndex]) + this.epsilon));
            Tensor updateBiases = Tensor.tensorMapping(momentumBiasesHat, (flatIndex, m) -> this.learningRate * m / (Math.sqrt(velocityBiasesHat.getData()[flatIndex]) + this.epsilon));

            this.weightsMatrices[layerIndex].subtract(updateWeights);
            this.biasesMatrices[layerIndex].subtract(updateBiases);
        }
    }

    /**
     * Function to train model for mass amount of training inputs and outputs with random samples.
     *
     * @param epochs          number of back-propagation iterations performed by the model.
     * @param trainingInputs  2D array of inputs for training the model.
     * @param trainingOutputs 2D array of outputs for training the model.
     */
    public void train(double[] @NotNull [] trainingInputs, double[][] trainingOutputs, int epochs) {
        if (trainingInputs[0].length != this.numberOfInputNode) {
            throw new IllegalArgumentException("Mismatch inputs size.");
        }
        if (trainingOutputs[0].length != this.netWorkDenseLayers[this.netWorkDenseLayers.length - 1].getNumberOfNodes()) {
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
            switch (this.jNetworkOptimizer) {
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

    public static Tensor getAppliedActivationFunctionTensor(Tensor tensor, @NotNull ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (tensor.getShape()[1] != 1) {
                throw new IllegalArgumentException("Unable to apply softmax due to more column existing in the given Tensor.");
            }
            //            System.out.println("Printing non applied Tensor: ");
            //            Tensor.printTensor();
            Tensor eRasiedTensor = Tensor.tensorMapping(tensor, (_, value) -> Math.exp(value));
            //            System.out.println("Printing powered by e Tensor: ");
            //            eRasiedTensor.printTensor();
            double sum = IntStream.range(0, eRasiedTensor.getShape()[0]).mapToDouble(a -> eRasiedTensor.getEntry(a, 0)).sum();
            Tensor result = Tensor.tensorMapping(eRasiedTensor, ((flatIndex, value) -> value / sum));
            //            System.out.println("After getting averaged by the sum: ");
            //            result.printTensor();
            return result;
        }
        return Tensor.tensorMapping(tensor, activationFunction.getEquation());
    }

    public static Tensor getDeactivatedActivationFunctionTensor(Tensor activatedTensor, @NotNull ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (activatedTensor.getShape()[1] != 1) {
                throw new IllegalArgumentException("Softmax derivative expects a single vector output.");
            }
            int n = Math.max(activatedTensor.getShape()[0], activatedTensor.getShape()[1]);
            Tensor result = new Tensor(n, n);
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    double entry = a == b ? (activatedTensor.getEntry(a, b) * (1 - activatedTensor.getEntry(a, b))) : -activatedTensor.getEntry(a, b) * activatedTensor.getEntry(b, a);
                    result.setEntry(entry, a, b);
                }
            }
        }
        return Tensor.tensorMapping(activatedTensor, activationFunction.getDerivative());
    }

    public double getEpsilon() {
        return epsilon;
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

    public JNetworkOptimizer getJNeuralNetworkOptimizer() {
        return jNetworkOptimizer;
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

    public JNeuralNetwork setJNeuralNetworkOptimizer(JNetworkOptimizer jNetworkOptimizer) {
        this.jNetworkOptimizer = jNetworkOptimizer;
        return this;
    }

    private void randomize(@NotNull Tensor tensor) {
        double stdDev = Math.sqrt(2.0 / tensor.getShape()[1]);
        for (int a = 0; a < tensor.getShape()[0]; a++) {
            for (int b = 0; b < tensor.getShape()[1]; b++) {
                double u1 = random.nextDouble();
                double u2 = random.nextDouble();
                double gaussian = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                tensor.setEntry(gaussian * stdDev, a, b);
            }
        }
    }
}