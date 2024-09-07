package com.mkproductions.jnn.network;

import com.mkproductions.jnn.entity.*;
import com.mkproductions.jnn.entity.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.entity.optimzers.JNeuralNetworkOptimizer;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class JNeuralNetwork implements Serializable {
    private final int numberOfInputNode;
    private int networkAccuracy = 0;

    private final Layer[] netWorkLayers;
    private final Matrix2DDouble[] weightsMatrices;
    private final Matrix2DDouble[] biasesMatrices;
    private final Matrix2DDouble[] velocityWeightsMatrices;
    private final Matrix2DDouble[] velocityBiasesMatrices;
    private final Matrix2DDouble[] momentumWeightsMatrices;
    private final Matrix2DDouble[] momentumBiasesMatrices;

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
        this.learningRate = 0.01F;
        this.numberOfInputNode = numberOfInputNode;
        this.netWorkLayers = netWorkLayers;
        // Initializing the arrays
        this.weightsMatrices = new Matrix2DDouble[netWorkLayers.length];
        this.biasesMatrices = new Matrix2DDouble[netWorkLayers.length];
        this.velocityWeightsMatrices = new Matrix2DDouble[this.weightsMatrices.length];
        this.velocityBiasesMatrices = new Matrix2DDouble[this.weightsMatrices.length];
        this.momentumWeightsMatrices = new Matrix2DDouble[this.weightsMatrices.length];
        this.momentumBiasesMatrices = new Matrix2DDouble[this.weightsMatrices.length];
        this.momentumFactorBeta1 = 0.9F;
        // Assign weights and biases and velocity to matrix arrays
        for (int layerIndex = 0; layerIndex < this.weightsMatrices.length; layerIndex++) {
            if (layerIndex == 0) {
                this.weightsMatrices[layerIndex] = new Matrix2DDouble(this.netWorkLayers[layerIndex].numberOfNodes(), this.numberOfInputNode);
            } else {
                this.weightsMatrices[layerIndex] = new Matrix2DDouble(this.netWorkLayers[layerIndex].numberOfNodes(), this.netWorkLayers[layerIndex - 1].numberOfNodes());
            }
            this.biasesMatrices[layerIndex] = new Matrix2DDouble(this.netWorkLayers[layerIndex].numberOfNodes(), 1);
            // Randomizing the weights and bias
            randomize(this.weightsMatrices[layerIndex]);
            randomize(this.biasesMatrices[layerIndex]);
            // Initializing the velocity matrices.
            if (this.jNeuralNetworkOptimizer != JNeuralNetworkOptimizer.SGD) {
                this.velocityWeightsMatrices[layerIndex] = new Matrix2DDouble(this.weightsMatrices[layerIndex].getNumRows(), this.weightsMatrices[layerIndex].getNumColumns());
                this.velocityBiasesMatrices[layerIndex] = new Matrix2DDouble(this.biasesMatrices[layerIndex].getNumRows(), this.biasesMatrices[layerIndex].getNumColumns());
            }
            if (this.jNeuralNetworkOptimizer == JNeuralNetworkOptimizer.ADAM) {
                this.momentumWeightsMatrices[layerIndex] = new Matrix2DDouble(this.weightsMatrices[layerIndex].getNumRows(), this.weightsMatrices[layerIndex].getNumColumns());
                this.momentumBiasesMatrices[layerIndex] = new Matrix2DDouble(this.biasesMatrices[layerIndex].getNumRows(), this.biasesMatrices[layerIndex].getNumColumns());
            }
        }
    }

    public JNeuralNetwork(JNeuralNetwork jNeuralNetwork) {
        this.numberOfInputNode = jNeuralNetwork.numberOfInputNode;
        this.netWorkLayers = jNeuralNetwork.netWorkLayers;
        this.learningRate = jNeuralNetwork.getLearningRate();
        // Initializing the arrays
        this.weightsMatrices = new Matrix2DDouble[netWorkLayers.length];
        this.biasesMatrices = new Matrix2DDouble[netWorkLayers.length];
        this.velocityWeightsMatrices = new Matrix2DDouble[jNeuralNetwork.weightsMatrices.length];
        this.velocityBiasesMatrices = new Matrix2DDouble[jNeuralNetwork.weightsMatrices.length];
        this.momentumWeightsMatrices = new Matrix2DDouble[jNeuralNetwork.weightsMatrices.length];
        this.momentumBiasesMatrices = new Matrix2DDouble[jNeuralNetwork.weightsMatrices.length];
        this.networkAccuracy = jNeuralNetwork.networkAccuracy;
        this.lossFunctionable = jNeuralNetwork.lossFunctionable;
        this.jNeuralNetworkOptimizer = jNeuralNetwork.jNeuralNetworkOptimizer;
        this.momentumFactorBeta1 = jNeuralNetwork.getMomentumFactorBeta1();
        this.momentumFactorBeta2 = jNeuralNetwork.getMomentumFactorBeta2();
        this.debugMode = jNeuralNetwork.debugMode;
        // Assign weights and biases and velocity to matrix arrays
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
        if (length != this.numberOfInputNode) throw new RuntimeException("Mismatch length of inputs to the network.");
    }

    private Matrix2DDouble[] forwardPropagation(double[] inputs) {
        final Matrix2DDouble inputMatrix2DDouble = new Matrix2DDouble(new double[][]{inputs});
        Matrix2DDouble[] outputMatrices = new Matrix2DDouble[this.netWorkLayers.length];
        for (int a = 0; a < this.weightsMatrices.length; a++) {
//            if (a == 0)
//                outputMatrices[a] = Matrix2DDouble.add(Matrix2DDouble.matrixMultiplication(this.weightsMatrices[a], inputMatrix2DDouble), this.biasesMatrices[a]);
//            else
//                outputMatrices[a] = Matrix2DDouble.add(Matrix2DDouble.matrixMultiplication(this.weightsMatrices[a], outputMatrices[a - 1]), this.biasesMatrices[a]);
//            outputMatrices[a] = getAppliedActivationFunctionMatrix2DDouble(outputMatrices[a], this.netWorkLayers[a].activationFunction());
        }
        return outputMatrices;
    }

    /**
     * Process inputs and produces outputs as per the network schema.
     *
     * @param inputs A double array to predict the output.
     * @return double array of output predicted by the network.
     */
    public double[] processInputs(double[] inputs) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Matrix2DDouble[] outputMatrices = this.forwardPropagation(inputs);
        double[] outputs = new double[this.netWorkLayers[this.netWorkLayers.length - 1].numberOfNodes()];
        for (int a = 0; a < outputs.length; a++) {
            outputs[a] = outputMatrices[a].get(a, 0);
        }
        return outputs;
    }

    private Matrix2DDouble[][] backPropagation(double[] inputs, double[] targets) {
        Matrix2DDouble[] biasesGradients = new Matrix2DDouble[this.weightsMatrices.length];
        Matrix2DDouble[] weightsGradients = new Matrix2DDouble[this.weightsMatrices.length];
//        Matrix2DDouble targetMatrix2DDouble = new Matrix2DDouble(targets, targets.length, 1);
        Matrix2DDouble[] outputMatrices = forwardPropagation(inputs);
        Matrix2DDouble outputMatrix2DDouble = outputMatrices[outputMatrices.length - 1];
//        Matrix2DDouble errorMatrix2DDouble = this.getLossFunctionable().getDerivativeMatrix2DDouble(outputMatrix2DDouble, targetMatrix2DDouble);
//        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
//            biasesGradients[layerIndex] = getDactivatedActivationFunctionMatrix2DDouble(outputMatrices[layerIndex], this.netWorkLayers[layerIndex].activationFunction());
//            biasesGradients[layerIndex] = Matrix2DDouble.elementWiseMultiply(biasesGradients[layerIndex], errorMatrix2DDouble);
//            biasesGradients[layerIndex] = Matrix2DDouble.scalarMultiply(biasesGradients[layerIndex], -this.learningRate);
//            errorMatrix2DDouble = Matrix2DDouble.matrixMultiplication(this.weightsMatrices[layerIndex].transpose(), errorMatrix2DDouble);
//            Matrix2DDouble previousOutputMatrix2DDouble = (layerIndex == 0) ? new Matrix2DDouble(inputs, inputs.length, 1) : outputMatrices[layerIndex - 1].transpose();
//            weightsGradients[layerIndex] = Matrix2DDouble.matrixMultiplication(biasesGradients[layerIndex], previousOutputMatrix2DDouble);
//        }
        return new Matrix2DDouble[][]{biasesGradients, weightsGradients};
    }

    /**
     * Function to perform back-propagate and adjusts weights and biases as per the given inputs with targets.
     *
     * @param inputs       2D array of inputs to be learned by network.
     * @param targetOutput 2D array to train the network as per inputs index.
     */
    private void backPropagateSGD(double[] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        // Calculating the gradients.
        Matrix2DDouble[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix2DDouble[] biasesGradients = gradients[0];
        Matrix2DDouble[] deltaWeightsMatrix2DDouble = gradients[1];
//        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
////            // Applying the change of weights in the current weights of the network.
////            this.weightsMatrices[layerIndex].subtract(deltaWeightsMatrix2DDouble[layerIndex]);
////            this.biasesMatrices[layerIndex].subtract(biasesGradients[layerIndex]);
//        }
    }

    private void backPropagateSGDWithMomentum(double[] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Matrix2DDouble[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix2DDouble[] biasesGradients = gradients[0];
        Matrix2DDouble[] deltaWeightsMatrix2DDouble = gradients[1];
//        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
//            // Calculating the velocities of the weights and biases
//            this.velocityWeightsMatrices[layerIndex] = Matrix2DDouble.add(Matrix2DDouble.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta1), Matrix2DDouble.scalarMultiply(deltaWeightsMatrix2DDouble[layerIndex], 1 - this.momentumFactorBeta1));
//            this.velocityBiasesMatrices[layerIndex] = Matrix2DDouble.add(Matrix2DDouble.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta1), Matrix2DDouble.scalarMultiply(biasesGradients[layerIndex], 1 - this.momentumFactorBeta1));
//            // Applying the change of weights in the current weights of the network.
//            this.weightsMatrices[layerIndex].subtract(this.velocityWeightsMatrices[layerIndex]);
//            this.biasesMatrices[layerIndex].subtract(this.velocityBiasesMatrices[layerIndex]);
//        }
    }

    private void backPropagateRMSProp(double[] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Matrix2DDouble[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix2DDouble[] biasesGradients = gradients[0];
        Matrix2DDouble[] deltaWeightsMatrix2DDouble = gradients[1];
//        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
//            // Weights
//            Matrix2DDouble squaredDeltaWeightsMatrix2DDouble = Matrix2DDouble.matrixMapping(deltaWeightsMatrix2DDouble[layerIndex], (r, c, val) -> Math.pow(val, 2));
//            this.velocityWeightsMatrices[layerIndex] = Matrix2DDouble.add(Matrix2DDouble.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta1), Matrix2DDouble.scalarMultiply(squaredDeltaWeightsMatrix2DDouble, 1 - this.momentumFactorBeta1));
//            Matrix2DDouble newWeightsVelocityMatrix2DDouble = this.velocityWeightsMatrices[layerIndex];
//            Matrix2DDouble rootWithVelocityWegihtsMatrix2DDouble = Matrix2DDouble.matrixMapping(deltaWeightsMatrix2DDouble[layerIndex], (r, c, deltaWeight) -> this.learningRate * deltaWeight / Math.sqrt(newWeightsVelocityMatrix2DDouble.getEntry(r, c) + this.epsilonRMSProp));
//            // Biases
//            Matrix2DDouble squaredGradientsMatrix2DDouble = Matrix2DDouble.matrixMapping(biasesGradients[layerIndex], (r, c, gradient) -> Math.pow(gradient, 2));
//            this.velocityBiasesMatrices[layerIndex] = Matrix2DDouble.add(Matrix2DDouble.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta1), Matrix2DDouble.scalarMultiply(squaredGradientsMatrix2DDouble, 1 - this.momentumFactorBeta1));
//            Matrix2DDouble newBiasesVelocityMatrix2DDouble = this.velocityBiasesMatrices[layerIndex];
//            Matrix2DDouble rootWithVelocityBiasesMatrix2DDouble = Matrix2DDouble.matrixMapping(biasesGradients[layerIndex], (r, c, gradient) -> this.learningRate * gradient / Math.sqrt(newBiasesVelocityMatrix2DDouble.getEntry(r, c) + this.epsilonRMSProp));
//            // Applying the change of weights in the current weights of the network.
//            this.weightsMatrices[layerIndex].subtract(rootWithVelocityWegihtsMatrix2DDouble);
//            this.biasesMatrices[layerIndex].subtract(rootWithVelocityBiasesMatrix2DDouble);
//        }
    }

    private void backPropagateAdaGrad(double[] inputs, double[] targetOutput) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);
        Matrix2DDouble[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix2DDouble[] biasesGradients = gradients[0];
        Matrix2DDouble[] deltaWeightsMatrix2DDouble = gradients[1];
//        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
//            Matrix2DDouble squaredDeltaWeightsMatrix2DDouble = Matrix2DDouble.matrixMapping(deltaWeightsMatrix2DDouble[layerIndex], (r, c, deltaWeight) -> Math.pow(deltaWeight, 2));
//            Matrix2DDouble squaredGradientsMatrix2DDouble = Matrix2DDouble.matrixMapping(biasesGradients[layerIndex], (r, c, gradient) -> Math.pow(gradient, 2));
//            // Weights
//            this.velocityWeightsMatrices[layerIndex] = Matrix2DDouble.add(this.velocityWeightsMatrices[layerIndex], squaredDeltaWeightsMatrix2DDouble);
//            Matrix2DDouble currentWeightVelocity = this.velocityWeightsMatrices[layerIndex];
//            Matrix2DDouble rootWithVelocityWegihtsMatrix2DDouble = Matrix2DDouble.matrixMapping(deltaWeightsMatrix2DDouble[layerIndex], (row, column, deltaWeight) -> this.learningRate * deltaWeight / Math.sqrt(currentWeightVelocity.getEntry(row, column) + this.epsilonRMSProp));
//            this.weightsMatrices[layerIndex].subtract(rootWithVelocityWegihtsMatrix2DDouble);
//            // Biases
//            this.velocityBiasesMatrices[layerIndex] = Matrix2DDouble.add(this.velocityBiasesMatrices[layerIndex], squaredGradientsMatrix2DDouble);
//            Matrix2DDouble currentBiasesVelocity = this.velocityBiasesMatrices[layerIndex];
//            Matrix2DDouble rootWithVelocityBiasesMatrix2DDouble = Matrix2DDouble.matrixMapping(biasesGradients[layerIndex], (row, column, gradient) -> this.learningRate * gradient / Math.sqrt(currentBiasesVelocity.getEntry(row, column) + this.epsilonRMSProp));
//            this.biasesMatrices[layerIndex].subtract(rootWithVelocityBiasesMatrix2DDouble);
//        }
    }

    private void backPropagateAdam(double[] inputs, double[] targetOutput, int iterationCount) {
        this.generateIfInvalidParametersExceptionGenerates(inputs.length);

        Matrix2DDouble[][] gradients = this.backPropagation(inputs, targetOutput);
        Matrix2DDouble[] biasesGradients = gradients[0];
        Matrix2DDouble[] deltaWeightsMatrix2DDouble = gradients[1];
//        for (int layerIndex = this.netWorkLayers.length - 1; layerIndex >= 0; layerIndex--) {
//            // Calculating the velocity and momentum of the weights and biases.
//            this.momentumWeightsMatrices[layerIndex] = Matrix2DDouble.add(Matrix2DDouble.scalarMultiply(this.momentumWeightsMatrices[layerIndex], this.momentumFactorBeta1), Matrix2DDouble.scalarMultiply(deltaWeightsMatrix2DDouble[layerIndex], 1 - this.momentumFactorBeta1));
//            this.momentumBiasesMatrices[layerIndex] = Matrix2DDouble.add(Matrix2DDouble.scalarMultiply(this.momentumBiasesMatrices[layerIndex], this.momentumFactorBeta1), Matrix2DDouble.scalarMultiply(biasesGradients[layerIndex], 1 - this.momentumFactorBeta1));
//
//            Matrix2DDouble squaredDeltaWeightsMatrix2DDouble = Matrix2DDouble.matrixMapping(deltaWeightsMatrix2DDouble[layerIndex], (r, c, deltaWeight) -> Math.pow(deltaWeight, 2));
//            Matrix2DDouble squaredGradientsMatrix2DDouble = Matrix2DDouble.matrixMapping(biasesGradients[layerIndex], (r, c, gradient) -> Math.pow(gradient, 2));
//
//            this.velocityWeightsMatrices[layerIndex] = Matrix2DDouble.add(Matrix2DDouble.scalarMultiply(this.velocityWeightsMatrices[layerIndex], this.momentumFactorBeta2), Matrix2DDouble.scalarMultiply(squaredDeltaWeightsMatrix2DDouble, 1 - this.momentumFactorBeta2));
//            this.velocityBiasesMatrices[layerIndex] = Matrix2DDouble.add(Matrix2DDouble.scalarMultiply(this.velocityBiasesMatrices[layerIndex], this.momentumFactorBeta2), Matrix2DDouble.scalarMultiply(squaredGradientsMatrix2DDouble, 1 - this.momentumFactorBeta2));
//
//            double beta1_t = Math.pow(this.momentumFactorBeta1, iterationCount + 1);
//            double beta2_t = Math.pow(this.momentumFactorBeta2, iterationCount + 1);
//
//            Matrix2DDouble momentumWeightsHatMatrix2DDouble = Matrix2DDouble.matrixMapping(this.momentumWeightsMatrices[layerIndex], ((row, column, weightMomentum) -> weightMomentum / (1 - beta1_t)));
//            Matrix2DDouble momentumBiasesHatMatrix2DDouble = Matrix2DDouble.matrixMapping(this.momentumBiasesMatrices[layerIndex], ((row, column, biasMomentum) -> biasMomentum / (1 - beta1_t)));
//
//            Matrix2DDouble velocityWeightsHatMatrix2DDouble = Matrix2DDouble.matrixMapping(this.velocityWeightsMatrices[layerIndex], ((row, column, weightVelocity) -> weightVelocity / (1 - beta2_t)));
//            Matrix2DDouble velocityBiasesHatMatrix2DDouble = Matrix2DDouble.matrixMapping(this.velocityBiasesMatrices[layerIndex], ((row, column, biasVelocity) -> biasVelocity / (1 - beta2_t)));
//
//            Matrix2DDouble mometumRootWithVelocityWegihtsMatrix2DDouble = Matrix2DDouble.matrixMapping(momentumWeightsHatMatrix2DDouble, (row, column, weightsMomentum) -> this.learningRate * weightsMomentum / (Math.sqrt(velocityWeightsHatMatrix2DDouble.getEntry(row, column)) + this.epsilonRMSProp));
//            Matrix2DDouble mometumRootWithVelocityBiasesMatrix2DDouble = Matrix2DDouble.matrixMapping(momentumBiasesHatMatrix2DDouble, (row, column, biasMomentum) -> this.learningRate * biasMomentum / (Math.sqrt(velocityBiasesHatMatrix2DDouble.getEntry(row, column)) + this.epsilonRMSProp));
//
//            // Applying the change of weights in the current weights of the network.
//            this.weightsMatrices[layerIndex].subtract(mometumRootWithVelocityWegihtsMatrix2DDouble);
//            this.biasesMatrices[layerIndex].subtract(mometumRootWithVelocityBiasesMatrix2DDouble);
//        }
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

//    public static Matrix2DDouble getAppliedActivationFunctionMatrix2DDouble(Matrix2DDouble matrix, ActivationFunction activationFunction) {
//        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
//            if (matrix.getNumColumns() != 1) {
//                throw new RuntimeException("Unable to apply softmax due to more column exsisting in the given matric.");
//            } else {
//                double sum = Arrays.stream(matrix.getColumn(0)).map(Math::exp).sum();
//                return Matrix2DDouble.matrixMapping(matrix, (a, b, value) -> Math.exp(value) / sum);
//            }
//        }
//        return Matrix2DDouble.matrixMapping(matrix, activationFunction.equation);
//    }
//
//    public static Matrix2DDouble getDactivatedActivationFunctionMatrix2DDouble(Matrix2DDouble activatedMatrix2DDouble, ActivationFunction activationFunction) {
//        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
//            if (activatedMatrix2DDouble.getNumColumns() != 1) {
//                throw new RuntimeException("Unable to apply softmax due to more column existing in the given matrix.");
//            } else {
//                Matrix2DDouble diagonalMatrix2DDouble = Matrix2DDouble.createFromArrayToDiagonalMatrix2DDouble(activatedMatrix2DDouble.getColumn(0));
//                Matrix2DDouble identityMatrix2DDouble = Matrix2DDouble.createFromArrayToIdentityMatrix2DDouble(activatedMatrix2DDouble.getNumRows());
//                System.out.println("Diagonal matrix: ");
//                diagonalMatrix2DDouble.printMatrix2DDouble();
//                System.out.println("Identity matrix: ");
//                identityMatrix2DDouble.printMatrix2DDouble();
//                var result = Matrix2DDouble.matrixMultiplication(diagonalMatrix2DDouble, Matrix2DDouble.subtract(identityMatrix2DDouble, diagonalMatrix2DDouble));
//                System.out.println("Deactivated matrix: ");
//                result.printMatrix2DDouble();
//                return Matrix2DDouble.createFromDiagonalToColumnMatrix2DDouble(result);
//            }
//        }
//        return Matrix2DDouble.matrixMapping(activatedMatrix2DDouble, activationFunction.derivative);
//    }

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

    private void randomize(Matrix2DDouble matrix2DDouble) {
        for (int a = 0; a < matrix2DDouble.getNumRows(); a++) {
            for (int i = 0; i < matrix2DDouble.getNumColumns(); i++) {
                matrix2DDouble.set(a, i, Math.random() * 2 - 1);
            }
        }
    }
}