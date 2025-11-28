package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.gpu.TaskGraphMatrixSolver;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

import java.security.SecureRandom;
import java.util.Random;

public class JGPUNeuralNetwork {
    private int epochs = 0;
    private final int numberOfInputsNode;
    private final DenseLayer[] networkLayers;
    private final Matrix2DDouble[] weightsMatrices;
    private final Matrix2DDouble[] biasesMatrices;
    private final Matrix2DDouble[] outputMatrices;
    private final Matrix2DDouble[] errorMatrices;
    private final Matrix2DDouble[] previousOutputTransposeMatrices;
    private final Matrix2DDouble[] biasesGradientsMatrices;
    private final Matrix2DDouble[] weightGradientsMatrices;
    private final Matrix2DDouble[] weightsTransposeMatrices;
    private double learningRate;
    private double[][] inputs;
    private double[][] targets;

    private TornadoExecutionPlan networkInitializatonTornadoExecutionPlan;
    private TornadoExecutionPlan networkFeedForwardTornadoExecutionPlan;
    private TornadoExecutionPlan networkBackPropagationTornadoExecutionPlan;
    private final LossFunction lossFunction;
    private final Random random = new SecureRandom();

    public JGPUNeuralNetwork(LossFunction lossFunction, int numberOfInputsNode, DenseLayer... networkLayers) {
        this.numberOfInputsNode = numberOfInputsNode;
        this.lossFunction = lossFunction;
        this.networkLayers = networkLayers;
        this.weightsMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.biasesMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.outputMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.errorMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.previousOutputTransposeMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.biasesGradientsMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.weightGradientsMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.weightsTransposeMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.inputs = new double[][] { new double[this.numberOfInputsNode] };
        this.learningRate = 0.01;
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            if (layerIndex == 0) {
                this.weightsMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].getNumberOfNodes(), this.numberOfInputsNode);
            } else {
                this.weightsMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].getNumberOfNodes(), this.networkLayers[layerIndex - 1].getNumberOfNodes());
            }
            this.outputMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].getNumberOfNodes(), 1);
            this.biasesMatrices[layerIndex] = new Matrix2DDouble(this.outputMatrices[layerIndex].getNumRows(), this.outputMatrices[layerIndex].getNumColumns());
            this.errorMatrices[layerIndex] = new Matrix2DDouble(this.outputMatrices[layerIndex].getNumRows(), this.outputMatrices[layerIndex].getNumColumns());
            this.biasesGradientsMatrices[layerIndex] = new Matrix2DDouble(this.biasesMatrices[layerIndex].getNumRows(), this.biasesMatrices[layerIndex].getNumColumns());
            this.weightGradientsMatrices[layerIndex] = new Matrix2DDouble(this.weightsMatrices[layerIndex].getNumRows(), this.weightsMatrices[layerIndex].getNumColumns());
            this.weightsTransposeMatrices[layerIndex] = new Matrix2DDouble(this.weightsMatrices[layerIndex].getNumColumns(), this.weightsMatrices[layerIndex].getNumRows());
            if (layerIndex == 0) {
                this.previousOutputTransposeMatrices[layerIndex] = new Matrix2DDouble(1, this.numberOfInputsNode);
            } else {
                this.previousOutputTransposeMatrices[layerIndex] = new Matrix2DDouble(this.outputMatrices[layerIndex - 1].getNumColumns(), this.outputMatrices[layerIndex - 1].getNumRows());
            }
        }
        this.initializeRandomNetwork();
        this.targets = new double[][] { new double[this.networkLayers[this.networkLayers.length - 1].getNumberOfNodes()] };
    }

    private void initializeRandomNetwork() {
        // Transferring required variables to the computation device.
        TaskGraph initializeNetworkTaskGraph = new TaskGraph("InitializationNetwork");
        initializeNetworkTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.numberOfInputsNode);
        initializeNetworkTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, learningRate);
        for (int a = 0; a < this.networkLayers.length; a++) {
            initializeNetworkTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.weightsMatrices[a]);
            initializeNetworkTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.biasesMatrices[a]);
        }
        // Initializing all the weights and biases matrices.
        for (int a = 0; a < this.networkLayers.length; a++) {
            TaskGraphMatrixSolver.initializeMatrixWithRandomNumbers(initializeNetworkTaskGraph, STR."initWeightMatrix\{a}", this.weightsMatrices[a]);
            TaskGraphMatrixSolver.initializeMatrixWithRandomNumbers(initializeNetworkTaskGraph, STR."initBiasMatrix\{a}", this.biasesMatrices[a]);
        }
        // Reverting required variables to the host machine.
        for (int a = 0; a < this.networkLayers.length; a++) {
            initializeNetworkTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, this.weightsMatrices[a]);
            initializeNetworkTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, this.biasesMatrices[a]);
        }
        this.networkInitializatonTornadoExecutionPlan = new TornadoExecutionPlan(initializeNetworkTaskGraph.snapshot());
        System.out.println("Initialization of the network can be successfully ");
    }

    private void prepareFeedForwardTaskGraph() {
        // Creating the task graph for performing the feed forward propagation.
        TaskGraph feedForwardTaskGraph = new TaskGraph("FeedForwardNetwork");
        Matrix2DDouble inputMatrix = new Matrix2DDouble(this.inputs[0].length, 1);
        for (int a = 0; a < inputMatrix.getNumRows(); a++) {
            inputMatrix.set(a, 0, this.inputs[0][a]);
        }
        // Transferring all network data to the computational device.
        feedForwardTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputMatrix);
        for (int layerIndex = 0; layerIndex < outputMatrices.length; layerIndex++) {
            feedForwardTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.outputMatrices[layerIndex]);
            feedForwardTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.weightsMatrices[layerIndex]);
            feedForwardTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.biasesMatrices[layerIndex]);
        }
        // Attaching all the computation tasks in the task graph.
        for (int layerIndex = 0; layerIndex < this.outputMatrices.length; layerIndex++) {
            if (layerIndex == 0) {
                TaskGraphMatrixSolver.solveMatrixMultiplication(feedForwardTaskGraph, "inputWithWeightMultiplication_", this.weightsMatrices[layerIndex], inputMatrix, this.outputMatrices[layerIndex]);
            } else {
                TaskGraphMatrixSolver.solveMatrixMultiplication(feedForwardTaskGraph, STR."outputOfLayer_MatrixMultiplication_\{layerIndex}", this.weightsMatrices[layerIndex],
                        this.outputMatrices[layerIndex - 1], this.outputMatrices[layerIndex]);
            }
            TaskGraphMatrixSolver.solveAddition(feedForwardTaskGraph, STR."outputOfLayerWithBiases+\{layerIndex}", this.outputMatrices[layerIndex], this.biasesMatrices[layerIndex]);
            TaskGraphMatrixSolver.applyActivationFunction(feedForwardTaskGraph, STR."applyingActivationFunctionToLayerOutput\{layerIndex}", this.outputMatrices[layerIndex],
                    this.networkLayers[layerIndex].getActivationFunction());
        }
        // Reverting required outputs of all the layers.
        for (Matrix2DDouble outputMatrix : outputMatrices) {
            feedForwardTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, outputMatrix);
        }

        this.networkFeedForwardTornadoExecutionPlan = new TornadoExecutionPlan(feedForwardTaskGraph.snapshot());
    }

    private void prepareBackPropagationTaskGraph() {
        TaskGraph backPropagationTaskGraph = new TaskGraph("BackPropagationNetwork");
        int randomIndex = random.nextInt(this.inputs.length);
        double[] input = this.inputs[randomIndex];
        double[] target = this.targets[randomIndex];
        Matrix2DDouble targetMatrix = new Matrix2DDouble(target.length, 1);
        Matrix2DDouble inputMatrix = new Matrix2DDouble(this.inputs[0].length, 1);
        for (int a = 0; a < inputMatrix.getNumRows(); a++) {
            inputMatrix.set(0, a, input[a]);
        }
        for (int a = 0; a < targetMatrix.getNumRows(); a++) {
            targetMatrix.set(0, a, target[a]);
        }
        backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputMatrix);
        backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, targetMatrix);
        backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.previousOutputTransposeMatrices[0]);
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.outputMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.previousOutputTransposeMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.errorMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.weightsMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.biasesMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.weightsTransposeMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.weightGradientsMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.biasesGradientsMatrices[layerIndex]);
        }
        for (int layerIndex = 0; layerIndex < this.outputMatrices.length; layerIndex++) {
            if (layerIndex == 0) {
                TaskGraphMatrixSolver.solveMatrixMultiplication(backPropagationTaskGraph, "inputWithWeightMultiplication:", this.weightsMatrices[layerIndex], inputMatrix,
                        this.outputMatrices[layerIndex]);
            } else {
                TaskGraphMatrixSolver.solveMatrixMultiplication(backPropagationTaskGraph, STR."outputOfLayer_MatrixMultiplication:\{layerIndex}", this.weightsMatrices[layerIndex],
                        this.outputMatrices[layerIndex - 1], this.outputMatrices[layerIndex]);
            }
            TaskGraphMatrixSolver.solveAddition(backPropagationTaskGraph, STR."outputOfLayerWithBiases:\{layerIndex}", this.outputMatrices[layerIndex], this.biasesMatrices[layerIndex]);
            TaskGraphMatrixSolver.applyActivationFunction(backPropagationTaskGraph, STR." applyingActivationFunctionToLayerOutput \{layerIndex}", this.outputMatrices[layerIndex],
                    this.networkLayers[layerIndex].getActivationFunction());
        }
        // Performing back propagation.
        TaskGraphMatrixSolver.

                calculateLossDerivative(backPropagationTaskGraph, "calculatingError:", this.outputMatrices[this.outputMatrices.length - 1], targetMatrix,
                this.errorMatrices[this.outputMatrices.length - 1], this.lossFunction);
        for (int layerIndex = this.networkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            backPropagationTaskGraph.

                    transferToDevice(DataTransferMode.UNDER_DEMAND, this.errorMatrices[layerIndex]);
            // Calculating the gradients for bias matrix.
            TaskGraphMatrixSolver.applyActivationFunctionDerivative(backPropagationTaskGraph, STR."activationFunctionDerivative\{layerIndex}", this.outputMatrices[layerIndex],
                    this.networkLayers[layerIndex].getActivationFunction());
            TaskGraphMatrixSolver.solveElementWiseMultiplication(backPropagationTaskGraph, STR."elementWiseGradients\{layerIndex}", this.biasesMatrices[layerIndex], this.errorMatrices[layerIndex],
                    this.biasesGradientsMatrices[layerIndex]);
            TaskGraphMatrixSolver.solveMatrixScaling(backPropagationTaskGraph, STR."learningRateMultiplication\{layerIndex}", this.biasesGradientsMatrices[layerIndex], -this.learningRate);
            //Calculating the gradients for weight matrix.
            TaskGraphMatrixSolver.transpose(backPropagationTaskGraph, STR."weightsTranspose\{layerIndex}", this.weightsMatrices[layerIndex], this.weightsTransposeMatrices[layerIndex]);
            // Resetting the error as per calculating iterative layers.
            if (layerIndex != 0) {
                Matrix2DDouble hiddenLayerErrorMatrix = new Matrix2DDouble(this.outputMatrices[layerIndex - 1].getNumRows(), this.outputMatrices[layerIndex - 1].getNumColumns());
                backPropagationTaskGraph.transferToDevice(DataTransferMode.UNDER_DEMAND, hiddenLayerErrorMatrix);
                TaskGraphMatrixSolver.solveMatrixMultiplication(backPropagationTaskGraph, STR."weightTransposeErrorMatrixMultiplication:\{layerIndex}", this.weightsTransposeMatrices[layerIndex],
                        this.errorMatrices[layerIndex], hiddenLayerErrorMatrix);
            }
            TaskGraphMatrixSolver.transpose(backPropagationTaskGraph, STR."previousMatrix:\{layerIndex}", layerIndex == 0 ? inputMatrix : this.outputMatrices[layerIndex - 1],
                    this.previousOutputTransposeMatrices[layerIndex]);
            TaskGraphMatrixSolver.solveMatrixMultiplication(backPropagationTaskGraph, STR."calculatingWrightGradients:\{layerIndex}", this.biasesGradientsMatrices[layerIndex],
                    this.previousOutputTransposeMatrices[layerIndex], this.weightGradientsMatrices[layerIndex]);
            // Updating the weights of the network.
            TaskGraphMatrixSolver.solveSubtraction(backPropagationTaskGraph, STR."updatingWeights:\{layerIndex}", this.weightsMatrices[layerIndex], this.weightGradientsMatrices[layerIndex]);
            TaskGraphMatrixSolver.solveSubtraction(backPropagationTaskGraph, STR." updatingBiases: \{layerIndex}", this.biasesMatrices[layerIndex], this.biasesGradientsMatrices[layerIndex]);
        }
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            // Reverting all the data to the host machine.
            backPropagationTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, this.weightsMatrices[layerIndex]);
            backPropagationTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, this.biasesMatrices[layerIndex]);
        }
        this.networkBackPropagationTornadoExecutionPlan = new TornadoExecutionPlan(backPropagationTaskGraph.snapshot());
    }

    public void train(double[][] trainingInputs, double[][] trainingTargets, int epochs) {
        this.epochs = epochs;
        this.inputs = trainingInputs;
        this.targets = trainingTargets;
        this.prepareBackPropagationTaskGraph();
        var device = TornadoExecutionPlan.getDevice(0, 0);
        System.out.println(STR."Device: \{device}");
        this.networkBackPropagationTornadoExecutionPlan.withDevice(device).execute();
    }

    public void initializeNetwork() {
        networkInitializatonTornadoExecutionPlan.execute();
    }

    public double[] predict(double[] inputs) {
        this.inputs = new double[][] { inputs };
        this.prepareFeedForwardTaskGraph();
        this.networkFeedForwardTornadoExecutionPlan.execute();
        Matrix2DDouble output = this.outputMatrices[this.outputMatrices.length - 1];
        return output.column(0).asBuffer().array();
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void printData() {
        System.out.println("Network layers: ");
        DenseLayer[] layers = this.networkLayers;
        for (int a = 0; a < layers.length; a++) {
            System.out.println(STR."Layer index: \{a}");
            DenseLayer DenseLayer = layers[a];
            System.out.println(DenseLayer);
            System.out.println(" Weights: ");
            System.out.println(this.weightsMatrices[a]);
            System.out.println(" Weights gradients:");
            System.out.println(this.weightGradientsMatrices[a]);
            System.out.println("Biases: ");
            System.out.println(this.biasesMatrices[a]);
            System.out.println("Biases gradients: ");
            System.out.println(this.biasesGradientsMatrices[a]);
            System.out.println();
        }
        System.out.println();
    }
}