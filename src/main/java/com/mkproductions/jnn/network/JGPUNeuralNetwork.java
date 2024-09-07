package com.mkproductions.jnn.network;

import com.mkproductions.jnn.entity.NetworkLayer;
import com.mkproductions.jnn.entity.lossFunctions.LossFunctionManager;
import com.mkproductions.jnn.entity.solvers.TaskGraphMatrixSolver;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

public class JGPUNeuralNetwork {
    private int epochs = 0;
    private final int numberOfInputsNode;
    private final NetworkLayer[] networkLayers;
    private final Matrix2DDouble[] weightsMatrices;
    private final Matrix2DDouble[] biasesMatrices;
    private final Matrix2DDouble[] outputMatrices;
    private final Matrix2DDouble[] biasGradientsMatrices;
    private final Matrix2DDouble[] weightGradientsMatrices;
    private double learningRate;
    private double[] inputs;
    private double[] targets;

    private TornadoExecutionPlan networkInitializatonTornadoExecutionPlan;
    private TornadoExecutionPlan networkFeedForwardTornadoExecutionPlan;
    private TornadoExecutionPlan networkBackPropagationTornadoExecutionPlan;
    private LossFunctionManager.LossFunction lossFunction;

    public JGPUNeuralNetwork(LossFunctionManager.LossFunction lossFunction, int numberOfInputsNode, NetworkLayer... networkLayers) {
        this.numberOfInputsNode = numberOfInputsNode;
        this.lossFunction = lossFunction;
        this.networkLayers = networkLayers;
        this.weightsMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.biasesMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.outputMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.biasGradientsMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.weightGradientsMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.inputs = new double[this.numberOfInputsNode];
        this.learningRate = 0.01;
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            if (layerIndex == 0) {
                this.weightsMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].numberOfNodes(), this.numberOfInputsNode);
            } else {
                this.weightsMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].numberOfNodes(), this.networkLayers[layerIndex - 1].numberOfNodes());
            }
            this.biasesMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].numberOfNodes(), 1);
            this.outputMatrices[layerIndex] = new Matrix2DDouble(this.biasesMatrices[layerIndex].getNumRows(), 1);
            this.biasGradientsMatrices[layerIndex] = new Matrix2DDouble(this.outputMatrices[layerIndex].getNumRows(), this.outputMatrices[layerIndex].getNumColumns());
            this.weightGradientsMatrices[layerIndex] = new Matrix2DDouble(this.weightsMatrices[layerIndex].getNumRows(), this.weightsMatrices[layerIndex].getNumColumns());
        }
        this.initializeRandomNetwork();
        this.targets = new double[this.networkLayers[this.networkLayers.length - 1].numberOfNodes()];
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
            TaskGraphMatrixSolver.initializeMatrixWithRandomNumbers(initializeNetworkTaskGraph, STR."initWeightMatrix\{a}", this.weightsMatrices[a], System.nanoTime());
            TaskGraphMatrixSolver.initializeMatrixWithRandomNumbers(initializeNetworkTaskGraph, STR."initBiasMatrix\{a}", this.biasesMatrices[a], System.nanoTime());
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
        Matrix2DDouble inputMatrix = new Matrix2DDouble(this.inputs.length, 1);
        for (int a = 0; a < inputMatrix.getNumRows(); a++) {
            inputMatrix.set(a, 0, this.inputs[a]);
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
                TaskGraphMatrixSolver.solveMatrixMultiplication(feedForwardTaskGraph, "inputWithWeightMultiplication", this.weightsMatrices[layerIndex], inputMatrix, this.outputMatrices[layerIndex]);
            } else {
                TaskGraphMatrixSolver.solveMatrixMultiplication(feedForwardTaskGraph, STR."outputOfLayer\{layerIndex}_MatrixMultiplication", this.weightsMatrices[layerIndex], this.outputMatrices[layerIndex - 1], this.outputMatrices[layerIndex]);
            }
            TaskGraphMatrixSolver.solveAddition(feedForwardTaskGraph, STR."outputOfLayer\{layerIndex}WithBiases", this.outputMatrices[layerIndex], this.biasesMatrices[layerIndex]);
            TaskGraphMatrixSolver.applyActivationFunction(feedForwardTaskGraph, STR."applyingActivationFunctionToLayerOutput\{layerIndex}", this.outputMatrices[layerIndex], this.networkLayers[layerIndex].activationFunction());
        }
        // Reverting required outputs of all the layers.
        for (Matrix2DDouble outputMatrix : outputMatrices) {
            feedForwardTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, outputMatrix);
        }

        this.networkFeedForwardTornadoExecutionPlan = new TornadoExecutionPlan(feedForwardTaskGraph.snapshot());
    }

    private void prepareBackPropagationTaskGraph() {
        TaskGraph backPropagationTaskGraph = new TaskGraph("BackPropagationNetwork");
        // Creating the task graph for performing the feed forward propagation.
        Matrix2DDouble inputMatrix = new Matrix2DDouble(this.inputs.length, 1);
        Matrix2DDouble targetMatrix = new Matrix2DDouble(this.targets.length, 1);
        Matrix2DDouble errorMatrix = new Matrix2DDouble(this.targets.length, 1);
        for (int a = 0; a < inputMatrix.getNumRows(); a++) inputMatrix.set(a, 0, this.inputs[a]);
        for (int a = 0; a < targetMatrix.getNumRows(); a++) targetMatrix.set(a, 0, this.targets[a]);
        // Transferring all network data to the computational device.
        backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputMatrix);
        backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, targetMatrix);
        backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, errorMatrix);
        for (int layerIndex = 0; layerIndex < outputMatrices.length; layerIndex++) {
            backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.outputMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.weightsMatrices[layerIndex]);
            backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, this.biasesMatrices[layerIndex]);
        }

        // Attaching all the computation tasks in the task graph.
        for (int layerIndex = 0; layerIndex < this.outputMatrices.length; layerIndex++) {
            if (layerIndex == 0)
                TaskGraphMatrixSolver.solveMatrixMultiplication(backPropagationTaskGraph, "inputWithWeightMultiplication", this.weightsMatrices[layerIndex], inputMatrix, this.outputMatrices[layerIndex]);
            else
                TaskGraphMatrixSolver.solveMatrixMultiplication(backPropagationTaskGraph, STR."outputOfLayer\{layerIndex}_MatrixMultiplication", this.weightsMatrices[layerIndex], this.outputMatrices[layerIndex - 1], this.outputMatrices[layerIndex]);
            TaskGraphMatrixSolver.solveAddition(backPropagationTaskGraph, STR."outputOfLayer\{layerIndex}WithBiases", this.outputMatrices[layerIndex], this.biasesMatrices[layerIndex]);
            TaskGraphMatrixSolver.applyActivationFunction(backPropagationTaskGraph, STR."applyingActivationFunctionToLayerOutput\{layerIndex}", this.outputMatrices[layerIndex], this.networkLayers[layerIndex].activationFunction());
        }

        // Performing back propagation.
        TaskGraphMatrixSolver.calculateLossDerivative(backPropagationTaskGraph, "calculatingError", this.outputMatrices[this.outputMatrices.length - 1], targetMatrix, errorMatrix, this.lossFunction);
        for (int layerIndex = this.networkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            // Calculating the gradients for bias matrix.
            TaskGraphMatrixSolver.applyActivationFunctionDerivative(backPropagationTaskGraph, STR."activationFunctionDerivative\{layerIndex}:", this.outputMatrices[layerIndex], this.networkLayers[layerIndex].activationFunction());
            TaskGraphMatrixSolver.solveElementWiseMultiplication(backPropagationTaskGraph, STR."elementWiseGradients\{layerIndex}", this.biasesMatrices[layerIndex], errorMatrix, this.biasGradientsMatrices[layerIndex]);
            TaskGraphMatrixSolver.solveMatrixScaling(backPropagationTaskGraph, STR."learingRateMultiplication\{layerIndex}", this.biasGradientsMatrices[layerIndex], -this.learningRate, this.biasGradientsMatrices[layerIndex]);

            // Resetting the error as per calculating iterative layers.
            errorMatrix = new Matrix2DDouble(this.outputMatrices[layerIndex].getNumRows(), this.outputMatrices[layerIndex].getNumColumns());

            backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, errorMatrix);
            //Calculating the gradients for weight matrix.
            Matrix2DDouble weightsTranspose = new Matrix2DDouble(this.weightsMatrices[layerIndex].getNumRows(), this.weightsMatrices[layerIndex].getNumColumns());
            TaskGraphMatrixSolver.transpose(backPropagationTaskGraph, STR."weightsTranspose\{layerIndex}", this.weightsMatrices[layerIndex], weightsTranspose);
            TaskGraphMatrixSolver.solveMatrixMultiplication(backPropagationTaskGraph, STR."weightMatrixMultiplication\{layerIndex}", weightsTranspose, errorMatrix, errorMatrix);
            Matrix2DDouble previousMatrix;
            if (layerIndex == 0) {
                previousMatrix = new Matrix2DDouble(inputMatrix.getNumColumns(), inputMatrix.getNumRows());
            } else {
                previousMatrix = new Matrix2DDouble(this.outputMatrices[layerIndex - 1].getNumRows(), this.outputMatrices[layerIndex - 1].getNumColumns());
            }
            backPropagationTaskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, previousMatrix);
            TaskGraphMatrixSolver.transpose(backPropagationTaskGraph, STR."previousMatrix\{layerIndex}", layerIndex == 0 ? inputMatrix : this.outputMatrices[layerIndex - 1], previousMatrix);
            TaskGraphMatrixSolver.solveMatrixMultiplication(backPropagationTaskGraph, STR."calculatingWrightGradients\{layerIndex}", this.biasGradientsMatrices[layerIndex], previousMatrix, this.weightGradientsMatrices[layerIndex]);
            // Updating the weights of the network.
            TaskGraphMatrixSolver.solveSubtraction(backPropagationTaskGraph, STR."updatingWeights\{layerIndex}", this.weightsMatrices[layerIndex], this.weightGradientsMatrices[layerIndex]);
            TaskGraphMatrixSolver.solveSubtraction(backPropagationTaskGraph, STR."updatingBiases\{layerIndex}", this.biasesMatrices[layerIndex], this.biasGradientsMatrices[layerIndex]);
        }

        // Reverting required outputs of all the layers.
        for (Matrix2DDouble outputMatrix : outputMatrices) {
            backPropagationTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, outputMatrix);
        }

        this.networkBackPropagationTornadoExecutionPlan = new TornadoExecutionPlan(backPropagationTaskGraph.snapshot());
    }

    private void train(double[] trainingInputs, double[] trainingTargets) {
        this.inputs = trainingInputs;
        this.targets = trainingTargets;
        this.prepareBackPropagationTaskGraph();
        this.networkBackPropagationTornadoExecutionPlan.execute();
    }

    public void initializeNetwork() {
        networkInitializatonTornadoExecutionPlan.execute();
    }

    public double[] predict(double[] inputs) {
        this.inputs = inputs;
        this.prepareFeedForwardTaskGraph();
        this.networkFeedForwardTornadoExecutionPlan.execute();
        Matrix2DDouble output = this.outputMatrices[this.outputMatrices.length - 1];
        double[] predictions = new double[this.networkLayers[this.networkLayers.length - 1].numberOfNodes()];
        for (int i = 0; i < predictions.length; i++) {
            predictions[i] = output.get(i, 0);
        }
        return predictions;
    }

    public void printData() {
        System.out.println("Network layers: ");
        for (NetworkLayer networkLayer : this.networkLayers)
            System.out.println(networkLayer);
        System.out.println("All Weights Matrices");
        for (int a = 0; a < this.networkLayers.length; a++) {
            System.out.println(this.weightsMatrices[a]);
        }
        System.out.println("All Biases Matrices");
        for (int a = 0; a < this.networkLayers.length; a++) {
            System.out.println(this.biasesMatrices[a]);
        }
    }
}
