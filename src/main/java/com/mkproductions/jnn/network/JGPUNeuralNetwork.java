package com.mkproductions.jnn.network;

import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.entity.solvers.TaskGraphMatrixSolver;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

public class JGPUNeuralNetwork {
    private final int numberOfInputsNode;
    private final Layer[] networkLayers;
    private final Matrix2DDouble[] weightsMatrices;
    private final Matrix2DDouble[] biasesMatrices;
    private final Matrix2DDouble[] outputMatrices;
    private double learningRate;
    private TornadoExecutionPlan networkInitializatonTornadoExecutionPlan;
    private TornadoExecutionPlan networkFeedForwardTornadoExecutionPlan;
    private double[] inputs;

    public JGPUNeuralNetwork(int numberOfInputsNode, Layer... networkLayers) {
        this.numberOfInputsNode = numberOfInputsNode;
        this.networkLayers = networkLayers;
        this.weightsMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.biasesMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.outputMatrices = new Matrix2DDouble[this.networkLayers.length];
        this.inputs = new double[this.numberOfInputsNode];
        this.learningRate = 0.01;
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            if (layerIndex == 0) {
                this.weightsMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].numberOfNodes(), this.numberOfInputsNode);
            } else {
                this.weightsMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].numberOfNodes(), this.networkLayers[layerIndex - 1].numberOfNodes());
            }
            this.biasesMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].numberOfNodes(), 1);
            this.outputMatrices[layerIndex] = new Matrix2DDouble(this.networkLayers[layerIndex].numberOfNodes(), 1);
        }
        this.initializeRandomNetwork();
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
        }
        // Reverting required outputs of all the layers.
        for (Matrix2DDouble outputMatrix : outputMatrices) {
            feedForwardTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, outputMatrix);
        }

        this.networkFeedForwardTornadoExecutionPlan = new TornadoExecutionPlan(feedForwardTaskGraph.snapshot());
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
        for (Layer networkLayer : this.networkLayers)
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
