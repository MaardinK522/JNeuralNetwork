package com.mkproductions;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.layers.ConvolutionLayer;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.layers.FlattenLayer;
import com.mkproductions.jnn.cpu.layers.PoolingLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;

import com.mkproductions.jnn.gpu.solver.ActivationFunctionSolver;
import com.mkproductions.jnn.gpu.TaskGraphMatrixSolver;
import com.mkproductions.jnn.networks.JSequential;
import com.mkproductions.jnn.networks.JGPUNeuralNetwork;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;
import org.jetbrains.annotations.NotNull;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

import java.util.Arrays;

public class Demo {
    private final static double[][] trainingInputs = new double[][] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, };
    private final static double[][] trainingTargets = new double[][] { { 0 }, { 1 }, { 1 }, { 0 }, };

    // TornadoVM version: 1.0.7
    static void main() {
        //        testAdditionOperationOnDevice();
        //        testMatrixMultiplicationOperationOnDevice();
        //        testingRandomNumberGenerator();
                testingNetworkInitialization();
        //        testingInterfaceFunctions();
//        testingConvolutionNeuralNetwork();
        //        testingTensorOperations();
    }

    private static void testingConvolutionNeuralNetwork() {
        ConvolutionLayer conv1 = new ConvolutionLayer(15, 3, 1, 0, ActivationFunction.RE_LU);
        ConvolutionLayer conv2 = new ConvolutionLayer(2, 3, 1, 0, ActivationFunction.RE_LU);
        PoolingLayer poolingLayer = new PoolingLayer(4, 1, PoolingLayer.PoolingLayerType.MAX);
        FlattenLayer flattenLayer = new FlattenLayer();
        DenseLayer dense = new DenseLayer(10, ActivationFunction.RE_LU);
        int[] inputShape = new int[] { 3, 28, 28 };
        // Step 2: Create the CNN
        JSequential cnn = new JSequential(inputShape, LossFunction.BINARY_CROSS_ENTROPY, JNetworkOptimizer.ADAM, conv2, poolingLayer, conv1, flattenLayer, dense);

        // Step 3: Create a dummy input tensor
        Tensor input = new Tensor(3, 28, 28);
        JSequential.randomize(input);
        // Step 4: Forward propagate
        //        Tensor result = ;
        cnn.forwardPropagation(input);

        // Step 5: Print output
        //        System.out.println(STR."Output shape: \{Arrays.toString(result.getShape())}");
        //        System.out.println(STR."Output data: \{Arrays.toString(result.getData())}");
    }

    private static void testingTensorOperations() {
        Tensor input = new Tensor(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 3, 3);

        Tensor kernel = new Tensor(new double[] { 1, 0, 0, -1 }, 2, 2);

        Tensor result = Tensor.convolve2D(input, kernel, 1, 0);
        System.out.println(result);
    }

    private static void testingInterfaceFunctions() {
        Matrix2DDouble matrix1 = new Matrix2DDouble(5, 5);
        var device = TornadoExecutionPlan.getDevice(1, 0);
        System.out.println(device);
        TaskGraph taskGraph = new TaskGraph("matrixOperation").transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix1);
        TaskGraphMatrixSolver.applyActivationFunction(taskGraph, "activationFunctionEquation", matrix1, ActivationFunction.SIGMOID);
        TaskGraphMatrixSolver.applyActivationFunctionDerivative(taskGraph, "activationFunctionDerivative", matrix1, ActivationFunction.SIGMOID);
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, matrix1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.withDevice(device).execute();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
        System.out.println(matrix1);
    }

    private static void testingNetworkInitialization() {
        JGPUNeuralNetwork jgpuNeuralNetwork = getJgpuNeuralNetwork();
        jgpuNeuralNetwork.setLearningRate(0.5);
        for (double[] trainingInput : trainingInputs) {
            System.out.println(Arrays.toString(jgpuNeuralNetwork.predict(trainingInput)));
        }
        jgpuNeuralNetwork.setLearningRate(0.01);
        for (int i = 0; i < 100; i++) {
            jgpuNeuralNetwork.train(trainingInputs, trainingTargets, 1);
        }
        System.out.println();
        for (double[] trainingInput : trainingInputs) {
            System.out.println(Arrays.toString(jgpuNeuralNetwork.predict(trainingInput)));
        }
    }

    private static @NotNull JGPUNeuralNetwork getJgpuNeuralNetwork() {
        JGPUNeuralNetwork jgpuNeuralNetwork = new JGPUNeuralNetwork(// Network.
                LossFunction.BINARY_CROSS_ENTROPY, // Loss function.
                2, // Number of inputs.
                new DenseLayer(8, ActivationFunction.SIGMOID), // Dense Layer
                new DenseLayer(8, ActivationFunction.SIGMOID), // Dense Layer
                new DenseLayer(1, ActivationFunction.SIGMOID)  // Dense Layer
        );
        jgpuNeuralNetwork.initializeNetwork();
        jgpuNeuralNetwork.printData();
        System.out.println("==========================NETWORK DATA==========================");
        return jgpuNeuralNetwork;
    }

    private static void testingRandomNumberGenerator() {
        Matrix2DDouble matrix1 = new Matrix2DDouble(5, 5);

        var device = TornadoExecutionPlan.getDevice(1, 0);
        System.out.println(device);
        TaskGraph taskGraph = new TaskGraph("matrixOperation").transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix1);
        TaskGraphMatrixSolver.fill(taskGraph, "fillMatrix", 5.0, matrix1);
        TaskGraphMatrixSolver.initializeMatrixWithRandomNumbers(taskGraph, "randomNumbers", matrix1);
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, matrix1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.withDevice(device).execute();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
        System.out.println(matrix1);
    }

    private static void testAdditionOperationOnDevice() {
        Matrix2DDouble matrix1 = new Matrix2DDouble(5, 5);
        Matrix2DDouble matrix2 = new Matrix2DDouble(5, 5);
        Matrix2DDouble matrix3 = new Matrix2DDouble(5, 5);

        var device = TornadoExecutionPlan.getDevice(1, 0);
        System.out.println(device);
        TaskGraph taskGraph = new TaskGraph("matrixOperation").transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix1, matrix2);
        TaskGraphMatrixSolver.fill(taskGraph, "fillMatrix1", 5.0, matrix1);
        TaskGraphMatrixSolver.fill(taskGraph, "fillMatrix2", 5.0, matrix2);
        TaskGraphMatrixSolver.solveAddition(taskGraph, "addition", matrix1, matrix2, matrix3);
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, matrix1, matrix2, matrix3);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.withDevice(device).execute();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
        System.out.println(matrix1);
        System.out.println(matrix2);
        System.out.println(matrix3);
    }

    private static void testMatrixMultiplicationOperationOnDevice() {
        Matrix2DDouble matrix1 = new Matrix2DDouble(5, 5);
        Matrix2DDouble matrix2 = new Matrix2DDouble(5, 10);
        Matrix2DDouble matrix3 = new Matrix2DDouble(5, 10);

        var device = TornadoExecutionPlan.getDevice(1, 0);
        System.out.println(device);
        TaskGraph taskGraph = new TaskGraph("matrixOperation").transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix1, matrix2);
        TaskGraphMatrixSolver.fill(taskGraph, "fillMatrix1", 5.0, matrix1);
        TaskGraphMatrixSolver.fill(taskGraph, "fillMatrix2", 1.0, matrix2);
        TaskGraphMatrixSolver.solveMatrixMultiplication(taskGraph, "matrixMultiplication", matrix1, matrix2, matrix3);
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, matrix1, matrix2, matrix3);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.withDevice(device).execute();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
        System.out.println(matrix1);
        System.out.println(matrix2);
        System.out.println(matrix3);
    }
}