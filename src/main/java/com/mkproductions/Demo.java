package com.mkproductions;

import com.mkproductions.jnn.entity.NetworkLayer;

import com.mkproductions.jnn.entity.activationFunctions.ActivationFunctionManager;
import com.mkproductions.jnn.entity.lossFunctions.LossFunctionManager;
import com.mkproductions.jnn.entity.solvers.TaskGraphMatrixSolver;
import com.mkproductions.jnn.network.JGPUNeuralNetwork;
import uk.ac.manchester.tornado.api.*;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

import java.util.Arrays;

public class Demo {
    // TornadoVM version: 1.0.7
    public static void main() {
//        testAdditionOperationOnDevice();
//        testMatrixMultiplicationOperationOnDevice();
//        testingRandomNumberGenerator();
        testingNetworkInitialization();
//        testingInterfaceFunctions();
    }

    private static void testingInterfaceFunctions() {
        Matrix2DDouble matrix1 = new Matrix2DDouble(5, 5);
        var device = TornadoExecutionPlan.getDevice(1, 0);
        System.out.println(device);
        TaskGraph taskGraph = new TaskGraph("matrixOperation").transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix1);
        TaskGraphMatrixSolver.applyActivationFunction(taskGraph, "activationFunctionEquation", matrix1, ActivationFunctionManager.NetworkActivationFunction.SIGMOID);
        TaskGraphMatrixSolver.applyActivationFunctionDerivative(taskGraph, "activationFunctionDerivative", matrix1, ActivationFunctionManager.NetworkActivationFunction.SIGMOID);
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, matrix1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(taskGraph.snapshot())) {
            plan.withDevice(device).execute();
        } catch (TornadoExecutionPlanException e) {
            throw new RuntimeException(e);
        }
        System.out.println(matrix1);
    }

    private static void testingNetworkInitialization() {
        JGPUNeuralNetwork jgpuNeuralNetwork = new JGPUNeuralNetwork(
                LossFunctionManager.LossFunction.BINARY_CROSS_ENTROPY,
                2,
                new NetworkLayer(5, ActivationFunctionManager.NetworkActivationFunction.SIGMOID),
                new NetworkLayer(2, ActivationFunctionManager.NetworkActivationFunction.SIGMOID),
                new NetworkLayer(10, ActivationFunctionManager.NetworkActivationFunction.LINEAR)
        );
        jgpuNeuralNetwork.initializeNetwork();
        jgpuNeuralNetwork.printData();
        System.out.println(Arrays.toString(jgpuNeuralNetwork.predict(new double[]{0.1, 0.1})));
    }

    private static void testingRandomNumberGenerator() {
        Matrix2DDouble matrix1 = new Matrix2DDouble(5, 5);

        var device = TornadoExecutionPlan.getDevice(1, 0);
        System.out.println(device);
        TaskGraph taskGraph = new TaskGraph("matrixOperation").transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix1);
        TaskGraphMatrixSolver.fill(taskGraph, "fillMatrix", 5.0, matrix1);
        TaskGraphMatrixSolver.initializeMatrixWithRandomNumbers(taskGraph, "randomNumbers", matrix1, System.nanoTime());
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
