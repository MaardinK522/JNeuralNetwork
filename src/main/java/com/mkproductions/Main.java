package com.mkproductions;

import com.mkproductions.jnn.graphics.mnist.MNISTFrame;
import com.mkproductions.jnn.graphics.xor.XORFrame;

public class Main {
    private static final double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private static final double[][] trainingOutputs = {{0}, {1}, {1}, {0}};

    static void main() {
        testingXORProblem();
        //        testingMNISTCSVTrainingTesting();
        //        renderNetwork();
//        testingOnGPU();
    }

    private static void testingMNISTCSVTrainingTesting() {
        MNISTFrame mnistFrame = new MNISTFrame("MNIST testing.");
        try {
            mnistFrame.startRendering();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static void testingXORProblem() {
        XORFrame graphicsFrame = new XORFrame("Main Graphics");
        graphicsFrame.startRendering();
    }

    private static void testingOnGPU() {
//        TornadoDevice device = TornadoExecutionPlan.getDevice(0, 0);
//        System.out.println(STR."Device: \{device}");
//        JGPUNeuralNetwork jgpuNeuralNetwork = new JGPUNeuralNetwork(LossFunction.MEAN_SQUARED_ERROR, 2, new DenseLayer(4, ActivationFunction.SIGMOID), new DenseLayer(4, ActivationFunction.SIGMOID), new DenseLayer(1, ActivationFunction.SIGMOID));
    }
}