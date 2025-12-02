package com.mkproductions;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.networks.JGPUNeuralNetwork;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;
import com.mkproductions.jnn.graphics.mnist.MNISTFrame;
import com.mkproductions.jnn.graphics.training_view.NeuralNetworkTrainingViewerJFrame;
import com.mkproductions.jnn.graphics.xor.XORFrame;
import com.mkproductions.jnn.networks.JNeuralNetwork;
import org.jetbrains.annotations.NotNull;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

public class Main {
    private static final double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private static final double[][] trainingOutputs = {{0}, {1}, {1}, {0}};

    static void main() {
        //        testingXORProblem();
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

    private static void renderNetwork() {
        JNeuralNetwork jNeuralNetwork = new JNeuralNetwork(LossFunction.MEAN_SQUARED_ERROR, JNetworkOptimizer.SGD, 2, new DenseLayer(4, ActivationFunction.SIGMOID), new DenseLayer(4, ActivationFunction.SIGMOID), new DenseLayer(1, ActivationFunction.SIGMOID));
        new NeuralNetworkTrainingViewerJFrame(jNeuralNetwork, trainingInputs, trainingOutputs).startRendering();
    }

    public static void testingXORProblem() {
        XORFrame graphicsFrame = new XORFrame("Main Graphics");
        graphicsFrame.startRendering();
    }

    private static void testingOnGPU() {
        TornadoDevice device = TornadoExecutionPlan.getDevice(0, 0);
        System.out.println(STR."Device: \{device}");
        JGPUNeuralNetwork jgpuNeuralNetwork = new JGPUNeuralNetwork(LossFunction.MEAN_SQUARED_ERROR, 2, new DenseLayer(4, ActivationFunction.SIGMOID), new DenseLayer(4, ActivationFunction.SIGMOID), new DenseLayer(1, ActivationFunction.SIGMOID));
    }
}