package com.mkproductions;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNeuralNetworkOptimizer;
import com.mkproductions.jnn.graphics.mnist.MNISTFrame;
import com.mkproductions.jnn.graphics.training_view.NeuralNetworkTrainingViewerJFrame;
import com.mkproductions.jnn.graphics.xor.XORFrame;
import com.mkproductions.jnn.networks.JNeuralNetwork;
import org.jetbrains.annotations.NotNull;

public class Main {
    private static final double[][] trainingInputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    private static final double[][] trainingOutputs = { { 0 }, { 1 }, { 1 }, { 0 } };

    public static void main(String[] args) {
        //        testingXORProblem();
        //        testingMNISTCSVTrainingTesting();
        //        renderNetwork();
    }

    private static void testingMNISTCSVTrainingTesting() {
        MNISTFrame mnistFrame = new MNISTFrame("MNIST testing.");
        try {
            mnistFrame.startRendering();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @NotNull
    private static JNeuralNetwork getjNeuralNetwork() {
        DenseLayer[] denseLayers = new DenseLayer[] { new DenseLayer(32, ActivationFunction.SIGMOID), new DenseLayer(64, ActivationFunction.SIGMOID), new DenseLayer(128, ActivationFunction.SIGMOID),
                new DenseLayer(10, ActivationFunction.SOFTMAX) };
        JNeuralNetwork jNeuralNetwork = new JNeuralNetwork(LossFunction.ABSOLUTE_ERROR, JNeuralNetworkOptimizer.SGD_MOMENTUM, 28 * 28, denseLayers);
        jNeuralNetwork.setLearningRate(0.01F);
        return jNeuralNetwork;
    }

    public static void testingXORProblem() {
        XORFrame graphicsFrame = new XORFrame("Main Graphics");
        graphicsFrame.startRendering();
    }

    private static void renderNetwork() {
        JNeuralNetwork jNeuralNetwork = new JNeuralNetwork(LossFunction.MEAN_SQUARED_ERROR, JNeuralNetworkOptimizer.SGD_MOMENTUM, 2, new DenseLayer(4, ActivationFunction.SIGMOID),
                new DenseLayer(4, ActivationFunction.SIGMOID), new DenseLayer(1, ActivationFunction.SIGMOID));
        new NeuralNetworkTrainingViewerJFrame(jNeuralNetwork, trainingInputs, trainingOutputs).startRendering();
    }
}