package com.mkproductions;


import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.graphics.MyFrame;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        testingXORProblem();
//        testingNetworkTraining();
    }

    public static void testingXORProblem() {
        MyFrame graphicsFrame = new MyFrame("Main Graphics");
        graphicsFrame.startRendering();
    }

    public static void testingNetworkTraining() {
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] trainingTargets = {
                {0},
                {1},
                {1},
                {0}
        };
        JNeuralNetwork jnn = new JNeuralNetwork(
                2,
                new Layer(4, ActivationFunction.SIGMOID),
                new Layer(1, ActivationFunction.RE_LU)
        );
        int epochs = 1000;
//        double[] testingInputs = new double[]{0, 0};
        jnn.setLearningRate(0.1);
        try {
            System.out.println("Network output: ");
            System.out.println("Top left corner: " + Arrays.toString(jnn.processInputs(new double[]{0, 0})));
            System.out.println("Top right corner: " + Arrays.toString(jnn.processInputs(new double[]{0, 1})));
            System.out.println("Bottom left corner: " + Arrays.toString(jnn.processInputs(new double[]{1, 0})));
            System.out.println("Bottom right corner: " + Arrays.toString(jnn.processInputs(new double[]{1, 1})));

            jnn.train(trainingInputs, trainingTargets, epochs);

            System.out.println("After training for " + epochs + " times");

            System.out.println("Network output: ");
            System.out.println("Top left corner: " + Arrays.toString(jnn.processInputs(new double[]{0, 0})));
            System.out.println("Top right corner: " + Arrays.toString(jnn.processInputs(new double[]{0, 1})));
            System.out.println("Bottom left corner: " + Arrays.toString(jnn.processInputs(new double[]{1, 0})));
            System.out.println("Bottom right corner: " + Arrays.toString(jnn.processInputs(new double[]{1, 1})));

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
