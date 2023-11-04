package com.mkproductions;


import com.mkproductions.jnn.entity.Matrix;
import com.mkproductions.jnn.graphics.MyFrame;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        MyFrame graphicsFrame = new MyFrame("Main Graphics");
        graphicsFrame.setFrameTimer(new Timer(1000 / 60, e -> graphicsFrame.repaint()));
        try {
            graphicsFrame.startRendering();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//        double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
//        double[][] trainingTargets = {
//                {0},
//                {1},
//                {1},
//                {0}
//        };
//        JNeuralNetwork jnn = new JNeuralNetwork(
//                2,
//                new Layer(4, ActivationFunction.SIGMOID),
//                new Layer(1, ActivationFunction.SIGMOID)
//        );
//        int epochs = 100;
//        double[] testingInputs = new double[]{0, 0};
//        jnn.setLearningRate(0.001);
//        try {
//            System.out.println("Network output: " + Arrays.toString(jnn.processInputs(testingInputs)));
//            jnn.train(trainingInputs, trainingTargets, epochs);
//            System.out.println("After training for " + epochs + " times");
//            System.out.println(Arrays.toString(jnn.processInputs(testingInputs)));
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
    }
}
