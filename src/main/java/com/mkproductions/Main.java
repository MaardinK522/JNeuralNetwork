package com.mkproductions;

import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.network.JNeuralNetwork;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        try {
            process();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void process() throws Exception {
        double[] input = {1, 0, 1, 0};
        double[] output;
        double[][] trainingInputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[][] trainingTargets = {{0}, {1}, {1}, {0}};

        JNeuralNetwork jnn = new JNeuralNetwork(
                0.0001,
                input.length,
                new Layer(5, ActivationFunction.RE_LU),
                new Layer(5, ActivationFunction.RE_LU),
                new Layer(4, ActivationFunction.RE_LU)
        );

        System.out.println("JNN");
        jnn.printNetwork();
//        System.out.println("TNN");
//        // Processing before training
        output = jnn.processInputs(input);
        System.out.println("Network output: " + Arrays.toString(output));
//
//        // Training
//        jnn.train(trainingInputs, trainingTargets, 1000);
//        System.out.println("After training: ");
//
//        // Processing after training.
//        output = jnn.processInputs(input);
//        System.out.println("Network output: " + Arrays.toString(output));
    }

    private static double map(double value, double fromMin, double fromMax, double toMin, double toMax) {
        return toMin + (value - fromMin) * (toMax - toMin) / (fromMax - fromMin);
    }
}
