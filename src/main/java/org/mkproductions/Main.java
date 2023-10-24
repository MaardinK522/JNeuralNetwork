package org.mkproductions;

import org.mkproductions.jnn.network.JNeuralNetwork;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {

        double[] input = {0.23, 0.976, 0.3974};
        int[] schema = {2};
        double[] output = new double[1];

        double[] target = {0};
        JNeuralNetwork jnn = new JNeuralNetwork(input.length, schema, output.length);
        try {
            System.out.println("Inputs: " + Arrays.toString(input));
            System.out.println("Outputs: " + Arrays.toString(output));

            // Calculating the output
            output = jnn.processInputs(input);
            System.out.println("Network output: " + Arrays.toString(output));
            // Trying to train the network.
            jnn.train(new double[][]{input}, new double[][]{target}, 100);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    private static double map(double value, double fromMin, double fromMax, double toMin, double toMax) {
        return toMin + (value - fromMin) * (toMax - toMin) / (fromMax - fromMin);
    }
}
