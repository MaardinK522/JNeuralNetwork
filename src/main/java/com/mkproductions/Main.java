package com.mkproductions;


import com.mkproductions.jnn.entity.*;
import com.mkproductions.jnn.graphics.mnist.MNISTFrame;
import com.mkproductions.jnn.graphics.xor.XORFrame;
import com.mkproductions.jnn.network.JNeuralNetwork;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
//        testingXORProblem();
//        testingNetworkTraining();
        testingCSVNetworkTrainingTesting();
//        testingCSVBufferedReader();
    }

    private static void testingCSVNetworkTrainingTesting() {
        MNISTFrame mnistFrame = new MNISTFrame("MNIST testing.");
        mnistFrame.startRendering();
    }

    private static void testingCSVBufferedReader() {
        String dataPath = System.getProperty("user.dir") + "//src//main//resources//com//mkproductions//";

        String testingDataPath = dataPath + "testing_data//mnist_test.csv";
        String trainingDataPath = dataPath + "training_data//mnist_train.csv";

        CSVBufferedReader csvTestingDataBufferedReader = new CSVBufferedReader(testingDataPath);
        CSVBufferedReader csvTrainingDataBufferedReader = new CSVBufferedReader(trainingDataPath);

        List<List<Double>> csvTestingTableData = csvTestingDataBufferedReader.getTable();
        List<Double> csvTestingOutputColumn = csvTestingDataBufferedReader.getColumn("label");

        List<List<Double>> csvTrainingDataTable = csvTrainingDataBufferedReader.getTable();
        List<Double> csvTrainingOutputColumn = csvTrainingDataBufferedReader.getColumn("label");

        double[][] testingInputs = new double[csvTestingTableData.size()][csvTestingTableData.get(0).size()];
        double[][] testingOutputs = new double[csvTestingOutputColumn.size()][10];

        double[][] trainingInputs = new double[csvTrainingDataTable.size()][csvTrainingDataTable.get(0).size()];
        double[][] trainingOutputs = new double[csvTrainingOutputColumn.size()][10];

        // For filtering training data.
        for (int a = 0; a < csvTrainingDataTable.size(); a++) {
            for (int b = 0; b < csvTrainingDataTable.get(a).size(); b++) {
                trainingInputs[a][b] = Mapper.mapRangeToRange(csvTrainingDataTable.get(a).get(b), 0, 255, 0, 1);
            }
        }
        // Converting outputs into raw arrays.
        for (int a = 0; a < trainingOutputs.length; a++) {
            int outputIndex = csvTrainingOutputColumn.get(a).intValue();
            trainingOutputs[a][outputIndex] = 1;
        }
        // For filtering the testing data.
        for (int a = 0; a < csvTestingTableData.size(); a++) {
            for (int b = 0; b < csvTestingTableData.get(a).size(); b++) {
                testingInputs[a][b] = Mapper.mapRangeToRange(csvTestingTableData.get(a).get(b), 0, 255, 0, 1);
            }
        }
//        // Converting outputs into raw arrays.
        for (int a = 0; a < csvTestingOutputColumn.size(); a++) {
            int outputIndex = csvTestingOutputColumn.get(a).intValue();
            testingOutputs[a][outputIndex] = 1;
        }

        JNeuralNetwork jNeuralNetwork = getjNeuralNetwork();
        int epochs = 1000000;
        int outputIndex = 1;

        System.out.println("Actual output: " + Arrays.toString(testingOutputs[outputIndex]));
        System.out.println("Before training: " + Arrays.toString(jNeuralNetwork.processInputs(testingInputs[outputIndex])));

        jNeuralNetwork.train(trainingInputs, trainingOutputs, epochs);

        System.out.println("After training for " + epochs + " times.");
        System.out.println("After training: " + Arrays.toString(jNeuralNetwork.processInputs(testingInputs[outputIndex])));
        System.out.println("Actual output: " + Arrays.toString(testingOutputs[outputIndex]));
    }

    @NotNull
    private static JNeuralNetwork getjNeuralNetwork() {
        Layer[] layers = new Layer[]{
                new Layer(50, ActivationFunction.SIGMOID),
                new Layer(50, ActivationFunction.SIGMOID),
                new Layer(10, ActivationFunction.SIGMOID),
        };
        JNeuralNetwork jNeuralNetwork = new JNeuralNetwork(
                784,
                layers
        );
        jNeuralNetwork.setLearningRate(0.01);
        return jNeuralNetwork;
    }

    public static void testingXORProblem() {
        XORFrame graphicsFrame = new XORFrame("Main Graphics");
        graphicsFrame.startRendering();
    }

    public static void testingNetworkTraining() {
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] trainingTargets = {{0}, {1}, {1}, {0}};
        JNeuralNetwork jnn = new JNeuralNetwork(2, new Layer(4, ActivationFunction.SIGMOID), new Layer(1, ActivationFunction.RE_LU));
        int epochs = 1000;
//        double[] testingInputs = new double[]{0, 0};
        jnn.setLearningRate(0.01);
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
