package com.mkproductions;

import com.mkproductions.jnn.cpu.*;
import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Layer;
import com.mkproductions.jnn.cpu.entity.Mapper;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNeuralNetworkOptimizer;
import com.mkproductions.jnn.graphics.mnist.MNISTFrame;
import com.mkproductions.jnn.graphics.training_view.NeuralNetworkTrainingViewerJFrame;
import com.mkproductions.jnn.graphics.xor.XORFrame;
import com.mkproductions.jnn.network.JNeuralNetwork;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;

public class Main {
    private static final double[][] trainingInputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    private static final double[][] trainingOutputs = { { 0 }, { 1 }, { 1 }, { 0 } };

    public static void main(String[] args) {
        testingXORProblem();
//        testingNetworkTraining();
        //        testingMNISTCSVTrainingTesting();
        //        testingCSVBufferedReader();
        //        performingConvolution();
        //        renderNetwork();
    }

    private static void renderNetwork() {
        JNeuralNetwork jNeuralNetwork = new JNeuralNetwork(LossFunction.MEAN_SQUARED_ERROR, JNeuralNetworkOptimizer.SGD_MOMENTUM, 2, new Layer(4, ActivationFunction.SIGMOID),
                new Layer(4, ActivationFunction.SIGMOID), new Layer(1, ActivationFunction.SIGMOID));
        new NeuralNetworkTrainingViewerJFrame(jNeuralNetwork, trainingInputs, trainingOutputs).startRendering();
    }
    //
    //    private static void performingConvolution() {
    //        double[][] imageData = new double[][]{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1},};
    //        double[][] filterData = new double[][]{{-1, 0, 0.5}, {0, 0.1, 0}, {0.5, 0, -1},};
    //        Matrix image = new Matrix(imageData);
    //        Matrix filter = new Matrix(filterData);
    //        image.printMatrix();
    ////        Matrix.convolute(image, filter).printMatrix();
    //    }

    private static void testingMNISTCSVTrainingTesting() {
        MNISTFrame mnistFrame = new MNISTFrame("MNIST testing.");
        mnistFrame.startRendering();
    }

    private static void testingCSVBufferedReader() {
        String dataPath = STR."\{System.getProperty("user.dir")}//src//main//resources//com//mkproductions//";

        String testingDataPath = STR."\{dataPath}testing_data//mnist_test.csv";
        String trainingDataPath = STR."\{dataPath}training_data//mnist_train.csv";

        CSVBufferedReader csvTestingDataBufferedReader = new CSVBufferedReader(testingDataPath);
        CSVBufferedReader csvTrainingDataBufferedReader = new CSVBufferedReader(trainingDataPath);

        List<List<Double>> csvTestingTableData = csvTestingDataBufferedReader.getTable();
        var csvTestingOutputColumn = csvTestingDataBufferedReader.getColumn("label");

        List<List<Double>> csvTrainingDataTable = csvTrainingDataBufferedReader.getTable();
        var csvTrainingOutputColumn = csvTrainingDataBufferedReader.getColumn("label");

        double[][] testingInputs = new double[csvTestingTableData.size()][csvTestingTableData.getFirst().size()];
        double[][] testingOutputs = new double[csvTestingOutputColumn.size()][10];

        double[][] trainingInputs = new double[csvTrainingDataTable.size()][csvTrainingDataTable.getFirst().size()];
        double[][] trainingOutputs = new double[csvTrainingOutputColumn.size()][10];

        // For filtering training data.
        for (int a = 0; a < csvTrainingDataTable.size(); a++) {
            for (int b = 0; b < csvTrainingDataTable.get(a).size(); b++) {
                trainingInputs[a][b] = Mapper.mapRangeToRange(csvTrainingDataTable.get(a).get(b), 0, 255, 0, 1);
            }
        }
        // Converting outputs into raw arrays.
        for (int a = 0; a < trainingOutputs.length; a++) {
            int outputIndex = csvTrainingOutputColumn.get(a);
            trainingOutputs[a][outputIndex] = 1;
        }
        // For filtering the testing data.
        for (int a = 0; a < csvTestingTableData.size(); a++) {
            for (int b = 0; b < csvTestingTableData.get(a).size(); b++) {
                testingInputs[a][b] = Mapper.mapRangeToRange(csvTestingTableData.get(a).get(b), 0, 255, 0, 1);
            }
        }
        // Converting outputs into raw arrays.
        for (int a = 0; a < csvTestingOutputColumn.size(); a++) {
            int outputIndex = csvTestingOutputColumn.get(a);
            testingOutputs[a][outputIndex] = 1;
        }

        JNeuralNetwork jNeuralNetwork = getjNeuralNetwork();
        int epochs = 1000000;
        int outputIndex = 1;

        System.out.println(STR."Actual output: \{Arrays.toString(testingOutputs[outputIndex])}");
        System.out.println(STR."Before training: \{Arrays.toString(jNeuralNetwork.processInputs(testingInputs[outputIndex]))}");

        jNeuralNetwork.train(trainingInputs, trainingOutputs, epochs);

        System.out.println(STR."After training for \{epochs} times.");
        System.out.println(STR."After training: \{Arrays.toString(jNeuralNetwork.processInputs(testingInputs[outputIndex]))}");
        System.out.println(STR."Actual output: \{Arrays.toString(testingOutputs[outputIndex])}");
    }

    @NotNull
    private static JNeuralNetwork getjNeuralNetwork() {
        Layer[] layers = new Layer[] { new Layer(32, ActivationFunction.SIGMOID), new Layer(64, ActivationFunction.SIGMOID), new Layer(128, ActivationFunction.SIGMOID),
                new Layer(10, ActivationFunction.SIGMOID) };
        JNeuralNetwork jNeuralNetwork = new JNeuralNetwork(LossFunction.LOG_COSH, JNeuralNetworkOptimizer.SGD_MOMENTUM, 784, layers);
        jNeuralNetwork.setLearningRate(0.01F);
        return jNeuralNetwork;
    }

    public static void testingXORProblem() {
        XORFrame graphicsFrame = new XORFrame("Main Graphics");
        graphicsFrame.startRendering();
    }

    public static void testingNetworkTraining() {
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        double[][] trainingInputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        double[][] trainingTargets = { { 0 }, { 1 }, { 1 }, { 0 } };
        JNeuralNetwork jnn = new JNeuralNetwork(LossFunction.LOG_COSH, JNeuralNetworkOptimizer.SGD_MOMENTUM, 2, new Layer(4, ActivationFunction.SIGMOID), new Layer(1, ActivationFunction.RE_LU));
        int epochs = 1000;
        //        double[] testingInputs = new double[]{0, 0};
        jnn.setLearningRate(0.01F);
        try {
            System.out.println("Network output: ");
            System.out.println(STR."Top left corner: \{Arrays.toString(jnn.processInputs(new double[] { 0, 0 }))}");
            System.out.println(STR."Top right corner: \{Arrays.toString(jnn.processInputs(new double[] { 0, 1 }))}");
            System.out.println(STR."Bottom left corner: \{Arrays.toString(jnn.processInputs(new double[] { 1, 0 }))}");
            System.out.println(STR."Bottom right corner: \{Arrays.toString(jnn.processInputs(new double[] { 1, 1 }))}");

            jnn.train(trainingInputs, trainingTargets, epochs);

            System.out.println(STR."After training for \{epochs} times");

            System.out.println("Network output: ");
            System.out.println(STR."Top left corner: \{Arrays.toString(jnn.processInputs(new double[] { 0, 0 }))}");
            System.out.println(STR."Top right corner: \{Arrays.toString(jnn.processInputs(new double[] { 0, 1 }))}");
            System.out.println(STR."Bottom left corner: \{Arrays.toString(jnn.processInputs(new double[] { 1, 0 }))}");
            System.out.println(STR."Bottom right corner: \{Arrays.toString(jnn.processInputs(new double[] { 1, 1 }))}");

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
