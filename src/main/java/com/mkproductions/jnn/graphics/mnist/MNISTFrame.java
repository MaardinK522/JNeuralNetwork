package com.mkproductions.jnn.graphics.mnist;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.CSVBufferedReader;
import com.mkproductions.jnn.cpu.entity.Layer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNeuralNetworkOptimizer;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.List;

public class MNISTFrame extends JFrame {
    private static final double[][] dataGrid = new double[28][28];
    public static double networkAccuracy;
    int w = 560;
    int h = 560;
    private final MNISTTestingJPanel mnistTestingJPanel;
    private final MNISTNetworkJPanel mnistNetworkJPanel;
    private boolean running = false;
    private final double[][] trainingInputs;
    private final double[][] trainingOutputs;
    public static double[][] testingInputs;
    public static double[][] testingOutputs;
    private static JNeuralNetwork jNeuralNetwork;
    public static Layer[] networkLayers;

    public MNISTFrame(String frameName) {
        // Initializing network
        networkLayers = new Layer[]{
                new Layer(128, ActivationFunction.RE_LU),
                new Layer(10, ActivationFunction.SOFTMAX),
        };
        this.restartNetwork();
        // Declaring size of the inputs & outputs.
        double[][][] trainingTestingData = this.prepareTrainingTestingDataSet();
        this.trainingInputs = trainingTestingData[0];
        this.trainingOutputs = trainingTestingData[1];
        testingInputs = trainingTestingData[2];
        testingOutputs = trainingTestingData[3];

        this.mnistTestingJPanel = new MNISTTestingJPanel(this.w / 2, this.h);
        this.mnistNetworkJPanel = new MNISTNetworkJPanel(this.w / 2, this.h);

        add(this.mnistNetworkJPanel, BorderLayout.EAST);
        add(this.mnistTestingJPanel, BorderLayout.WEST);
        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                super.keyPressed(e);
                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) running = false;
                if (e.getKeyCode() == KeyEvent.VK_T) triggerNetworkTraining();
                if (e.getKeyCode() == KeyEvent.VK_C) clearGridData();
                if (e.getKeyCode() == KeyEvent.VK_R) restartNetwork();
            }
        });
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowOpened(WindowEvent e) {
                super.windowOpened(e);
                System.out.println(frameName + " has started.");
            }

            @Override
            public void windowClosed(WindowEvent e) {
                super.windowClosed(e);
                System.out.println(frameName + " has ended.");
            }
        });
        setLayout(new GridLayout(1, 2));
        setTitle(frameName);
        setSize(this.w, this.h);
        setLocationRelativeTo(null);
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    public static double[] processNetworkInputs(double[] inputs) {
        return jNeuralNetwork.processInputs(inputs);
    }

    public static double[][] getDataGrid() {
        return dataGrid;
    }

    private void restartNetwork() {
        jNeuralNetwork = new JNeuralNetwork(
                LossFunction.MEAN_SQUARED_ERROR,
                JNeuralNetworkOptimizer.RMS_PROP,
                28 * 28,
                networkLayers
        );
        jNeuralNetwork.setLearningRate(0.001);
        jNeuralNetwork.setMomentumFactorBeta1(0.9);
        jNeuralNetwork.setMomentumFactorBeta1(0.998);
        jNeuralNetwork.setDebugMode(true);
    }

    public void startRendering() {
        this.running = true;
        while (this.running) {
            this.mnistTestingJPanel.repaint();
            this.mnistNetworkJPanel.repaint();
            try {
                Thread.sleep(1000 / 60);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        dispose();
        System.exit(-200);
    }

    private double[][][] prepareTrainingTestingDataSet() {
        System.out.println("Please! Wait for preparing training data...");
        String dataPath = System.getProperty("user.dir") + "//src//main//resources//com//mkproductions//";

        String trainingDataPath = dataPath + "training_data//mnist_train.csv";
        String testingDataPath = dataPath + "testing_data//mnist_test.csv";

        CSVBufferedReader csvTrainingDataBufferedReader = new CSVBufferedReader(trainingDataPath);
        CSVBufferedReader csvTestingDataBufferedReader = new CSVBufferedReader(testingDataPath);

        List<List<Double>> csvTrainingDataTable = csvTrainingDataBufferedReader.getTable();
        var csvTrainingOutputColumn = csvTrainingDataBufferedReader.getColumn("label");
        List<List<Double>> csvTestingDataTable = csvTestingDataBufferedReader.getTable();
        var csvTestingOutputColumn = csvTestingDataBufferedReader.getColumn("label");

        double[][] trainingInputs = new double[csvTrainingDataTable.size()][csvTrainingDataTable.get(0).size()];
        double[][] trainingOutputs = new double[csvTrainingOutputColumn.size()][10];
        double[][] testingInputs = new double[csvTestingDataTable.size()][csvTestingDataTable.get(0).size()];
        double[][] testingOutputs = new double[csvTestingOutputColumn.size()][10];

        // For filtering training data.
        for (int a = 0; a < trainingInputs.length; a++) {
            for (int b = 0; b < trainingInputs[0].length; b++) {
                double value = csvTrainingDataTable.get(a).get(b);
                trainingInputs[a][b] = value;//Mapper.mapRangeToRange(value, 0, 255, 0, 1);
            }
        }
        // Converting training outputs into raw arrays.
        for (int a = 0; a < trainingOutputs.length; a++) {
            trainingOutputs[a][0] = csvTrainingOutputColumn.get(a);
        }
        // For filtering testing data.
        for (int a = 0; a < testingInputs.length; a++) {
            for (int b = 0; b < testingInputs[0].length; b++) {
                testingInputs[a][b] = csvTestingDataTable.get(a).get(b); //Mapper.mapRangeToRange(csvTestingDataTable.get(a).get(b), 0, 255, 0, 1);
            }
        }
        // Converting testing outputs into raw outputs.
        for (int a = 0; a < testingOutputs.length; a++) {
            testingOutputs[a][0] = csvTestingOutputColumn.get(a);
        }
        System.out.println("Thanks for waiting!!!");
        return new double[][][]{trainingInputs, trainingOutputs, testingInputs, testingOutputs,};
    }

    void triggerNetworkTraining() {
        int epochs = 10;
        MNISTFrame.jNeuralNetwork.train(this.trainingInputs, this.trainingOutputs, epochs);
        networkAccuracy = MNISTFrame.jNeuralNetwork.calculateAccuracy(MNISTFrame.testingInputs, MNISTFrame.testingOutputs);
    }

    public void clearGridData() {
        for (int a = 0; a < dataGrid.length; a++) {
            for (int b = 0; b < dataGrid[0].length; b++) {
                dataGrid[a][b] = 0;
            }
        }
    }
}
