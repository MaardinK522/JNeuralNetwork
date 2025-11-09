package com.mkproductions.jnn.graphics.mnist;

import com.google.gson.ExclusionStrategy;
import com.google.gson.FieldAttributes;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.CSVBufferedReader;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.entity.LossFunctionAble;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;
import com.mkproductions.jnn.networks.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MNISTFrame extends JFrame {
    private static int epochCount = 10;
    final int w = 1000;
    final int h = 560;
    private final MNISTDrawningJPanel mnistDrawningJPanel;
    private final MNISTNetworkOutputJPanel mnistNetworkOutputJPanel;
    private boolean running;
    private final double[][] trainingInputs;
    private final double[][] trainingOutputs;
    private final double[][] testingInputs;
    private final double[][] testingOutputs;
    private static final int[][] dataGrid = new int[28][28];
    private static JNeuralNetwork jNeuralNetwork;
    private final SaveNeuralNetworkDialog saveNeuralNetworkDialog;
    public static boolean autoTrainNetwork = false;
    public static double networkAccuracy = 0.0;
    public static DenseLayer[] networkDenseLayers;

    public MNISTFrame(String frameName) {
        this.running = false;
        // Declaring the size of the inputs and outputs.
        double[][][] trainingTestingData = this.prepareTrainingTestingDataSet();
        this.trainingInputs = trainingTestingData[0];
        this.trainingOutputs = trainingTestingData[1];
        this.testingInputs = trainingTestingData[2];
        this.testingOutputs = trainingTestingData[3];
        this.restartNetwork();
        this.saveNeuralNetworkDialog = new SaveNeuralNetworkDialog(this, this::saveJNeuralNetwork);
        saveNeuralNetworkDialog.setLocationRelativeTo(this);
        this.mnistDrawningJPanel = new MNISTDrawningJPanel(this.w / 2, this.h);
        this.mnistNetworkOutputJPanel = new MNISTNetworkOutputJPanel(this.w / 2, this.h / 2);
        MNISTNetworkSettingsJPanel mnistNetworkSettingsJPanel = new MNISTNetworkSettingsJPanel(this.w / 2, this.h / 2, _ -> {
            clearGridData();
            return null;
        }, _ -> {
            triggerNetworkTraining();
            return null;
        }, _ -> {
            restartNetwork();
            return null;
        }, _ -> {
            mnistNetworkOutputJPanel.triggerNetworkPrediction();
            return null;
        }, _ -> {
            saveNeuralNetworkDialog.setVisible(true);
            return null;
        });
        JPanel leftPanel = new JPanel(new GridLayout(2, 1));
        leftPanel.add(this.mnistNetworkOutputJPanel);
        leftPanel.add(mnistNetworkSettingsJPanel);
        add(leftPanel);
        add(this.mnistDrawningJPanel);
        //        addKeyListener(new KeyAdapter() {
        //            @Override
        //            public void keyPressed(KeyEvent e) {
        //                super.keyPressed(e);
        //                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
        //                    running = false;
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_T) {
        //                    triggerNetworkTraining();
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_C) {
        //                    clearGridData();
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_R) {
        //                    restartNetwork();
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_UP) {
        //                    jNeuralNetwork.setEpochCount(jNeuralNetwork.getEpochCount() + 1);
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_M) {
        //                    printSingleTrainingEntry();
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_DOWN && jNeuralNetwork.getEpochCount() > 1) {
        //                    jNeuralNetwork.setEpochCount(jNeuralNetwork.getEpochCount() - 1);
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_LEFT && jNeuralNetwork.getLearningRate() > 0.0) {
        //                    jNeuralNetwork.setLearningRate(jNeuralNetwork.getLearningRate() - 0.0001);
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_RIGHT && jNeuralNetwork.getLearningRate() < 1.0) {
        //                    jNeuralNetwork.setLearningRate(jNeuralNetwork.getLearningRate() + 0.0001);
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_SPACE) {
        //                    trainNetwork = !trainNetwork;
        //                }
        //                if (e.getKeyCode() == KeyEvent.VK_ENTER) {
        //                    mnistNetworkOutputJPanel.triggerNetworkPrediction();
        //                }
        //                if (e.isControlDown() && e.getKeyCode() == KeyEvent.VK_S) {
        //                    saveNeuralNetworkDialog.setVisible(true);
        //                }
        //            }
        //        });
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowOpened(WindowEvent e) {
                super.windowOpened(e);
                System.out.println(STR."\{frameName}has started.");
            }

            @Override
            public void windowClosed(WindowEvent e) {
                super.windowClosed(e);
                System.out.println(STR."\{frameName} has ended.");
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

    public static int[][] getDataGrid() {
        return dataGrid;
    }

    public static JNeuralNetwork getJNeuralNetwork() {
        return jNeuralNetwork;
    }

    public static double getNetworkAccuracy() {
        return networkAccuracy;
    }

    public static LossFunctionAble getNetworkLoss() {
        return jNeuralNetwork.getLossFunctionable();
    }

    public static void setEpochCount(int epochCount) {
        MNISTFrame.epochCount = epochCount;
    }

    public static int getTrainingEpochs() {
        return epochCount;
    }

    public static double getLearningRate() {
        return jNeuralNetwork.getLearningRate();
    }

    public static JNetworkOptimizer getJNeuralNetworkOptimizer() {
        return jNeuralNetwork.getjNeuralNetworkOptimizer();
    }

    public static Boolean getTrainNetworkStatus() {
        return autoTrainNetwork;
    }

    public static void setAutoTrainingMode(Boolean autoTrainNetwork) {
        MNISTFrame.autoTrainNetwork = autoTrainNetwork;
    }

    private void restartNetwork() {
        System.out.println("A new network for training MNISt haas been pre-paired.");
        networkDenseLayers = new DenseLayer[] { // Layer arrays.
                //                new Layer(256, ActivationFunction.RE_LU), // ReLu layer
                new DenseLayer(128, ActivationFunction.RE_LU), // ReLu layer
                new DenseLayer(128, ActivationFunction.RE_LU), // ReLu layer
                new DenseLayer(128, ActivationFunction.RE_LU), // ReLu layer
                new DenseLayer(10, ActivationFunction.SIGMOID), // Sigmoid layer
        };
        jNeuralNetwork = new JNeuralNetwork( // Neural Network.
                LossFunction.LOG_COSH,//Loss Function
                JNetworkOptimizer.ADAM, // Optimizer
                28 * 28, // Input nodes
                networkDenseLayers // Network Layers.
        );
        jNeuralNetwork.setLearningRate(0.001);
        jNeuralNetwork.setMomentumFactorBeta1(0.99);
        jNeuralNetwork.setDebugMode(true);
    }

    private SaveStatus saveJNeuralNetwork(String fileName) {
        System.out.println("Neural network is being saved...");
        try {
            String resourcePath = STR."\{System.getProperty("user.dir")}//src//main//resources//com//mkproductions//\{fileName}.json";
            File file = new File(resourcePath);
            if (file.exists()) {
                int choice = JOptionPane.showConfirmDialog(this, "File already exists. Do you want to overwrite it?", "File Exists", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);

                if (choice != JOptionPane.YES_OPTION) {
                    return SaveStatus.FILE_EXISTS;
                }
            }

            Gson gson = new GsonBuilder().setExclusionStrategies(new ExclusionStrategy() {
                @Override
                public boolean shouldSkipField(FieldAttributes fieldAttributes) {
                    return fieldAttributes.getDeclaredType().equals(Random.class);
                }

                @Override
                public boolean shouldSkipClass(Class<?> aClass) {
                    return false;
                }
            }).create();
            FileWriter writer = new FileWriter(resourcePath);
            gson.toJson(jNeuralNetwork, writer);
            writer.flush();
            writer.close();
            System.out.println("Neural network saved successfully!");
            return SaveStatus.SAVED;
        } catch (IOException e) {
            System.out.println(STR."Error saving neural network: \{e.getMessage()}");
            return SaveStatus.FAILED;
        }
    }

    public void startRendering() throws InterruptedException {
        this.running = true;
        while (this.running) {
            this.mnistDrawningJPanel.repaint();
            this.mnistNetworkOutputJPanel.repaint();
            if (networkAccuracy > 0.0 && networkAccuracy <= 98 && autoTrainNetwork) {
                triggerNetworkTraining();
            }
            Thread.sleep(16);
        }
        dispose();
        System.exit(-200);
    }

    private double[][][] prepareTrainingTestingDataSet() {
        System.out.println("Please! Wait for preparing training data...");
        String dataPath = STR."\{System.getProperty("user.dir")}//src//main//resources//com//mkproductions//";

        String trainingDataPath = STR."\{dataPath}training_data//mnist_train.csv";
        String testingDataPath = STR."\{dataPath}testing_data//mnist_test.csv";

        CSVBufferedReader csvTrainingDataBufferedReader = new CSVBufferedReader(trainingDataPath);
        CSVBufferedReader csvTestingDataBufferedReader = new CSVBufferedReader(testingDataPath);

        List<List<Double>> csvTrainingDataTable = csvTrainingDataBufferedReader.getTable();
        var csvTrainingOutputColumn = csvTrainingDataBufferedReader.getColumn("label");
        List<List<Double>> csvTestingDataTable = csvTestingDataBufferedReader.getTable();
        var csvTestingOutputColumn = csvTestingDataBufferedReader.getColumn("label");

        double[][] trainingInputs = new double[csvTrainingDataTable.size()][csvTrainingDataTable.getFirst().size()];
        double[][] trainingOutputs = new double[csvTrainingOutputColumn.size()][10];
        double[][] testingInputs = new double[csvTestingDataTable.size()][csvTestingDataTable.getFirst().size()];
        double[][] testingOutputs = new double[csvTestingOutputColumn.size()][10];

        // For filtering training data.
        for (int a = 0; a < trainingInputs.length; a++) {
            for (int b = 0; b < trainingInputs[0].length; b++) {
                double value = csvTrainingDataTable.get(a).get(b);
                trainingInputs[a][b] = value / 255;//Mapper.mapRangeToRange(value, 0, 255, 0, 1);
            }
        }
        //        for (double[] rows : trainingInputs) {
        //            System.out.println(STR."Row: \{Arrays.toString(rows)}");
        //        }
        // Converting training outputs into raw arrays.
        for (int a = 0; a < trainingOutputs.length; a++) {
            trainingOutputs[a][csvTrainingOutputColumn.get(a)] = 1.0;
            //            System.out.println(STR."Index: \{csvTrainingOutputColumn.get(a)}");
        }
        // For filtering testing data.
        for (int a = 0; a < testingInputs.length; a++) {
            for (int b = 0; b < testingInputs[0].length; b++) {
                double value = csvTestingDataTable.get(a).get(b); //Mapper.mapRangeToRange(csvTestingDataTable.get(a).get(b), 0, 255, 0, 1);
                testingInputs[a][b] = value / 255;
            }
        }
        // Converting testing outputs into raw outputs.
        for (int a = 0; a < testingOutputs.length; a++) {
            testingOutputs[a][csvTestingOutputColumn.get(a)] = 1.0;
        }
        System.out.println("Thanks for waiting!!!");
        return new double[][][] { trainingInputs, trainingOutputs, testingInputs, testingOutputs };
    }

    void triggerNetworkTraining() {
        MNISTFrame.jNeuralNetwork.train(this.trainingInputs, this.trainingOutputs, epochCount);
        networkAccuracy = MNISTFrame.jNeuralNetwork.calculateAccuracy(this.testingInputs, this.testingOutputs);
        mnistNetworkOutputJPanel.triggerNetworkPrediction();
    }

    public void clearGridData() {
        for (int a = 0; a < dataGrid.length; a++) {
            for (int b = 0; b < dataGrid[0].length; b++) {
                dataGrid[a][b] = 0;
            }
        }
    }

    private void printSingleTrainingEntry() {
        //        var random = new Random();
        int randomIndex = 0; //random.nextInt(this.trainingInputs.length);
        double[] trainingInput = trainingInputs[randomIndex];
        double[] trainingOutput = trainingOutputs[randomIndex];
        int index = 0;
        // Printing as console image.
        for (int a = 0; a < 28; a++) {
            for (int b = 0; b < 28; b++) {
                double input = trainingInput[index++];
                System.out.print(input == 0 ? "_" : "#");
            }
            System.out.println();
        }
        // Printing as core values.
        index = 0;
        for (int a = 0; a < 28; a++) {
            for (int b = 0; b < 28; b++) {
                double input = trainingInput[index];
                System.out.print(input);
                if (index != (28 * 28) - 1) {
                    System.out.print(",");
                }
                index++;
            }
            System.out.println();
        }
        System.out.println(STR."Label: \{Arrays.toString(trainingOutput)}");
        double[] prediction = jNeuralNetwork.processInputs(trainingInput);
        System.out.println(STR."Prediction: \{Arrays.toString(prediction)}");
    }
}