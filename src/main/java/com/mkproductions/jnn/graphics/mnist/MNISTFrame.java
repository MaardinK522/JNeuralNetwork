package com.mkproductions.jnn.graphics.mnist;

import com.google.gson.ExclusionStrategy;
import com.google.gson.FieldAttributes;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.CSVBufferedReader;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.layers.*;
import com.mkproductions.jnn.cpu.entity.LossFunctionAble;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.networks.JSequential;
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
    private final Tensor[] trainingInputs;
    private final Tensor[] trainingOutputs;
    private final Tensor[] testingInputs;
    private final Tensor[] testingOutputs;
    private static final int[][] dataGrid = new int[28][28];
    private static JSequential jNeuralNetwork;
    private final SaveNeuralNetworkDialog saveNeuralNetworkDialog;
    public static boolean autoTrainNetwork = false;
    public static double networkAccuracy = 0.0;
    public static Layer[] networkDenseLayers;

    public MNISTFrame(String frameName) {
        this.running = false;
        // Declaring the size of the inputs and outputs.
        Tensor[][] trainingTestingData = this.prepareTrainingTestingDataSet();
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
        /**        addKeyListener(new KeyAdapter() {
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
         */
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

    private void restartNetwork() {
        System.out.println("A new network for training MNISt haas been pre-paired.");
        networkDenseLayers = new Layer[]{ // Layer arrays.
                //                new Layer(256, ActivationFunction.RE_LU), // ReLu layer
                new ConvolutionLayer(3, 16, 1, 1, ActivationFunction.RE_LU), // CNN Relu
                new PoolingLayer(2, 2, PoolingLayer.PoolingLayerType.MAX), // Pool
                new ConvolutionLayer(3, 32, 1, 1, ActivationFunction.RE_LU), // CNN Relu
                new PoolingLayer(2, 2, PoolingLayer.PoolingLayerType.MAX), // Pool
                new FlattenLayer(),
                new DenseLayer(10, ActivationFunction.SOFTMAX), // Sigmoid layer
        };
        jNeuralNetwork = new JSequential( // Neural Network.
                new int[]{1, 28, 28}, // Input shape
                LossFunction.SPARSE_CATEGORICAL_CROSS_ENTROPY, //Loss Function
                JNetworkOptimizer.ADAM, // Optimizer
                networkDenseLayers // Network Layers.
        );
        jNeuralNetwork.setLearningRate(0.001);
        jNeuralNetwork.setMomentumFactorBeta1(0.99);
        jNeuralNetwork.setDebugMode(true);
    }

    public static Tensor processNetworkInputs(Tensor inputs) {
        return jNeuralNetwork.processInputs(inputs);
    }

    public static int[][] getDataGrid() {
        return dataGrid;
    }

    public static JSequential getJNeuralNetwork() {
        return jNeuralNetwork;
    }

    public static double getNetworkAccuracy() {
        return networkAccuracy;
    }

    public static LossFunctionAble getNetworkLoss() {
        return jNeuralNetwork.getLossFunction();
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
        return jNeuralNetwork.getOptimizer();
    }

    public static Boolean getTrainNetworkStatus() {
        return autoTrainNetwork;
    }

    public static void setAutoTrainingMode(Boolean autoTrainNetwork) {
        MNISTFrame.autoTrainNetwork = autoTrainNetwork;
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

    private Tensor[][] prepareTrainingTestingDataSet() {
        System.out.println("Please! Wait for preparing training data...");

        // --- FIX: Using File.separator for cross-platform compatibility ---
        String userDir = System.getProperty("user.dir");
        String sep = File.separator;

        // Construct base data path using system separator
        String dataPath = STR."\{userDir}\{sep}src\{sep}main\{sep}resources\{sep}com\{sep}mkproductions\{sep}";

        String trainingDataPath = STR."\{dataPath}training_data\{sep}mnist_train.csv";
        String testingDataPath = STR."\{dataPath}testing_data\{sep}mnist_test.csv";
        // --- END FIX ---

        CSVBufferedReader csvTrainingDataBufferedReader = new CSVBufferedReader(trainingDataPath);
        CSVBufferedReader csvTestingDataBufferedReader = new CSVBufferedReader(testingDataPath);
        System.out.println("Data has been loaded successfully.");

        System.out.println("Wait for data to tensor conversion...");
        List<List<Double>> csvTrainingDataTable = csvTrainingDataBufferedReader.getTable();
        var csvTrainingOutputColumn = csvTrainingDataBufferedReader.getColumn("label");
        List<List<Double>> csvTestingDataTable = csvTestingDataBufferedReader.getTable();
        var csvTestingOutputColumn = csvTestingDataBufferedReader.getColumn("label");

        Tensor[] trainingInputs = new Tensor[csvTrainingDataTable.size()];
        Tensor[] trainingOutputs = new Tensor[csvTrainingOutputColumn.size()];
        Tensor[] testingInputs = new Tensor[csvTestingDataTable.size()];
        Tensor[] testingOutputs = new Tensor[csvTestingOutputColumn.size()];

        final int barLength = 50;
        int totalTrainingSamples = csvTrainingDataTable.size();
        int totalTestingSamples = csvTestingDataTable.size();

        // ----------------------------------------------------
        // 1. Converting Training Inputs (Pixel Data)
        // ----------------------------------------------------
        System.out.println("\nConverting Training Inputs (Images)...");
        int lastProgress = -1;
        for (int a = 0; a < totalTrainingSamples; a++) {

            trainingInputs[a] = new Tensor(1, 28, 28);
            // Assuming the first column is the label and is skipped, and image data follows.
            // If the label column is included in the List<List<Double>>, adjust 'b' starting index.
            for (int b = 0; b < csvTrainingDataTable.getFirst().size(); b++) {
                double value = csvTrainingDataTable.get(a).get(b);
                // Assume the CSV data order matches the flattened tensor data order
                trainingInputs[a].getData()[b] = value / 255.0; // Normalize pixel data
            }

            // --- Progress Tracking Logic ---
            int progress = (int) ((double) (a + 1) * 100 / totalTrainingSamples);
            if (progress != lastProgress) {
                lastProgress = progress;

                int filled = progress * barLength / 100;
                String filledBar = "#".repeat(filled);
                String emptyBar = " ".repeat(barLength - filled);

                System.out.printf("\rProgress: %d/%d (%.2f%%) [%s%s]", a + 1, totalTrainingSamples, (double) progress, filledBar, emptyBar);
            }
        }
        System.out.println("\nTraining Input Conversion Complete.");

        // ----------------------------------------------------
        // 2. Converting Training Outputs (Labels to One-Hot)
        // ----------------------------------------------------
        System.out.println("Converting Training Outputs (Labels)...");
        // This is fast and usually doesn't need a bar, but we iterate for completeness.
        for (int a = 0; a < csvTrainingOutputColumn.size(); a++) {
            trainingOutputs[a] = new Tensor(10, 1);
            int label = csvTrainingOutputColumn.get(a);
            if (label >= 0 && label < 10) {
                trainingOutputs[a].getData()[label] = 1.0;
            } else {
                // Handle invalid label if necessary
                throw new RuntimeException(STR."Invalid label: \{label}");
            }
        }

        // ----------------------------------------------------
        // 3. Converting Testing Inputs (Pixel Data)
        // ----------------------------------------------------
        System.out.println("Converting Testing Inputs (Images)...");
        lastProgress = -1;
        for (int a = 0; a < totalTestingSamples; a++) {
            testingInputs[a] = new Tensor(1, 28, 28);
            for (int b = 0; b < csvTestingDataTable.getFirst().size(); b++) {
                double value = csvTestingDataTable.get(a).get(b);
                testingInputs[a].getData()[b] = value / 255.0; // Normalize pixel data
            }

            // --- Progress Tracking Logic ---
            int progress = (int) ((double) (a + 1) * 100 / totalTestingSamples);
            if (progress != lastProgress) {
                lastProgress = progress;

                int filled = progress * barLength / 100;
                String filledBar = "#".repeat(filled);
                String emptyBar = " ".repeat(barLength - filled);

                System.out.printf("\rProgress: %d/%d (%.2f%%) [%s%s]", a + 1, totalTestingSamples, (double) progress, filledBar, emptyBar);
            }
        }
        System.out.println("\nTesting Input Conversion Complete.");

        // ----------------------------------------------------
        // 4. Converting Testing Outputs (Labels to One-Hot)
        // ----------------------------------------------------
        System.out.println("Converting Testing Outputs (Labels)...");
        for (int a = 0; a < csvTestingOutputColumn.size(); a++) {
            testingOutputs[a] = new Tensor(10, 1);
            int label = csvTestingOutputColumn.get(a);
            if (label >= 0 && label < 10) {
                testingOutputs[a].getData()[label] = 1.0;
            }
        }

        System.out.println("Thanks for waiting!!! Data preparation successful.");
        return new Tensor[][]{trainingInputs, trainingOutputs, testingInputs, testingOutputs};
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
        var random = new Random();
        int randomIndex = random.nextInt(this.trainingInputs.length);
        Tensor trainingInput = trainingInputs[randomIndex];
        Tensor trainingOutput = trainingOutputs[randomIndex];
        int index = 0;
        // Printing as console image.
        for (int a = 0; a < 28; a++) {
            for (int b = 0; b < 28; b++) {
                double input = trainingInput.getData()[index++];
                System.out.print(input == 0 ? "_" : "#");
            }
            System.out.println();
        }
        // Printing as core values.
        index = 0;
        for (int a = 0; a < 28; a++) {
            for (int b = 0; b < 28; b++) {
                double input = trainingInput.getData()[index];
                System.out.print(input);
                if (index != (28 * 28) - 1) {
                    System.out.print(",");
                }
                index++;
            }
            System.out.println();
        }
        System.out.println(STR."Label: \{trainingOutput}");
        Tensor prediction = jNeuralNetwork.processInputs(trainingInput);
        System.out.println(STR."Prediction: \{prediction}");
    }
}