package com.mkproductions.jnn.graphics.mnist;

import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.CSVBufferedReader;
import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.entity.Mapper;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.List;

public class MNISTFrame extends JFrame {
    public static double[][] dataGrid = new double[28][28];
    int w = 560;
    int h = 560;

    private final MNISTTestingJPanel mnistTestingJPanel;
    private final MNISTNetworkJPanel mnistNetworkJPanel;
    private boolean running = false;
    private final double[][] trainingInputs;
    private final double[][] trainingOutputs;
    private final double[][] testingInputs;
    private final double[][] testingOutputs;
    public static JNeuralNetwork jNeuralNetwork;
    private final Layer[] networkLayers;
    private final int epochs = 100;
    private int mouseX = 0;
    private int mouseY = 0;

    public MNISTFrame(String frameName) {
        // Initializing network
        this.networkLayers = new Layer[]{
                new Layer(10, ActivationFunction.SIGMOID),
                new Layer(10, ActivationFunction.SIGMOID),
        };
        jNeuralNetwork = new JNeuralNetwork(784, networkLayers);
        jNeuralNetwork.setLearningRate(0.01);

        // Declaring size of the inputs & outputs.
        double[][][] trainingTestingData = this.prepareTrainingTestingDataSet();
        this.trainingInputs = trainingTestingData[0];
        this.trainingOutputs = trainingTestingData[1];
        this.testingInputs = trainingTestingData[2];
        this.testingOutputs = trainingTestingData[3];

        this.mnistTestingJPanel = new MNISTTestingJPanel(this.w / 2, this.h);
        this.mnistNetworkJPanel = new MNISTNetworkJPanel(this.w / 2, this.h);

        add(this.mnistNetworkJPanel, BorderLayout.EAST);
        add(this.mnistTestingJPanel, BorderLayout.WEST);
        addKeyListener(
                new KeyAdapter() {
                    @Override
                    public void keyPressed(KeyEvent e) {
                        super.keyPressed(e);
                        if (e.getKeyCode() == KeyEvent.VK_ESCAPE)
                            running = false;
                        if (e.getKeyCode() == KeyEvent.VK_T)
                            triggerNetworkTraining();
                        if (e.getKeyCode() == KeyEvent.VK_C)
                            clearGridData();
                        if (e.getKeyCode() == KeyEvent.VK_R)
                            restartNetwork();
                    }
                }
        );
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
        addMouseMotionListener(
                new MouseMotionAdapter() {
                    @Override
                    public void mouseDragged(MouseEvent e) {
                        super.mouseDragged(e);
                        mouseX = e.getX();
                        mouseY = e.getY();
                    }
                }
        );
        setLayout(new GridLayout(1, 2));
        setTitle(frameName);
        setSize(this.w, this.h);
        setLocationRelativeTo(null);
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
//        jNeuralNetwork.train(trainingInputs, trainingOutputs, this.epochs * 2);
    }

    private void restartNetwork() {
        jNeuralNetwork = new JNeuralNetwork(
                784,
                this.networkLayers
        );
        jNeuralNetwork.setLearningRate(0.0001);
    }

    public void startRendering() {
        this.running = true;
        while (this.running) {
            this.mnistTestingJPanel.repaint();
            this.mnistNetworkJPanel.repaint();
            try {
                Thread.sleep(1000 / 30);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        dispose();
    }

    private double[][][] prepareTrainingTestingDataSet() {
        System.out.println("Please! Wait for preparing training data...");
        String dataPath = System.getProperty("user.dir") + "//src//main//resources//com//mkproductions//";

        String trainingDataPath = dataPath + "training_data//mnist_train.csv";
        String testingDataPath = dataPath + "testing_data//mnist_test.csv";

        CSVBufferedReader csvTrainingDataBufferedReader = new CSVBufferedReader(trainingDataPath);
        CSVBufferedReader csvTestingDataBufferedReader = new CSVBufferedReader(testingDataPath);

        java.util.List<java.util.List<Double>> csvTrainingDataTable = csvTrainingDataBufferedReader.getTable();
        List<Double> csvTrainingOutputColumn = csvTrainingDataBufferedReader.getColumn("label");
        java.util.List<java.util.List<Double>> csvTestingDataTable = csvTestingDataBufferedReader.getTable();
        List<Double> csvTestingOutputColumn = csvTestingDataBufferedReader.getColumn("label");

        double[][] trainingInputs = new double[csvTrainingDataTable.size()][csvTrainingDataTable.get(0).size()];
        double[][] trainingOutputs = new double[csvTrainingOutputColumn.size()][10];
        double[][] testingInputs = new double[csvTestingDataTable.size()][csvTestingDataTable.get(0).size()];
        double[][] testingOutputs = new double[csvTestingOutputColumn.size()][10];

        // For filtering training data.
        for (int a = 0; a < csvTrainingDataTable.size(); a++) {
            for (int b = 0; b < csvTrainingDataTable.get(a).size(); b++) {
                trainingInputs[a][b] = Mapper.mapRangeToRange(csvTrainingDataTable.get(a).get(b), 0, 255, 0, 1);
            }
        }
        // Converting training outputs into raw arrays.
        for (int a = 0; a < trainingOutputs.length; a++) {
            int trainingOutputIndex = csvTrainingOutputColumn.get(a).intValue();
            trainingOutputs[a][trainingOutputIndex] = 1;
        }

        // For filtering testing data.
        for (int a = 0; a < csvTestingDataTable.size(); a++) {
            for (int b = 0; b < csvTestingDataTable.get(a).size(); b++) {
                testingInputs[a][b] = Mapper.mapRangeToRange(csvTestingDataTable.get(a).get(b), 0, 255, 0, 1);
            }
        }

        // Converting testing outputs into raw outputs.
        for (int a = 0; a < testingOutputs.length; a++) {
            int testingOutputIndex = csvTestingOutputColumn.get(a).intValue();
            testingOutputs[a][testingOutputIndex] = 1;
        }
        System.out.println("Thanks for waiting!!!");
        return new double[][][]{
                trainingInputs,
                trainingOutputs,
                testingInputs,
                testingOutputs,
        };
    }

    void triggerNetworkTraining() {
        MNISTFrame.jNeuralNetwork.train(this.trainingInputs, this.trainingOutputs, this.epochs);
    }

    public void clearGridData() {
        for (int a = 0; a < dataGrid.length; a++) {
            for (int b = 0; b < dataGrid[0].length; b++) {
                dataGrid[a][b] = 0;
            }
        }
    }
}
