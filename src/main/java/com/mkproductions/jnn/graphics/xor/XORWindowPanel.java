package com.mkproductions.jnn.graphics.xor;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.entity.Mapper;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNeuralNetworkOptimizer;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;

public class XORWindowPanel extends JPanel {
    private double cellCount = 10;
    private final JNeuralNetwork jNeuralNetwork;
    private final double[][] trainingInputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    private final double[][] trainingOutputs = { { 0 }, { 1 }, { 1 }, { 0 } };

    public XORWindowPanel(int width, int height) {
        DenseLayer[] networkDenseLayers = new DenseLayer[] { // Layers.
                new DenseLayer(4, ActivationFunction.SIGMOID), // First Layer
                new DenseLayer(1, ActivationFunction.SIGMOID) // Output Layer
        };
        this.jNeuralNetwork = new JNeuralNetwork( // Neural Network.
                LossFunction.CATEGORICAL_CROSS_ENTROPY, // Loss function of the network.
                JNeuralNetworkOptimizer.ADAM, // Optimizer
                2, // Output nodes
                networkDenseLayers // Layers
        );
        this.jNeuralNetwork.setDebugMode(false);
        setSize(width, height);
        setVisible(true);
        setBackground(Color.black);
        this.jNeuralNetwork.setLearningRate(0.0001);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int cellSize = 10;
        for (double x = 0; x < getWidth(); x += cellSize) {
            for (double y = 0; y < getHeight(); y += cellSize) {
                double[] inputs = { x / getWidth(), y / getHeight() };
                double prediction = this.jNeuralNetwork.processInputs(inputs)[0];
                if (x == 0 && y == 0) {
                    System.out.println("Prediction: " + prediction);
                }
                int output = (int) Mapper.mapRangeToRange(prediction, 0, 1, 0, 255);
                //                int output = (prediction < 1) ? 0 : 255;
                g.setColor(new Color(output, output, output));
                g.fillRect((int) x, (int) y, cellSize, cellSize);
            }
        }
        try {
            this.jNeuralNetwork.train(this.trainingInputs, this.trainingOutputs, 1000);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        g.setColor(Color.white);
        g.drawString("Learning rate: " + this.jNeuralNetwork.getLearningRate(), 10, 10);
    }

    public int getCellCount() {
        return (int) cellCount;
    }

    public void setCellCount(double cellCount) {
        this.cellCount = cellCount;
    }

    public void printAccuracy() {
        double accuracy = this.jNeuralNetwork.calculateAccuracy(this.trainingInputs, this.trainingOutputs);
        System.out.println("Accuracy of the network: " + accuracy);
    }
}