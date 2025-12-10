package com.mkproductions.jnn.graphics.xor;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.entity.Mapper;
import com.mkproductions.jnn.gpu.gpu_layers.Dense;
import com.mkproductions.jnn.gpu.gpu_layers.GeneralLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.networks.JGPUNeuralNetwork;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;

import javax.swing.*;
import java.awt.*;

public class XORWindowPanel extends JPanel {
    private double cellCount = 5;
    private final JGPUNeuralNetwork jNeuralNetwork;
    private final Tensor[] trainingInputs = { // Training inputs
            new Tensor(DoubleArray.fromArray(new double[]{0, 0}), 2, 1), // Top left corner
            new Tensor(DoubleArray.fromArray(new double[]{1, 0}), 2, 1), // Bottom left corner
            new Tensor(DoubleArray.fromArray(new double[]{0, 1}), 2, 1), // Top right corner
            new Tensor(DoubleArray.fromArray(new double[]{1, 1}), 2, 1) // Bottom right corner.
    };
    private final Tensor[] trainingOutputs = { // Training outputs
            new Tensor(DoubleArray.fromArray(new double[]{0}), 1, 1), // Top left corner
            new Tensor(DoubleArray.fromArray(new double[]{1}), 1, 1), // Bottom left corner
            new Tensor(DoubleArray.fromArray(new double[]{1}), 1, 1), // Top right corner
            new Tensor(DoubleArray.fromArray(new double[]{0}), 1, 1) // Bottom right corner
    };

    public XORWindowPanel(int width, int height) {
        GeneralLayer[] networkDenseLayers = new GeneralLayer[]{ // Layers.
                new Dense(4, ActivationFunction.TAN_H), // Layer
                new Dense(4, ActivationFunction.TAN_H), // Layer
                new Dense(1, ActivationFunction.SIGMOID) // Output Layer
        };
        this.jNeuralNetwork = new JGPUNeuralNetwork( // Neural Network.
                new int[]{2}, // Input nodes
                LossFunction.MEAN_SQUARED_ERROR, // Loss-function of the network.
                JNetworkOptimizer.RMS_PROP, // Optimizer
                networkDenseLayers // Layers
        );
//        this.jNeuralNetwork.setDebugMode(false);
        setSize(width, height);
        setVisible(true);
        setBackground(Color.black);
        this.jNeuralNetwork.setLearningRate(0.001);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int cellSize = 10;
        for (double x = 0; x < getWidth(); x += cellSize) {
            for (double y = 0; y < getHeight(); y += cellSize) {
                Tensor inputs = new Tensor(DoubleArray.fromArray(new double[]{x / getWidth(), y / getHeight()}), 2, 1);
                Tensor prediction = this.jNeuralNetwork.processInputs(inputs);
                if (x >= getWidth() && y >= getWidth()) {
                    System.out.println(STR."Prediction: \{prediction}");
                }
                int output = (int) Mapper.mapRangeToRange(prediction.getData().get(0), 0, 1, 0, 255);
//                int output = (prediction.getData().get(0) < 1) ? 0 : 255;
                g.setColor(new Color(output, output, output));
                g.fillRect((int) x, (int) y, cellSize, cellSize);
            }
        }
        try {
            this.jNeuralNetwork.train(this.trainingInputs, this.trainingOutputs, 1);
        } catch (Exception e) {
            e.printStackTrace();
            throw new IllegalArgumentException(e);
        }
        g.setColor(Color.white);
        g.drawString(STR."Learning rate: \{this.jNeuralNetwork.getLearningRate()}", 10, 10);
    }

    public int getCellCount() {
        return (int) cellCount;
    }

    public void setCellCount(double cellCount) {
        this.cellCount = cellCount;
    }

//    public void printAccuracy() {
//        double accuracy = this.jNeuralNetwork.calculateAccuracy(this.trainingInputs, this.trainingOutputs);
//        System.out.println(STR."Accuracy of the network: \{accuracy}");
//    }
}