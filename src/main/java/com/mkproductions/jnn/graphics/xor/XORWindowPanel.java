package com.mkproductions.jnn.graphics.xor;

import com.mkproductions.jnn.entity.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.entity.Mapper;
import com.mkproductions.jnn.entity.lossFunctions.RegressionLossFunction;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;


public class XORWindowPanel extends JPanel {
    private double cellCount = 10;
    private final JNeuralNetwork jNeuralNetwork;
    private final double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private final double[][] trainingOutputs = {{0}, {1}, {0}, {1}};

    public XORWindowPanel(int width, int height) {
        Layer[] networkLayers = new Layer[]{new Layer(10, ActivationFunction.SIGMOID), new Layer(1, ActivationFunction.SIGMOID),};
        this.jNeuralNetwork = new JNeuralNetwork(RegressionLossFunction.SMOOTH_MEAN_ABSOLUTE_ERROR, 2, networkLayers);
        setSize(width, height);
        setVisible(true);
        setBackground(Color.black);
        this.jNeuralNetwork.setLearningRate(0.01);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int cellSize = 10;
        for (double x = 0; x < getWidth(); x += cellSize) {
            for (double y = 0; y < getHeight(); y += cellSize) {
                double[] inputs = {x / getWidth(), y / getHeight()};
                double prediction = this.jNeuralNetwork.processInputs(inputs)[0];
                int output = (int) Mapper.mapRangeToRange(prediction, 0, 1, 0, 255);
                g.setColor(new Color(output, output, output));
                g.fillRect((int) x, (int) y, cellSize, cellSize);
            }
        }
        try {
            int epochs = 1000;
            this.jNeuralNetwork.train(this.trainingInputs, this.trainingOutputs, epochs);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        g.setColor(Color.white);
        g.drawString("Learning rate: " + this.jNeuralNetwork.getLearningRate(), 10, 10);
        System.out.println();
    }

    public int getCellCount() {
        return (int) cellCount;
    }

    public void setCellCount(double cellCount) {
        this.cellCount = cellCount;
    }

}
