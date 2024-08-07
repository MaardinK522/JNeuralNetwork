package com.mkproductions.jnn.graphics.xor;

import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.CSVBufferedReader;
import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.entity.Mapper;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.util.List;


public class XORWindowPanel extends JPanel {
    private double cellCount = 10;
    private final JNeuralNetwork jNeuralNetwork;
    private final Layer[] networkLayers;
    private final double[][] trainingInputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
    };
    private final double[][] trainingOutputs = {
            {0},
            {1},
            {0},
            {1}
    };
    private int cellSize = 10;

    public XORWindowPanel(int width, int height) {
        this.networkLayers = new Layer[]{
                new Layer(10, ActivationFunction.SIGMOID),
                new Layer(1, ActivationFunction.SIGMOID),
        };
        this.jNeuralNetwork = new JNeuralNetwork(2, this.networkLayers);
        setSize(width, height);
        setVisible(true);
        setBackground(Color.black);
        this.jNeuralNetwork.setLearningRate(0.01);
    }


    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        try {
            for (double x = 0; x < getWidth(); x += this.cellSize) {
                for (double y = 0; y < getHeight(); y += this.cellSize) {
                    double[] inputs = {x / getWidth(), y / getHeight()};
                    double prediction = this.jNeuralNetwork.processInputs(inputs)[0];
                    int output = (int) Mapper.mapRangeToRange(prediction, 0, 1, 0, 255);
                    g.setColor(new Color(output, output, output));
                    g.fillRect((int) x, (int) y, this.cellSize, this.cellSize);
                }
            }
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
