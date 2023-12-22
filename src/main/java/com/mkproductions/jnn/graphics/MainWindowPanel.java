package com.mkproductions.jnn.graphics;

import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.entity.Mapper;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;


public class MainWindowPanel extends JPanel {
    private int cellSize = 50;
    private final JNeuralNetwork jnn;
    double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double[][] trainingTargets = {{0}, {1}, {1}, {0}};
    private final Layer[] networkLayers;

    MainWindowPanel(int width, int height) {
        this.networkLayers = new Layer[]{
                new Layer(4, ActivationFunction.SIGMOID),
                new Layer(1, ActivationFunction.SIGMOID)
        };
        this.jnn = new JNeuralNetwork(2, this.networkLayers);
        setSize(width, height);
        setVisible(true);
        setBackground(Color.black);
        this.jnn.setLearningRate(0.01);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        try {
            for (double x = 0; x < 500; x += this.cellSize) {
                for (double y = 0; y < 500; y += this.cellSize) {
                    double[] inputs = {x / 500, y / 500};
                    double prediction = this.jnn.processInputs(inputs)[0];
                    int output = (int) Mapper.mapPredictionToRange(prediction, this.networkLayers[this.networkLayers.length - 1].activationFunction(), 0, 255);
                    g.setColor(new Color(output, output, output));
                    g.fillRect((int) x, (int) y, this.cellSize, this.cellSize);
                }
            }
            int epochs = 10;
            this.jnn.train(this.trainingInputs, this.trainingTargets, epochs);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        g.setColor(Color.white);
        g.drawString("Learning rate: " + this.jnn.getLearningRate(), 10, 10);
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println("Top left corner: " + Arrays.toString(this.jnn.processInputs(new double[]{0, 0})));
        System.out.println("Top right corner: " + Arrays.toString(this.jnn.processInputs(new double[]{0, 1})));
        System.out.println("Bottom left corner: " + Arrays.toString(this.jnn.processInputs(new double[]{1, 0})));
        System.out.println("Bottom right corner: " + Arrays.toString(this.jnn.processInputs(new double[]{1, 1})));
    }

    public int getCellSize() {
        return cellSize;
    }

    public void setCellSize(int cellSize) {
        this.cellSize = cellSize;
    }
}
