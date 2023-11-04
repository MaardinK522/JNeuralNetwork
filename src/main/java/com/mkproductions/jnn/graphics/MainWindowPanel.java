package com.mkproductions.jnn.graphics;

import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.Layer;
import com.mkproductions.jnn.entity.Mapper;
import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;

public class MainWindowPanel extends JPanel {
    private int cellSize = 50;
    private final JNeuralNetwork jnn;
    double[][] trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double[][] trainingTargets = {{0}, {1}, {1}, {0}};

    MainWindowPanel(int width, int height) {
        this.jnn = new JNeuralNetwork(
                2,
                new Layer(2, ActivationFunction.RE_LU),
                new Layer(2, ActivationFunction.SIGMOID),
                new Layer(1, ActivationFunction.TAN_H)
        );
        setSize(width, height);
        setVisible(true);
        setBackground(Color.black);
        this.jnn.setLearningRate(0.0001);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        try {
            for (double x = 0; x < 500; x += this.cellSize) {
                for (double y = 0; y < 500; y += this.cellSize) {
                    double[] inputs = {x / 500, y / 500};
                    double prediction = this.jnn.processInputs(inputs)[0];
                    int output = (int) Mapper.map(prediction, ActivationFunction.RE_LU, 0, 255);
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
    }

    public int getCellSize() {
        return cellSize;
    }

    public void setCellSize(int cellSize) {
        this.cellSize = cellSize;
    }
}
