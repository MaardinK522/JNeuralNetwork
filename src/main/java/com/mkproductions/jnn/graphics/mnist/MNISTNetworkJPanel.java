package com.mkproductions.jnn.graphics.mnist;

import com.mkproductions.jnn.entity.ActivationFunction;
import com.mkproductions.jnn.entity.Mapper;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

import static com.mkproductions.jnn.graphics.mnist.MNISTFrame.dataGrid;

public class MNISTNetworkJPanel extends JPanel {
    private double[] prediction = new double[10];

    public MNISTNetworkJPanel(int w, int h) {
        setSize(w, h);
        setVisible(true);
        setBackground(Color.blue);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int gap = 20;
        int x = 10;
        int y = 20;
        for (int a = 0; a < this.prediction.length; a++) {
            int color = (int) (this.prediction[a] * 255);
            g.setColor(new Color(color, color, color));
            g.fillRect(x, y, 50, 50);
            g.setColor(Color.white);
            g.drawString("" + a, x + 20, y - 5);
            x += 50 + gap;
            if (x + 50 >= getWidth() - 10) {
                y += 50 + gap;
                x = 10;
            }
        }
        this.triggerNetworkPrediction();
    }

    public void triggerNetworkPrediction() {
        double[] trainingInput = new double[dataGrid.length * dataGrid[0].length];
        int index = 0;
        for (double[] doubles : dataGrid) {
            for (int b = 0; b < dataGrid[0].length; b++) {
                double data = doubles[b];
                trainingInput[index++] = data;
            }
        }
        this.prediction = MNISTFrame.jNeuralNetwork.processInputs(trainingInput);
        System.out.println("Prediction: " + Arrays.toString(this.prediction));
    }
}
