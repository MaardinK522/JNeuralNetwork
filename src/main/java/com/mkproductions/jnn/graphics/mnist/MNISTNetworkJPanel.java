package com.mkproductions.jnn.graphics.mnist;

import com.mkproductions.jnn.entity.Mapper;
import com.mkproductions.jnn.entity.activationFunctions.ActivationFunction;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;


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
        System.out.println();
        for (int a = 0; a < 10; a++) {
            int color = (int) Mapper.mapRangeToRange(this.prediction[a], -1, 1, 0, 255);
            color = Math.abs(color);
            if (color > 255)
                color = 255;
            System.out.println("Prediction: " + this.prediction[a]);
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
        double[][] dataGrid = MNISTFrame.getDataGrid();
        double[] trainingInput = new double[dataGrid.length * dataGrid[0].length];
        int index = 0;
        for (double[] doubles : dataGrid) {
            for (double adouble : doubles) {
                trainingInput[index++] = adouble;
            }
        }
        this.prediction = MNISTFrame.processNetworkInputs(trainingInput);
        System.out.println("Prediction: " + Arrays.toString(this.prediction));
    }
}
