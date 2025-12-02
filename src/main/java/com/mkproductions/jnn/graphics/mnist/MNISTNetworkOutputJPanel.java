package com.mkproductions.jnn.graphics.mnist;

import com.mkproductions.jnn.cpu.entity.Mapper;
import com.mkproductions.jnn.cpu.entity.Tensor;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

public class MNISTNetworkOutputJPanel extends JPanel {
    private Tensor prediction = new Tensor(10);

    public MNISTNetworkOutputJPanel(int w, int h) {
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
        for (int a = 0; a < 10; a++) {
            int color = (int) Mapper.mapRangeToRange(prediction.getData()[a], 0, 1, 0, 255);
            //            int color = (this.prediction[a] < 1) ? 0 : 255;
            color = Math.abs(color);
            //            if (color > 255) color = 255;
            //            System.out.println("Prediction: " + this.prediction[a]);
            g.setColor(new Color(color, color, color));
            g.fillRect(x, y, 50, 50);
            g.setColor(Color.white);
            g.drawString(STR."\{a}", x + 20, y - 5);
            x += 50 + gap;
            if (x + 50 >= getWidth() - 10) {
                y += 50 + gap;
                x = 10;
            }
        }
    }

    public void triggerNetworkPrediction() {
        int[][] dataGrid = MNISTFrame.getDataGrid();
        double[] mappedGridData = new double[dataGrid.length * dataGrid[0].length];
        int index = 0;
        for (int a = 0; a < dataGrid.length; a++) {
            for (int b = 0; b < dataGrid[a].length; b++) {
                int gridCell = dataGrid[b][a];
                mappedGridData[index++] = gridCell / 255.0;
                System.out.print(gridCell == 0 ? "#" : "_");
            }
            System.out.println();
        }
        Tensor input = new Tensor(mappedGridData, 1, 28, 28);
        this.prediction = MNISTFrame.processNetworkInputs(input);
        System.out.println(STR."Prediction: \{this.prediction}");
        System.out.println();
    }
}