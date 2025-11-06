package com.mkproductions.jnn.graphics.mnist;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class MNISTDrawningJPanel extends JPanel {
    public int mouseX;
    public int mouseY;
    private boolean mousePressed1 = false;
    private boolean mousePressed2 = false;

    MNISTDrawningJPanel(int w, int h) {
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                super.mousePressed(e);
                if (SwingUtilities.isLeftMouseButton(e)) {
                    mousePressed1 = true;
                } else if (SwingUtilities.isRightMouseButton(e)) {
                    mousePressed2 = true;
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                super.mouseReleased(e);
                if (SwingUtilities.isLeftMouseButton(e)) {
                    mousePressed1 = false;
                } else if (SwingUtilities.isRightMouseButton(e)) {
                    mousePressed2 = false;
                }
            }
        });
        // Configuring JPanel properties.
        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                super.mouseDragged(e);
                mouseX = e.getX();
                mouseY = e.getY();
            }

            @Override
            public void mouseMoved(MouseEvent e) {
                super.mouseMoved(e);
                mouseX = e.getX();
                mouseY = e.getY();
            }
        });
        setSize(w, h);
        setVisible(true);
        setBackground(Color.DARK_GRAY);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        long startTime = System.nanoTime();
        int cellCount = 28;
        int cellDeltaX = getWidth() / cellCount;
        int cellDeltaY = getHeight() / cellCount;
        int x = 0;
        int[][] dataGrid = MNISTFrame.getDataGrid();
        for (int a = 0; a < dataGrid.length; a++) {
            int y = 0;
            for (int b = 0; b < dataGrid[0].length; b++) {
                if (mousePressed1 || mousePressed2) {
                    if (this.mouseX >= a * cellDeltaX && this.mouseX <= (a * cellDeltaX) + cellDeltaX) {
                        if (this.mouseY >= b * cellDeltaY && this.mouseY <= (b * cellDeltaY) + cellDeltaY) {
                            if (mousePressed1) {
                                dataGrid[a][b] = 255;
                            } else if (mousePressed2) {
                                dataGrid[a][b] = 0;
                            }
                        }
                    }
                }
                int color = dataGrid[a][b];
                g.setColor(new Color(color, color, color));
                g.fillRect(x, y, cellDeltaX, cellDeltaY);
                y += cellDeltaY;
            }
            x += cellDeltaX;
        }
        double fps = 100000000.0 / (System.nanoTime() - startTime);
        g.setColor(Color.white);

        g.drawString(STR."\{(int) fps}", 10, 20);
        g.drawString(STR."Mouse pressed: \{mousePressed1}", 10, 35);
        g.drawString(STR."Mouse coordinates: (\{mouseX},  \{mouseY})", 10, 50);
        g.drawString(STR."Network accuracy: \{MNISTFrame.getNetworkAccuracy()}%", 10, 65);
        g.drawString(STR."Auto-training status: \{MNISTFrame.getTrainNetworkStatus()}", 10, 80);
        g.drawString(STR."Network loss function: \{MNISTFrame.getNetworkLoss()}", 10, 95);
        g.drawString(STR."Network optimizer: \{MNISTFrame.getJNeuralNetworkOptimizer()}", 10, 110);
        g.drawString(STR."Training epochs: \{MNISTFrame.getTrainingEpochs()}", 10, 125);
        g.drawString(STR."Learning rate: \{String.format("%.4f", MNISTFrame.getLearningRate())}", 10, 140);
    }
}