package com.mkproductions.jnn.graphics.mnist;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Arrays;


public class MNISTTestingJPanel extends JPanel {
    public int mouseX;
    public int mouseY;
    private boolean mousePressed = false;

    MNISTTestingJPanel(int w, int h) {
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                super.mousePressed(e);
                mousePressed = true;
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                super.mouseReleased(e);
                mousePressed = false;
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
        double[][] dataGrid = MNISTFrame.getDataGrid();
        for (int a = 0; a < dataGrid.length; a++) {
            int y = 0;
            for (int b = 0; b < dataGrid[0].length; b++) {
                if (mousePressed) if (this.mouseX >= a * cellDeltaX && this.mouseX <= (a * cellDeltaX) + cellDeltaX)
                    if (this.mouseY >= b * cellDeltaY && this.mouseY <= (b * cellDeltaY) + cellDeltaY)
                        dataGrid[a][b] = 255;

                int color = (int) dataGrid[a][b];
                g.setColor(new Color(color, color, color));
                g.fillRect(x, y, cellDeltaX, cellDeltaY);
                y += cellDeltaY;
            }
            x += cellDeltaX;
        }
        double fps = 100000000.0 / (System.nanoTime() - startTime);
        g.setColor(Color.white);
        g.drawString("FPS: " + (int) fps, 10, 20);
        g.drawString("Mouse pressed: " + mousePressed, 10, 35);
        g.drawString("Network accuracy: " + MNISTFrame.networkAccuracy + "%", 10, 50);
    }
}
