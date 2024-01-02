package com.mkproductions.jnn.graphics.xor;

import javax.swing.*;
import java.awt.event.*;

public class XORFrame extends JFrame {
    final int w = 560;
    final int h = 560;
    public static int mouseX;
    public static int mouseY;
    private final XORWindowPanel windowPanel;
    private boolean running = false;

    public XORFrame(String frameName) {
        this.windowPanel = new XORWindowPanel(this.w, this.h);
        add(windowPanel);
        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                super.mouseMoved(e);
                mouseX = e.getX() - 8;
                mouseY = e.getY() - 31;
            }
        });
        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                super.keyPressed(e);
                switch (e.getKeyCode()) {
                    case KeyEvent.VK_Q -> {
                        if (e.isControlDown()) dispose();
                    }
                    case KeyEvent.VK_UP -> {
                        if (windowPanel.getCellCount() <= 100) windowPanel.setCellCount(windowPanel.getCellCount() + 10);
                    }
                    case KeyEvent.VK_DOWN -> {
                        if (windowPanel.getCellCount() >= 1) windowPanel.setCellCount(windowPanel.getCellCount() - 10);
                    }
                    case KeyEvent.VK_SPACE -> running = false;
                }
            }
        });
        setTitle(frameName);
        setSize(this.w, this.h);
        setLocationRelativeTo(null);
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    public void startRendering() {
        running = true;
        while (running) {
            this.windowPanel.repaint();
        }
        dispose();
    }
}
