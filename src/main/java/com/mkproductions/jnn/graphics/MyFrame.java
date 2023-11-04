package com.mkproductions.jnn.graphics;

import javax.swing.*;
import java.awt.event.*;

public class MyFrame extends JFrame {
    int WIDTH = 500;
    int HEIGHT = 500;
    private final String frameName;
    private final MainWindowPanel windowPanel;
    private Timer frameTimer = null;
    public static int mouseX;
    public static int mouseY;

    public MyFrame(String frameName) {
        this.frameName = frameName;
        this.windowPanel = new MainWindowPanel(WIDTH, HEIGHT);
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
                        if (windowPanel.getCellSize() < 100) windowPanel.setCellSize(windowPanel.getCellSize() + 5);
                    }
                    case KeyEvent.VK_DOWN -> {
                        if (windowPanel.getCellSize() >= 10) windowPanel.setCellSize(windowPanel.getCellSize() - 5);
                    }
                }
            }
        });
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                stopRendering();
            }

            @Override
            public void windowClosed(WindowEvent e) {
                super.windowClosed(e);
                stopRendering();
            }
        });
//        setUndecorated(true);
        setTitle(this.frameName);
        setSize(WIDTH, HEIGHT);
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    public void startRendering() {
        if (this.frameTimer != null) this.frameTimer.start();
        System.out.println("|" + frameName + "| JFrame is getting started.");
    }

    public void stopRendering() {
        if (this.frameTimer != null) this.frameTimer.stop();
        System.out.println("|" + this.frameName + "| JFrame is getting closed.");
    }

    public void setFrameTimer(Timer timer) {
        this.frameTimer = timer;
    }
}
