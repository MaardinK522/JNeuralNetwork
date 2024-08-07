package com.mkproductions.jnn.graphics.flappy_bird;

import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class FlappyJFrame extends JFrame {
    private static final int FPS = 60;
    private static boolean running;
    private final FlappyJPanel flappyJPanel;

    public FlappyJFrame() {
        int w = 1280;
        int h = 720;
        this.flappyJPanel = new FlappyJPanel(w, h);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowOpened(WindowEvent e) {
                super.windowOpened(e);
                System.out.println("Flappy bird has started.");
            }

            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                running = false;
            }

            @Override
            public void windowClosed(WindowEvent e) {
                super.windowClosed(e);
                System.out.println("Flappy bird has stopped.");
            }
        });
        setLayout(null);
        setTitle("Flappy bird !!!");
        setSize(w, h);
        add(this.flappyJPanel);
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    public void startRendering() {
        running = true;
        while (running) {
            this.flappyJPanel.repaint();
            try {
                Thread.sleep(1000 / FPS);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        dispose();
        System.exit(-200);
    }
}
