package com.mkproductions.jnn.graphics.training_view;

import com.mkproductions.jnn.networks.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class NeuralNetworkTrainingViewerJFrame extends JFrame implements Runnable {
    private static final int FPS = 60;
    private final NeuralNetworkViewrJPanel neuralNetworkViewerJPanel;
    private boolean running;
    private final Thread jPanelThread;
    private final String title = "JNeuralNetwork Viewer";

    public NeuralNetworkTrainingViewerJFrame(final JNeuralNetwork jNeuralNetwork, final double[][] trainingInputs, final double[][] trainingOutputs) {
        this.neuralNetworkViewerJPanel = new NeuralNetworkViewrJPanel(jNeuralNetwork, trainingInputs, trainingOutputs);
        this.jPanelThread = new Thread(this);
        this.running = false;
        GraphicsDevice graphicsDevice = GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice();
        getContentPane().add(neuralNetworkViewerJPanel, BorderLayout.CENTER);
        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                super.keyPressed(e);
                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                    dispose();
                }
            }
        });
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                System.out.println(title + " is closing.");
            }
        });
        setTitle(this.title);
        setUndecorated(true);
        graphicsDevice.setFullScreenWindow(this);
        setLocationRelativeTo(null);
        setVisible(true);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public void startRendering() {
        this.running = true;
        this.jPanelThread.start();
    }

    @Override
    public void run() {
        while (this.running) {
            this.neuralNetworkViewerJPanel.repaint();
            try {
                Thread.sleep(1000 / FPS);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}