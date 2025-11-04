package com.mkproductions.jnn.graphics.flappy_bird;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.layers.DenseLayer;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;

public class FlappyJPanel extends JPanel {
    private final int birdsCount = 200;
//    private final GeneticAlgorithm<Bird, JNeuralNetwork> birdsGeneticAlgorithm;
    private final ArrayList<Pipe> pipes;
    private final ArrayList<Bird> birds;
    public static int frameCount = 0;
    public int mouseX;
    public int mouseY;
    public static int width = 0;
    public static int height = 0;
    private int currentGenScore = 0;
    private int maxScore;
    private int aliveBirds;

    public FlappyJPanel(int width, int height) {
        FlappyJPanel.width = width;
        FlappyJPanel.height = height;
        this.maxScore = 0;
//        birdsGeneticAlgorithm = new GeneticAlgorithm<>(this.birdsCount, Bird::new);
//        birdsGeneticAlgorithm.setMutationRate(0.5);
        this.birds = new ArrayList<>();
        this.pipes = new ArrayList<>();
        this.aliveBirds = this.birdsCount;
        prepareNewGeneration();
        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                super.mouseMoved(e);
                mouseX = e.getX();
                mouseY = e.getY();
            }
        });
        setSize(width, height);
        setVisible(true);
        setBackground(Color.BLACK);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (frameCount % 60 == 0) {
            pipes.add(new Pipe());
        }
        if (!pipes.isEmpty()) {
            Pipe pipe = pipes.get(0);
            for (int a = 0; a < birds.size(); a++) {
                Bird bird = birds.get(a);
                bird.think(pipe);
                bird.show(g);
                if (bird.isAlive) {
                    if (!pipe.hasPassed) {
                        if (pipe.isCollidingToUpperPipe(bird)) {
                            bird.dead();
                        }
                        if (pipe.isCollidingToLowerPipe(bird)) {
                            bird.dead();
                        }
                        if (pipe.x + pipe.pipeWidth < bird.x) {
                            pipe.hasPassed = true;
                            this.currentGenScore++;
                        }
                    }
                    bird.update();
                } else {
                    birds.remove(bird);
                    aliveBirds--;
                }
            }
            for (Pipe p : this.pipes) {
                p.show(g);
                p.update();
            }
        }
        g.setColor(Color.white);
//        g.drawString("Max score: " + this.maxScore, 20, 20);
//        g.drawString("Gen " + birdsGeneticAlgorithm.getGenerationCount() + " score: " + this.currentGenScore, 20, 35);
//        g.drawString("Birds: " + aliveBirds, 20, 50);
//        g.drawString("Gen: " + birdsGeneticAlgorithm.getGenerationCount(), 20, 65);
        if (aliveBirds < 1 && this.currentGenScore > 0) {
            prepareNewChildGeneration();
        } else if (aliveBirds < 1) {
            System.out.println("Alive count: " + aliveBirds);
            System.out.println("Preparing new generation of the bird.");
            prepareNewGeneration();
        }
        if (maxScore <= currentGenScore) maxScore = currentGenScore;
        frameCount++;
        clearPassPipe();
    }

    private void prepareNewChildGeneration() {
        double sum = 0;
        for (Bird bird : this.birds)
            sum += bird.score;
        for (Bird bird : this.birds)
            bird.fitness = bird.score / sum;
    }

    private void prepareNewGeneration() {
        this.pipes.clear();
        this.birds.clear();
        Object[] networkParameters = {
                10,
                new DenseLayer(16, ActivationFunction.SIGMOID),
                new DenseLayer(2, ActivationFunction.SIGMOID)
        };
//        this.birds.addAll(this.birdsGeneticAlgorithm.getRandomGeneration(networkParameters));
        this.aliveBirds = this.birdsCount;
        this.currentGenScore = 0;
    }

    private void clearPassPipe() {
        if (!this.pipes.isEmpty()) {
            Pipe pipe = pipes.get(0);
            if (pipe.x + pipe.pipeWidth < 0) this.pipes.remove(pipe);
        }
    }
}