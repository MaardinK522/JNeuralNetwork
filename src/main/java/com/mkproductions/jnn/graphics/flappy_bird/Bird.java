package com.mkproductions.jnn.graphics.flappy_bird;

import com.mkproductions.jnn.network.GeneticAlgorithm;
import com.mkproductions.jnn.network.JNeuralNetwork;

import java.awt.*;
import java.util.Random;

import static com.mkproductions.jnn.graphics.flappy_bird.FlappyJPanel.height;
import static com.mkproductions.jnn.graphics.flappy_bird.FlappyJPanel.width;

public class Bird extends GeneticAlgorithm.MutableChromosome {
    public final double x = 100;
    public double y;
    private double vel = 0;
    public final double gravity = 0.5F;
    private final int lift;
    final int size;
    public Color birdColor;
    public boolean isAlive;
    private final JNeuralNetwork jNeuralNetwork;

    public int score = 0;

    public Bird(JNeuralNetwork jNeuralNetwork) {
        this.y = (double) (height / 2.0);
        this.size = 50;
        this.lift = -10;
        this.isAlive = true;
        Random random = new Random();
        this.birdColor = new Color(random.nextFloat(), random.nextFloat(), random.nextFloat(), 0.5f);
//        this.jNeuralNetwork = new JNeuralNetwork(10, new Layer(16, ActivationFunction.SIGMOID), new Layer(2, ActivationFunction.SIGMOID));
        this.jNeuralNetwork = new JNeuralNetwork(jNeuralNetwork);
    }

    public void think(Pipe pipe) {
        if (this.isAlive) {
            double[] pipeInputs = pipe.getNetworkInputs();
            double[] inputs = new double[10];
            inputs[0] = (double) ((x + size / 2.0) / width);
            inputs[1] = (double) ((y + size / 2.0) / height);
            System.arraycopy(pipeInputs, 0, inputs, 2, pipeInputs.length);
            double[] outputs = this.jNeuralNetwork.processInputs(inputs);
            if (outputs[0] > outputs[1]) {
                jump();
            }
        }
    }

    public void show(Graphics g) {
        g.setColor(birdColor);
        g.drawRect((int) this.x, (int) this.y, this.size, this.size);
        g.fillOval((int) this.x, (int) this.y, this.size, this.size);
    }

    public void update() {
        this.score++;
        this.vel += this.gravity;
        this.y += this.vel;
        if (this.y + this.size >= height) {
            this.y = height - size;
            this.dead();
        }
        if (this.y <= 0) {
            this.y = 0;
            this.dead();
        }
    }

    public void jump() {
        if (this.isAlive) this.vel += this.lift;
    }

    public void dead() {
        this.isAlive = false;
        this.birdColor = new Color(255, 0, 0, (int) (255 * 0.5));
    }

    @Override
    public void mutateGene(Random random, double mutationRate) {
//        this.jNeuralNetwork.mutate(random, mutationRate);
    }
}
