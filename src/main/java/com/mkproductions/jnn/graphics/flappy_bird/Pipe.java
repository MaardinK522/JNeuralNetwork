package com.mkproductions.jnn.graphics.flappy_bird;

import java.awt.*;
import java.util.Arrays;

import static com.mkproductions.jnn.graphics.flappy_bird.FlappyJPanel.height;
import static com.mkproductions.jnn.graphics.flappy_bird.FlappyJPanel.width;

public class Pipe {
    public int x;
    private final int upperPipeHeight;
    private final int lowerPipeHeight;
    private final int gap;
    public final int pipeWidth;
    public boolean hasPassed = false;

    public Pipe() {
        this.pipeWidth = 50 + 10;
        this.x = width - this.pipeWidth;
        this.gap = 150;
        this.upperPipeHeight = getRandomNumber(gap, height - gap * 2);
        this.lowerPipeHeight = height - upperPipeHeight - gap - 2;
        System.out.println(Arrays.toString(getNetworkInputs()));
    }

    public void show(Graphics g) {
        g.setColor(Color.green);
        g.drawRect(this.x, 0, this.pipeWidth, this.upperPipeHeight);
        g.drawRect(this.x, this.upperPipeHeight + this.gap, this.pipeWidth, this.lowerPipeHeight);
    }

    public void update() {
        this.x -= 5;
    }

    public boolean isCollidingToUpperPipe(Bird bird) {
        int R = bird.size / 2;
        int X1 = x;
        int Y1 = 0;
        int X2 = x + pipeWidth;
        int Xc = (int) (bird.x + bird.size / 2);
        int Yc = (int) (bird.y + bird.size / 2);
        int Xn = Math.max(X1, Math.min(Xc, X2));
        int Yn = Math.max(Y1, Math.min(Yc, upperPipeHeight));
        int Dx = Xn - Xc;
        int Dy = Yn - Yc;
        return (Dx * Dx + Dy * Dy) <= R * R;
    }

    public boolean isCollidingToLowerPipe(Bird bird) {
        int R = bird.size / 2;
        int X1 = x;
        int Y1 = this.upperPipeHeight + this.gap;
        int X2 = x + pipeWidth;
        int Xc = (int) (bird.x + bird.size / 2);
        int Yc = (int) (bird.y + bird.size / 2);
        int Xn = Math.max(X1, Math.min(Xc, X2));
        int Yn = Math.max(Y1, Math.min(Yc, this.upperPipeHeight + this.gap + this.lowerPipeHeight));
        int Dx = Xn - Xc;
        int Dy = Yn - Yc;
        return (Dx * Dx + Dy * Dy) <= R * R;
    }

    public double[] getNetworkInputs() {
        return new double[]{(double) x / width, 0.0 / height, (double) (x + this.pipeWidth) / width, (double) upperPipeHeight / height, (double) x / width, (double) (this.upperPipeHeight + this.gap) / height, (double) (x + this.pipeWidth) / width, 1};
    }

    public static int getRandomNumber(int min, int max) {
        return (int) (Math.random() * (max - min + 1)) + min;
    }
}
