package com.mkproductions.jnn.graphics.training_view;

import java.awt.*;

public abstract class NeuralNetworkRenderAble {
    private double x;
    private double y;
    private double width;
    private double height;
    private final Color color;

    protected NeuralNetworkRenderAble(double x, double y, double width, double height, Color color) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = color;
    }

    public void setX(double x) {
        this.x = x;
    }

    public void setY(double y) {
        this.y = y;
    }

    public void setWidth(double width) {
        this.width = width;
    }

    public void setHeight(double height) {
        this.height = height;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }

    public void render(Graphics g) {
        g.setColor(this.color);
        g.fillRect((int) this.x, (int) this.y, (int) this.width, (int) this.height);
    }
}
