package com.mkproductions.jnn.graphics.training_view;

public class NetworkElementsIntroducer {
    float x;
    float y;
    private float width;
    private float height;

    public NetworkElementsIntroducer(float x, float y, float width, float height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public float getWidth() {
        return width;
    }

    public float getHeight() {
        return height;
    }
}

