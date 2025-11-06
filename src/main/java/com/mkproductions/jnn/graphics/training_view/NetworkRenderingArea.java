package com.mkproductions.jnn.graphics.training_view;

import java.awt.*;

public class NetworkRenderingArea {
    private final Color color;
    final float x;
    final float y;
    final float width;
    final float height;

    public NetworkRenderingArea(float x, float y, float width, float height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = new Color(125, 125, 125);
    }

    public void render(Graphics g) {
        g.setColor(this.color);
        g.drawRect((int) x, (int) y, (int) width, (int) height);
    }

    public void updateNetworkRenderingArea(float x, float y, float width, float height) {
        // TODO: Update all the layers data.
    }
}