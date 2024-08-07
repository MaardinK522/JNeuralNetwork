package com.mkproductions.jnn.graphics.training_view;

import com.mkproductions.jnn.network.JNeuralNetwork;

import javax.swing.*;
import java.awt.*;

public class NeuralNetworkViewrJPanel extends JPanel {
    private final JNeuralNetwork jNeuralNetwork;
    private final double[][] trainingInputs;
    private final double[][] trainingOutputs;
    private int epochs;
    private final NetworkElementsIntroducer networkElementsIntroducer;
    private final NetworkRenderingArea networkRenderingArea;

    public NeuralNetworkViewrJPanel(JNeuralNetwork jNeuralNetwork, double[][] trainingInputs, double[][] trainingOutputs) {
        this.jNeuralNetwork = jNeuralNetwork;
        this.trainingInputs = trainingInputs;
        this.trainingOutputs = trainingOutputs;
        this.epochs = 100;
        this.networkElementsIntroducer = new NetworkElementsIntroducer(0, 0, 200, getHeight());
        this.networkRenderingArea = new NetworkRenderingArea(this.networkElementsIntroducer.getWidth() + 50, 0, getWidth() - this.networkElementsIntroducer.getWidth() - 50, getHeight() - 300);
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        setBackground(Color.WHITE);
        renderLayers(g);
        renderNeuralNetworkTrainingArea(g);
    }

    private void renderLayers(Graphics g) {
        // TODO: Implement rendering of layers views
    }

    private void renderNeuralNetworkTrainingArea(Graphics g) {
        // TODO: Implement neural networks training views.
        networkRenderingArea.render(g);
    }

    public void trainNetwork() {
        int epochs = this.epochs;
        this.jNeuralNetwork.train(this.trainingInputs, trainingOutputs, epochs);
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }
}
