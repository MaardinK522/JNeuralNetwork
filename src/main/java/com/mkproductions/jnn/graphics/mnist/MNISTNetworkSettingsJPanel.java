package com.mkproductions.jnn.graphics.mnist;

import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.networks.JNeuralNetwork;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;

import javax.swing.*;
import java.awt.*;
import java.util.function.Function;

public class MNISTNetworkSettingsJPanel extends JPanel {
    private final JComboBox<LossFunction> lossFunctionComboBox = new JComboBox<>();
    private final JComboBox<JNetworkOptimizer> optimizerComboBox = new JComboBox<>();
    private final JComboBox<Boolean> debugModeStatus = new JComboBox<>();
    private final JComboBox<Boolean> autoTrainingModeStatus = new JComboBox<>();
    private final JTextField learningRateTextField = new JTextField(10);
    private final JTextField epochCountTextField = new JTextField(10);

    public MNISTNetworkSettingsJPanel(int w, int h, Function<Void, Void> clearGrid, Function<Void, Void> trainNeuralNetwork, Function<Void, Void> resetNeuralNetwork,
            Function<Void, Void> triggerNetworkPrediction, Function<Void, Void> saveNeuralNetwork) {
        JNeuralNetwork jNeuralNetwork = MNISTFrame.getJNeuralNetwork();
        JButton updateButton = new JButton("Update");
        JButton clearGridButton = new JButton("Clear Grid");
        JButton trainButton = new JButton("Train");
        JButton resetNeuralNetworkButton = new JButton("Reset Neural Network");
        JButton triggerNetworkPredictionButton = new JButton("Predict");
        JButton saveNeuralNetworkButton = new JButton("Save Neural Network");

        lossFunctionComboBox.setModel(new DefaultComboBoxModel<>(LossFunction.values()));
        optimizerComboBox.setModel(new DefaultComboBoxModel<>(JNetworkOptimizer.values()));
        debugModeStatus.setModel(new DefaultComboBoxModel<>(new Boolean[] { true, false }));
        autoTrainingModeStatus.setModel(new DefaultComboBoxModel<>(new Boolean[] { true, false }));
        learningRateTextField.setText("" + jNeuralNetwork.getLearningRate());
        epochCountTextField.setText("" + MNISTFrame.getTrainingEpochs());

        updateButton.addActionListener(_ -> {
            MNISTFrame.getJNeuralNetwork() // Getting network data
                    .setLossFunctionAble(lossFunctionComboBox.getItemAt(lossFunctionComboBox.getSelectedIndex())) // Loss function
                    .setJNeuralNetworkOptimizer(optimizerComboBox.getItemAt(optimizerComboBox.getSelectedIndex())) // optimizer
                    .setDebugMode(debugModeStatus.getItemAt(debugModeStatus.getSelectedIndex())) // debug mode
                    .setLearningRate(Double.parseDouble(learningRateTextField.getText())); // Learning rate
            MNISTFrame.setEpochCount(Integer.parseInt(epochCountTextField.getText()));
            MNISTFrame.setAutoTrainingMode(autoTrainingModeStatus.getItemAt(autoTrainingModeStatus.getSelectedIndex()));
        }); // Auto training mode
        clearGridButton.addActionListener(_ -> clearGrid.apply(null));
        trainButton.addActionListener(_ -> trainNeuralNetwork.apply(null));
        resetNeuralNetworkButton.addActionListener(_ -> resetNeuralNetwork.apply(null));
        triggerNetworkPredictionButton.addActionListener(_ -> triggerNetworkPrediction.apply(null));
        saveNeuralNetworkButton.addActionListener(_ -> saveNeuralNetwork.apply(null));

        lossFunctionComboBox.setSelectedItem(jNeuralNetwork.getLossFunctionable());
        optimizerComboBox.setSelectedItem(jNeuralNetwork.getJNeuralNetworkOptimizer());
        debugModeStatus.setSelectedItem(jNeuralNetwork.isDebugMode());
        autoTrainingModeStatus.setSelectedItem(MNISTFrame.getTrainNetworkStatus());
        learningRateTextField.setText(String.valueOf(jNeuralNetwork.getLearningRate()));
        epochCountTextField.setText(String.valueOf(MNISTFrame.getTrainingEpochs()));

        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(5, 5, 5, 5);

        gbc.gridx = 0;
        gbc.gridy = 0;
        add(new JLabel("Loss Function:"), gbc);
        gbc.gridx = 1;
        add(lossFunctionComboBox, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        add(new JLabel("Optimizer:"), gbc);
        gbc.gridx = 1;
        add(optimizerComboBox, gbc);

        gbc.gridx = 0;
        gbc.gridy = 2;
        add(new JLabel("Debug Mode:"), gbc);
        gbc.gridx = 1;
        add(debugModeStatus, gbc);

        gbc.gridx = 0;
        gbc.gridy = 3;
        add(new JLabel(" Auto-Training mode:"), gbc);
        gbc.gridx = 1;
        add(autoTrainingModeStatus, gbc);

        gbc.gridx = 0;
        gbc.gridy = 4;
        add(new JLabel("Learning Rate:"), gbc);
        gbc.gridx = 1;
        add(learningRateTextField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 5;
        add(new JLabel("Epoch Count:"), gbc);
        gbc.gridx = 1;
        add(epochCountTextField, gbc);

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        buttonPanel.add(updateButton);
        buttonPanel.add(trainButton);
        buttonPanel.add(resetNeuralNetworkButton);
        buttonPanel.add(triggerNetworkPredictionButton);
        buttonPanel.add(saveNeuralNetworkButton);
        buttonPanel.add(clearGridButton);
        buttonPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        gbc.gridx = 0;
        gbc.gridy = 6;
        gbc.gridwidth = 2;
        add(buttonPanel, gbc);

        setSize(w, h);
        setVisible(true);
        setBackground(Color.white);

    }
}