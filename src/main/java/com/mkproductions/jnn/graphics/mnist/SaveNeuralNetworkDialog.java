package com.mkproductions.jnn.graphics.mnist;

import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.function.Function;

public class SaveNeuralNetworkDialog extends JDialog {
    public SaveNeuralNetworkDialog(Frame owner, Function<String, SaveStatus> saveNeuralNetwork) {
        super(owner, "Save Neural Network", true);
        setLocationRelativeTo(owner);
        setLayout(new FlowLayout());
        Font font1 = new Font("System", Font.BOLD, 20);
        JLabel titleLabel = new JLabel("Enter a name for the neural network:");
        titleLabel.setFont(font1);
        add(titleLabel);
        JTextField neuralNetworkNameTextField = new JTextField(20);
        neuralNetworkNameTextField.setFont(font1);
        add(neuralNetworkNameTextField);
        JButton saveButton = getJButton(saveNeuralNetwork, font1, neuralNetworkNameTextField);
        add(saveButton);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowOpened(WindowEvent e) {
                super.windowOpened(e);
                System.out.println("Network saving window has been opened!");
            }

            @Override
            public void windowClosed(WindowEvent e) {
                super.windowClosed(e);
                System.out.println("Network saving window has been closed!");
            }
        });
        setSize(450, 120);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
    }

    private @NotNull JButton getJButton(Function<String, SaveStatus> saveNeuralNetwork, Font font1, JTextField neuralNetworkNameTextField) {
        JButton saveButton = new JButton("Save");
        saveButton.setFont(font1);
        saveButton.addActionListener(_ -> {
            if (neuralNetworkNameTextField.getText().isEmpty()) {
                JOptionPane.showMessageDialog(this, "Neural network name cannot be empty!");
            } else {
                SaveStatus saveStatus = saveNeuralNetwork.apply(neuralNetworkNameTextField.getText());
                if (saveStatus == SaveStatus.FAILED) {
                    JOptionPane.showMessageDialog(this, "Neural network could not be saved!");
                } else if (saveStatus == SaveStatus.FILE_EXISTS) {
                    JOptionPane.showMessageDialog(this, "Neural network file already exists!");
                } else if (saveStatus == SaveStatus.SAVED) {
                    dispose();
                    JOptionPane.showMessageDialog(this, "Neural network saved successfully!");
                }
            }
        });
        return saveButton;
    }
}