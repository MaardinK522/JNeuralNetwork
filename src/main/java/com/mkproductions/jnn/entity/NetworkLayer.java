package com.mkproductions.jnn.entity;

import com.mkproductions.jnn.entity.activationFunctions.ActivationFunctionManager;


public record NetworkLayer(
        int numberOfNodes,
        ActivationFunctionManager.NetworkActivationFunction activationFunction
) {
    // Layer for JGPUNeuralNetwork.
}
