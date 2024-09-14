package com.mkproductions.jnn.gpu.entity;

import com.mkproductions.jnn.gpu.solver.ActivationFunctionSolver;


public record NetworkLayer(
        int numberOfNodes,
        ActivationFunctionSolver.NetworkActivationFunction activationFunction
) {
    // Layer for JGPUNeuralNetwork.
}
