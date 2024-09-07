package com.mkproductions.jnn.entity.activationFunctions;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

public class ActivationFunctionManager {
    // Activation function operations.
    public static void applyActivationFunction(Matrix2DDouble matrix, NetworkActivationFunction activationFunction) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                double activatedValue = switch (activationFunction) {
                    case SIGMOID -> sigmoid(matrix.get(i, j));
                    case RE_LU -> relu(matrix.get(i, j));
                    case LINEAR -> linear(matrix.get(i, j));
                    case TAN_H -> tanh(matrix.get(i, j));
                };
                matrix.set(i, j, activatedValue);
            }
        }
    }

    public static void applyActivationFunctionDerivative(Matrix2DDouble matrix, NetworkActivationFunction activationFunction) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                double activatedValue = switch (activationFunction) {
                    case SIGMOID -> sigmoidDerivative(matrix.get(i, j));
                    case RE_LU -> reluDerivative(matrix.get(i - 1, j));
                    case LINEAR -> linearDerivative(matrix.get(i, j));
                    case TAN_H -> tanhDerivative(matrix.get(i, j));
                };
                matrix.set(i, j, activatedValue);
            }
        }
    }

    // Equations of the network's activation functions.
    private static double sigmoid(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    private static double relu(double x) {
        return Math.max(0.0, x);
    }

    private static double linear(double x) {
        return x;
    }

    private static double tanh(double x) {
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
    }

    // Derivatives of the network's activation function.
    private static double sigmoidDerivative(double y) {
        return y / (1 - y);
    }

    private static double reluDerivative(double y) {
        return y < 0 ? 0 : 1;
    }

    private static double linearDerivative(double y) {
        return 1;
    }

    private static double tanhDerivative(double y) {
        return 1 - (y * y);
    }

    public enum NetworkActivationFunction {
        SIGMOID, RE_LU, LINEAR, TAN_H
    }
}
