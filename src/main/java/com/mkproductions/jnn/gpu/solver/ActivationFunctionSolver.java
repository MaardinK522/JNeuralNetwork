package com.mkproductions.jnn.gpu.solver;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

public class ActivationFunctionSolver {
    // Activation function operations.
    public static void applyActivationFunctionSigmoid(Matrix2DDouble matrix) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                matrix.set(i, j, sigmoid(matrix.get(i, j)));
            }
        }
    }

    public static void applyActivationFunctionReLu(Matrix2DDouble matrix) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                matrix.set(i, j, relu(matrix.get(i, j)));
            }
        }
    }

    public static void applyActivationFunctionTanH(Matrix2DDouble matrix) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                matrix.set(i, j, tanh(matrix.get(i, j)));
            }
        }
    }

    public static void applyActivationFunctionLinear(Matrix2DDouble matrix) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                matrix.set(i, j, linear(matrix.get(i, j)));
            }
        }
    }


    public static void applyActivationFunctionSigmoidDerivative(Matrix2DDouble matrix) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                matrix.set(i, j, sigmoidDerivative(matrix.get(i, j)));
            }
        }
    }

    public static void applyActivationFunctionReLuDerivative(Matrix2DDouble matrix) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                matrix.set(i, j, reluDerivative(matrix.get(i, j)));
            }
        }
    }

    public static void applyActivationFunctionTanHDerivative(Matrix2DDouble matrix) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                matrix.set(i, j, tanhDerivative(matrix.get(i, j)));
            }
        }
    }

    public static void applyActivationFunctionLinearDerivative(Matrix2DDouble matrix) {
        for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                matrix.set(i, j, linearDerivative(matrix.get(i, j)));
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