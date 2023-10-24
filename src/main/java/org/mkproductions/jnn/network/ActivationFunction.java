package org.mkproductions.jnn.network;

public class ActivationFunction {
    public static final int SIGMOID = 0;
    public static final int TAN_H = 1;
    public static final int RE_LU = 2;

    /**
     * Activate function converts the input into output as per the given activation function.
     *
     * @param ACTIVATION_FUNCTION should only be the SIGMOID,TAN_H and RE_Lu.
     * @param x                   is the raw output of the node.
     * @return Activation value as per the given ACTIVATION_FUNCTION.
     */
    public static double activate(int ACTIVATION_FUNCTION, double x) {
        return switch (ACTIVATION_FUNCTION) {
            case SIGMOID -> 1 / (1 + Math.exp(-x));
            case TAN_H -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
            case RE_LU -> Math.max(0, x);
            default -> 0;
        };
    }

    /**
     * Activate function converts the activated output into the raw inputs as per the given activation function.
     *
     * @param ACTIVATION_FUNCTION should only be the SIGMOID,TAN_H and RE_Lu.
     * @param y                   is the raw output of the node.
     * @return Activation value as per the given SIGMOID.
     */
    public static double deactivate(int ACTIVATION_FUNCTION, double y) {
        return switch (ACTIVATION_FUNCTION) {
            case SIGMOID -> y * (1 - y);
            case TAN_H -> 1 - (y * y);
            case RE_LU -> (y < 0) ? 0 : y + 0.0;
            default -> 0;
        };
    }
}

