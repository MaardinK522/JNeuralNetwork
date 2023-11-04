package com.mkproductions.jnn.entity;


import com.mkproductions.jnn.network.JNeuralNetwork;

public class Mapper {

    public static double mapRangeToRange(double value, double fromMin, double fromMax, double toMin, double toMax) {
        return toMin + (value - fromMin) * (toMax - toMin) / (fromMax - fromMin);
    }

    public static double mapPredictionToRange(double prediction, MapAble activationFunction, int fromValue, int toValue) {
        double output = 0;
        if (activationFunction.equals(JNeuralNetwork.SIGMOID_ACTIVATION)) output = prediction * toValue;
        if (activationFunction.equals(JNeuralNetwork.TANH_ACTIVATION))
            output = mapRangeToRange(prediction, -1, 1, fromValue, toValue);
        if (activationFunction.equals(JNeuralNetwork.RELU_ACTIVATION)) output = (prediction < 0) ? 0 : 1;
        return output;
    }
}

