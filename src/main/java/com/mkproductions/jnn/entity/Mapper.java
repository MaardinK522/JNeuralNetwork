package com.mkproductions.jnn.entity;


import com.mkproductions.jnn.entity.activationFunctions.ActivationFunction;

public class Mapper {

    public static double mapRangeToRange(double value, double fromMin, double fromMax, double toMin, double toMax) {
        return toMin + (value - fromMin) * (toMax - toMin) / (fromMax - fromMin);
    }

    public static double mapPredictionToRange(double prediction, ActivationFunction activationFunction, int fromValue, int toValue) {
        if (activationFunction.name().equals(ActivationFunction.SIGMOID.name()))
            return prediction * toValue;
        else if (activationFunction.name().equals(ActivationFunction.TAN_H.name()))
            return mapRangeToRange(prediction, -1, 1, fromValue, toValue);
        else if (activationFunction.name().equals(ActivationFunction.RE_LU.name()))
            return (prediction < 0) ? 0 : toValue;
        else if (activationFunction.name().equals(ActivationFunction.LINEAR.name()))
            return prediction * toValue;
        return 0;
    }
}

