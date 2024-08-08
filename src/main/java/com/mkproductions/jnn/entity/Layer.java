package com.mkproductions.jnn.entity;

import com.mkproductions.jnn.entity.activationFunctions.ActivationFunction;

public record Layer(int numberOfNodes, ActivationFunction activationFunction) {
}
