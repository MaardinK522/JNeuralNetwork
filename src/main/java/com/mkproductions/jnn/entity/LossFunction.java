package com.mkproductions.jnn.entity;

public interface LossFunction {
    Matrix getLossFunctionMatrix(Matrix prediction, Matrix target);
}