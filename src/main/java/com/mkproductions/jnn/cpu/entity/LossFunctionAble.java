package com.mkproductions.jnn.cpu.entity;

public interface LossFunctionAble {
    Matrix getLossFunctionMatrix(Matrix prediction, Matrix target);

    Matrix getDerivativeMatrix(Matrix prediction, Matrix target);
}