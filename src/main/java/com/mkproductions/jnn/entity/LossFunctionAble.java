package com.mkproductions.jnn.entity;

public interface LossFunctionAble {
    Matrix getLossFunctionMatrix(Matrix prediction, Matrix target);
}