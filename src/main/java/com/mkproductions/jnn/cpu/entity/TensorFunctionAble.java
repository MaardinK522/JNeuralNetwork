package com.mkproductions.jnn.cpu.entity;

public interface TensorFunctionAble {
    double map(int row, int column, int depth, double value);
}