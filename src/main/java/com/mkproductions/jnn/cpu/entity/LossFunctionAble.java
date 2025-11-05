package com.mkproductions.jnn.cpu.entity;

public interface LossFunctionAble {
    Tensor getLossFunctionTensor(Tensor prediction, Tensor target);

    Tensor getDerivativeTensor(Tensor prediction, Tensor target);
}