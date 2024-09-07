package com.mkproductions.jnn.entity;

import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

public interface LossFunctionAble {
    Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target);

    Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target);
}