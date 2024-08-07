package com.mkproductions.jnn.entity.lossFunctions;

import com.mkproductions.jnn.entity.LossFunction;
import com.mkproductions.jnn.entity.Matrix;

public enum RegressionLossFunction implements LossFunction {
    SQUARED_ERROR {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                var error = prediction.getEntry(row, column) - target.getEntry(row, column);
                return Math.pow(error, 2);
            });
        }
    }, MEAN_ABSOLUTE_ERROR {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            return null;
        }
    }, HUBE_LOSS {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            return null;
        }
    }
}