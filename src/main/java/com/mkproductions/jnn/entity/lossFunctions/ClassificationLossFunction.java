package com.mkproductions.jnn.entity.lossFunctions;

import com.mkproductions.jnn.entity.LossFunction;
import com.mkproductions.jnn.entity.Matrix;

public enum ClassificationLossFunction implements LossFunction {
    BINARY_CROSS_ENTROPY {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (r, c, val) -> {
                var y = target.getEntry(r, c);
                var p = prediction.getEntry(r, c);
                p = Math.max(p, 1e-15);
                p = Math.min(p, 1 - 1e-15);
                return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
            });
        }
    }, HINGE_LOSS {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, val) -> Math.max(0, 1 - (prediction.getEntry(row, column) * target.getEntry(row, column))));
        }
    }
}
