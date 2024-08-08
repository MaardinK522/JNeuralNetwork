package com.mkproductions.jnn.entity.lossFunctions;

import com.mkproductions.jnn.entity.LossFunction;
import com.mkproductions.jnn.entity.Matrix;

public enum ClassificationLossFunction implements LossFunction {
    BINARY_CROSS_ENTROPY {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Check for dimension compatibility
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate binary cross-entropy loss for each element
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                double y = target.getEntry(row, column);
                double p = prediction.getEntry(row, column);
                // Clip probabilities to prevent log(0)
                p = Math.max(p, 1e-15);
                p = Math.min(p, 1 - 1e-15);
                return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
            });
        }
    }, CATEGORICAL_CROSS_ENTROPY {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Check for dimension compatibility
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate categorical cross-entropy loss for each element
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                double y = target.getEntry(row, column);
                double p = prediction.getEntry(row, column);
                // Clip probabilities to prevent log(0)
                p = Math.max(p, 1e-15);
                p = Math.min(p, 1 - 1e-15);
                return -y * Math.log(p);
            });
        }
    }, SPARSE_CATEGORICAL_CROSS_ENTROPY {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Check for dimension compatibility
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate sparse categorical cross-entropy loss for each element
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                int trueClass = (int) target.transpose().getEntry(row, 0); // Assuming target is a single column matrix with class indices
                double predictedProb = prediction.getEntry(row, trueClass);
                predictedProb = Math.max(predictedProb, 1e-15); // Clip to prevent log(0)
                return -Math.log(predictedProb);
            });
        }
    }, HINGE_LOSS {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Check for dimension compatibility
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate hinge loss for each element
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, val) -> Math.max(0, 1 - (prediction.getEntry(row, column) * target.getEntry(row, column))));
        }
    }
}
