package com.mkproductions.jnn.entity.lossFunctions;

import com.mkproductions.jnn.entity.LossFunction;
import com.mkproductions.jnn.entity.Matrix;

/**
 * Enum representing different regression loss functions.
 */
public enum RegressionLossFunction implements LossFunction {
    /**
     * Absolute error loss function.
     */
    ABSOLUTE_ERROR {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            // Calculate the squared error for each element in the matrix.
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> prediction.getEntry(row, column) - target.getEntry(row, column));
        }
    },
    /**
     * Squared error loss function.
     */
    SQUARED_ERROR {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Ensure input matrices have the same dimensions
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate the squared error for each element in the matrix
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                var error = prediction.getEntry(row, column) - target.getEntry(row, column);
                return Math.pow(error, 2);
            });
        }
    },
    /**
     * Mean absolute error loss function.
     */
    MEAN_ABSOLUTE_ERROR {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Ensure input matrices have the same dimensions
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate the absolute error for each element in the matrix
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                var error = prediction.getEntry(row, column) - target.getEntry(row, column);
                return Math.abs(error);
            });
        }
    },
    /**
     * Smooth mean absolute error loss function.
     */
    SMOOTH_MEAN_ABSOLUTE_ERROR {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Ensure input matrices have the same dimensions
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate the smooth mean absolute error for each element in the matrix
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                double actual = target.getEntry(row, column);
                double forecast = prediction.getEntry(row, column);
                double absDiff = Math.abs(actual - forecast);
                double absSum = Math.abs(actual) + Math.abs(forecast);
                return absSum == 0 ? 0 : (2 * absDiff) / absSum; // Avoid division by zero
            });
        }
    }, LOG_COSH_LOSS {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // ... (similar structure as other loss functions)
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                double error = prediction.getEntry(row, column) - target.getEntry(row, column);
                return Math.log(Math.cosh(error));
            });
        }
    };
//    HUBER_LOSS {
//        @Override
//        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target, double delta) {
//            // ... (similar structure as other loss functions)
//            // You can adjust this hyperparameter
//            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
//                double error = prediction.getEntry(row, column) - target.getEntry(row, column);
//                return error * error <= delta * delta ? 0.5 * error * error : delta * (Math.abs(error) - delta / 2);
//            });
//        }
//    },
//    QUANTILE_LOSS {
//        @Override
//        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
//            // ... (similar structure as other loss functions)
//            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
//                double error = prediction.getEntry(row, column) - target.getEntry(row, column);
//                return quantile * Math.max(error, 0) + (1 - quantile) * Math.min(error, 0);
//            });
//        }
//    }
}
