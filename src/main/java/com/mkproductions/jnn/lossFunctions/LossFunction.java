package com.mkproductions.jnn.lossFunctions;

import com.mkproductions.jnn.cpu.entity.LossFunctionAble;
import com.mkproductions.jnn.cpu.entity.Matrix;

public enum LossFunction implements LossFunctionAble {
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
            return Matrix.matrixMapping(prediction, (row, column, value) -> (target.getEntry(row, column) - value));
        }

        @Override
        public Matrix getDerivativeMatrix(Matrix prediction, Matrix target) {
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, _) -> {
                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
                return error >= 0 ? 1 : -1;
            });
        }
    },
    /**
     * Squared error loss function.
     */
    MEAN_SQUARED_ERROR {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Ensure input matrices have the same dimensions
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            // Calculate the squared error for each element in the matrix
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()),
                    (row, column, _) -> Math.pow(prediction.getEntry(row, column) - target.getEntry(row, column), 2));
        }

        @Override
        public Matrix getDerivativeMatrix(Matrix prediction, Matrix target) {
            return Matrix.matrixMapping(new Matrix(prediction.getData()), ((row, column, _) -> -2 * (prediction.getEntry(row, column) - target.getEntry(row, column))));
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
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, _) -> {
                var error = prediction.getEntry(row, column) - target.getEntry(row, column);
                return Math.abs(error);
            });
        }

        @Override
        public Matrix getDerivativeMatrix(Matrix prediction, Matrix target) {
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, _) -> {
                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
                return error >= 0 ? 1 : -1;
            });
        }
    },
    // Smooth mean absolute error loss function.
    // TODO: Implement this loss function with the proper derivative.
    //    SMOOTH_MEAN_ABSOLUTE_ERROR {
    //        @Override
    //        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
    //            // Ensure input matrices have the same dimensions
    //            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
    //                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
    //            }
    //
    //            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
    //                double actual = target.getEntry(row, column);
    //                double forecast = prediction.getEntry(row, column);
    //                double absDiff = Math.abs(forecast - actual);
    //                double absSum = Math.abs(actual) + Math.abs(forecast);
    //                return absSum == 0 ? 0 : (2 * absDiff) / (absSum * absSum);
    //            });
    //        }
    //
    //        private double sign(double x) {
    //            if (x > 0)
    //                return 1;
    //            else if (x < 0)
    //                return -1;
    //            return 0;
    //        }
    //
    //
    //        @Override
    //        public Matrix getDerivativeMatrix(Matrix predictions, Matrix targets) {
    //            return Matrix.matrixMapping(new Matrix(predictions.getRowCount(), predictions.getColumnCount()), (row, column, prediction) -> {
    //                double target = targets.getEntry(row, column);
    //                double diff = target - prediction;
    //                double absDiff = Math.abs(diff);
    //                double absTarget = Math.abs(target);
    //                double absPrediction = Math.abs(prediction);
    //                double absSum = absTarget + absPrediction;
    //                return (2 * sign(diff) * (absSum - absDiff)) / (absSum * absSum);
    //            });
    //        }
    //    },
    /**
     * Log cosh loss function.
     */
    LOG_COSH {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // ... (similar structure as other loss functions)
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, _) -> {
                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
                return Math.log(Math.cosh(error));
            });
        }

        @Override
        public Matrix getDerivativeMatrix(Matrix prediction, Matrix target) {
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, _) -> {
                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
                return Math.tanh(error);
            });
        }
    }, // Binary cross entropy.
    BINARY_CROSS_ENTROPY {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // Check for dimension compatibility
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            // Calculate binary cross-entropy loss for each element
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, _) -> {
                double y = target.getEntry(row, column);
                double p = prediction.getEntry(row, column);
                // Clip probabilities to prevent log(0)
                p = Math.max(p, 1e-15);
                p = Math.min(p, 1 - 1e-15);
                return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
            });
        }

        @Override
        public Matrix getDerivativeMatrix(Matrix prediction, Matrix target) {
            // Ensure input matrices have the same dimensions
            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate the derivative of binary cross-entropy loss for each element
            return Matrix.matrixMapping(prediction, (row, column, _) -> (target.getEntry(row, column) - prediction.getEntry(row, column)));
        }
    }, // Categorical cross entropy.
    CATEGORICAL_CROSS_ENTROPY {
        @Override
        public Matrix getLossFunctionMatrix(Matrix predictions, Matrix targets) {
            // Check for dimension compatibility
            if (predictions.getRowCount() != targets.getRowCount() || predictions.getColumnCount() != targets.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            // Clip predictions to prevent log(0)
            predictions = Matrix.clip(predictions, 1e-15F, 1 - 1e-15);
            // Calculate categorical cross-entropy loss for each element
            return Matrix.matrixMapping(predictions, (row, column, value) -> -targets.getEntry(row, column) * Math.log(value));
        }

        @Override
        public Matrix getDerivativeMatrix(Matrix predictions, Matrix targets) {
            // Ensure input matrices have the same dimensions
            if (predictions.getRowCount() != targets.getRowCount() || predictions.getColumnCount() != targets.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            // Calculate the derivative of categorical cross-entropy loss for each element
            return Matrix.matrixMapping(predictions, (row, column, value) -> targets.getEntry(row, column) - value);
        }
    },
    ;
    // TODO: Yet to implement the sparse categorical cross entropy.
    //    SPARSE_CATEGORICAL_CROSS_ENTROPY {
    //        @Override
    //        public Matrix getLossFunctionMatrix(Matrix predictions, Matrix trueLabels) {
    //            // Check for dimension compatibility
    //            if (predictions.getRowCount() != trueLabels.getRowCount() || predictions.getColumnCount() != trueLabels.getColumnCount()) {
    //                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
    //            }
    //
    //            // Calculate sparse categorical cross-entropy loss for each element
    //            return Matrix.matrixMapping(new Matrix(predictions.getRowCount(), predictions.getColumnCount()), (row, column, value) -> {
    //                int trueClass = (int) trueLabels.getEntry(row, 0); // Assuming target is a single column matrix with class indices
    //                double predictedProb = predictions.getEntry(row, trueClass);
    //                predictedProb = Math.max(predictedProb, 1e-15); // Clip to prevent log(0)
    //                return -Math.log(predictedProb);
    //            });
    //        }
    //
    //        @Override
    //        public Matrix getDerivativeMatrix(Matrix predictions, Matrix trueLabels) {
    //            // Check for dimension compatibility
    //            if (predictions.getRowCount() != trueLabels.getRowCount() || predictions.getColumnCount() != trueLabels.getColumnCount()) {
    //                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
    //            }
    //
    //            // Calculate the derivative for each element
    //            return Matrix.matrixMapping(new Matrix(predictions.getRowCount(), predictions.getColumnCount()), (row, column, value) -> {
    //                int trueClass = (int) trueLabels.getEntry(row, 0);
    //                return column == trueClass ? value - 1 : value;
    //            });
    //        }
    //    },
}
