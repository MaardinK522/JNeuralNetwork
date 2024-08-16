package com.mkproductions.jnn.entity.lossFunctions;

import com.mkproductions.jnn.entity.LossFunctionAble;
import com.mkproductions.jnn.entity.Matrix;

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
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> Math.abs(target.getEntry(row, column) - prediction.getEntry(row, column)));
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
                var error = target.getEntry(row, column) - prediction.getEntry(row, column);
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
                var error = target.getEntry(row, column) - prediction.getEntry(row, column);
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
                double absDiff = Math.abs(forecast - actual);
                double absSum = Math.abs(actual) + Math.abs(forecast);
                return absSum == 0 ? 0 : (2 * absDiff) / absSum; // Avoid division by zero
            });
        }
    }, LOG_COSH_LOSS {
        @Override
        public Matrix getLossFunctionMatrix(Matrix prediction, Matrix target) {
            // ... (similar structure as other loss functions)
            return Matrix.matrixMapping(new Matrix(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
                return Math.log(Math.cosh(error));
            });
        }
    }, //    HUBER_LOSS {
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
    },

    CATEGORICAL_CROSS_ENTROPY {
        @Override
        public Matrix getLossFunctionMatrix(Matrix predictions, Matrix targets) {
            // Check for dimension compatibility
            if (predictions.getRowCount() != targets.getRowCount() || predictions.getColumnCount() != targets.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Clip predictions to prevent log(0)
            predictions = Matrix.clip(predictions, 1e-15, 1 - 1e-15);

            // Calculate categorical cross-entropy loss for each element
            return Matrix.matrixMapping(predictions, (row, column, value) -> -targets.getEntry(row, column) * Math.log(value));
        }
    },

    SPARSE_CATEGORICAL_CROSS_ENTROPY {
        @Override
        public Matrix getLossFunctionMatrix(Matrix predictions, Matrix trueLabels) {
            // Check for dimension compatibility
            if (predictions.getRowCount() != trueLabels.getRowCount() || predictions.getColumnCount() != trueLabels.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate sparse categorical cross-entropy loss for each element
            return Matrix.matrixMapping(new Matrix(predictions.getRowCount(), predictions.getColumnCount()), (row, column, value) -> {
                int trueClass = (int) trueLabels.getEntry(row, 0); // Assuming target is a single column matrix with class indices
                double predictedProb = predictions.getEntry(row, trueClass);
                predictedProb = Math.max(predictedProb, 1e-15); // Clip to prevent log(0)
                return -Math.log(predictedProb);
            });
        }
    },

    HINGE_LOSS {
        @Override
        public Matrix getLossFunctionMatrix(Matrix predictions, Matrix trueLabels) {
            // Check for dimension compatibility
            if (predictions.getRowCount() != trueLabels.getRowCount() || predictions.getColumnCount() != trueLabels.getColumnCount()) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }

            // Calculate hinge loss for each element
            return Matrix.matrixMapping(new Matrix(predictions.getRowCount(), predictions.getColumnCount()), (row, column, value) -> {
                double correctClassScore = predictions.getEntry(row, (int) trueLabels.getEntry(row, 0));
                double incorrectClassScore = predictions.getEntry(row, column);
                return Math.max(0, 1 - (correctClassScore - incorrectClassScore));
            });
        }
    },
}
