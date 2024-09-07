package com.mkproductions.jnn.entity.lossFunctions;

import com.mkproductions.jnn.entity.LossFunctionAble;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

public enum LossFunction implements LossFunctionAble {
    /**
     * Absolute error loss function.
     */
    ABSOLUTE_ERROR {
        @Override
        public Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
//            if (prediction.getNumRows() != target.getNumRows() || prediction.getNumColumns() != target.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//            // Calculate the squared error for each element in the matrix.
//            return Matrix2DDouble.matrixMapping(prediction, (row, column, value) -> (target.getEntry(row, column) - value));
            return null;
        }

        @Override
        public Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
//                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
//                return error >= 0 ? 1 : -1;
//            });
            return null;
        }
    },
    /**
     * Squared error loss function.
     */
    MEAN_SQUARED_ERROR {
        @Override
        public Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
//            // Ensure input matrices have the same dimensions
//            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//
//            // Calculate the squared error for each element in the matrix
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> Math.pow(prediction.getEntry(row, column) - target.getEntry(row, column), 2));
            return null;
        }

        @Override
        public Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getData(), prediction.getRowCount(), prediction.getColumnCount()), ((row, column, value) -> -2 * (prediction.getEntry(row, column) - target.getEntry(row, column))));
            return null;
        }
    },
    /**
     * Mean absolute error loss function.
     */
    MEAN_ABSOLUTE_ERROR {
        @Override
        public Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
            // Ensure input matrices have the same dimensions
//            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//
//            // Calculate the absolute error for each element in the matrix
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
//                var error = prediction.getEntry(row, column) - target.getEntry(row, column);
//                return Math.abs(error);
//            });
            return null;
        }

        @Override
        public Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
//                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
//                return error >= 0 ? 1 : -1;
//            });
            return null;
        }
    },
    /**
     * Smooth mean absolute error loss function.
     */
    // TODO: Implement this loss function with the proper derivative.
//    SMOOTH_MEAN_ABSOLUTE_ERROR {
//        @Override
//        public Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
//            // Ensure input matrices have the same dimensions
//            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
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
//        public Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble predictions, Matrix2DDouble targets) {
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(predictions.getRowCount(), predictions.getColumnCount()), (row, column, prediction) -> {
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
        public Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
            // ... (similar structure as other loss functions)
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
//                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
//                return Math.log(Math.cosh(error));
//            });
            return null;
        }

        @Override
        public Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
//                double error = target.getEntry(row, column) - prediction.getEntry(row, column);
//                return Math.tanh(error);
//            });
            return null;
        }
    },
    // Binary cross entropy.
    BINARY_CROSS_ENTROPY {
        @Override
        public Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
            // Check for dimension compatibility
//            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//            // Calculate binary cross-entropy loss for each element
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> {
//                double y = target.getEntry(row, column);
//                double p = prediction.getEntry(row, column);
//                // Clip probabilities to prevent log(0)
//                p = Math.max(p, 1e-15);
//                p = Math.min(p, 1 - 1e-15);
//                return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
//            });
            return null;
        }

        @Override
        public Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble prediction, Matrix2DDouble target) {
//            // Ensure input matrices have the same dimensions
//            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//
//            // Calculate the derivative of binary cross-entropy loss for each element
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(prediction.getRowCount(), prediction.getColumnCount()), (row, column, value) -> (target.getEntry(row, column) - prediction.getEntry(row, column)));
            return null;
        }
    },
    // Categorical cross entropy.
    CATEGORICAL_CROSS_ENTROPY {
        @Override
        public Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble predictions, Matrix2DDouble targets) {
//            // Check for dimension compatibility
//            if (predictions.getRowCount() != targets.getRowCount() || predictions.getColumnCount() != targets.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//
//            // Clip predictions to prevent log(0)
//            predictions = Matrix2DDouble.clip(predictions, 1e-15F, 1 - 1e-15);
//
//            // Calculate categorical cross-entropy loss for each element
//            return Matrix2DDouble.matrixMapping(predictions, (row, column, value) -> -targets.getEntry(row, column) * Math.log(value));
            return null;
        }

        @Override
        public Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble predictions, Matrix2DDouble targets) {
//            // Ensure input matrices have the same dimensions
//            if (predictions.getRowCount() != targets.getRowCount() || predictions.getColumnCount() != targets.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//
//            // Calculate the derivative of categorical cross-entropy loss for each element
//            return Matrix2DDouble.matrixMapping(predictions, (row, column, value) -> targets.getEntry(row, column) - value);
            return null;
        }
    },
    // TODO: Yet to implement the sparse categorical cross entropy.
//    SPARSE_CATEGORICAL_CROSS_ENTROPY {
//        @Override
//        public Matrix2DDouble getLossFunctionMatrix2DDouble(Matrix2DDouble predictions, Matrix2DDouble trueLabels) {
//            // Check for dimension compatibility
//            if (predictions.getRowCount() != trueLabels.getRowCount() || predictions.getColumnCount() != trueLabels.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//
//            // Calculate sparse categorical cross-entropy loss for each element
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(predictions.getRowCount(), predictions.getColumnCount()), (row, column, value) -> {
//                int trueClass = (int) trueLabels.getEntry(row, 0); // Assuming target is a single column matrix with class indices
//                double predictedProb = predictions.getEntry(row, trueClass);
//                predictedProb = Math.max(predictedProb, 1e-15); // Clip to prevent log(0)
//                return -Math.log(predictedProb);
//            });
//        }
//
//        @Override
//        public Matrix2DDouble getDerivativeMatrix2DDouble(Matrix2DDouble predictions, Matrix2DDouble trueLabels) {
//            // Check for dimension compatibility
//            if (predictions.getRowCount() != trueLabels.getRowCount() || predictions.getColumnCount() != trueLabels.getColumnCount()) {
//                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
//            }
//
//            // Calculate the derivative for each element
//            return Matrix2DDouble.matrixMapping(new Matrix2DDouble(predictions.getRowCount(), predictions.getColumnCount()), (row, column, value) -> {
//                int trueClass = (int) trueLabels.getEntry(row, 0);
//                return column == trueClass ? value - 1 : value;
//            });
//        }
//    },
}
