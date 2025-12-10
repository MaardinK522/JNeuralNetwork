package com.mkproductions.jnn.lossFunctions;

import com.mkproductions.jnn.cpu.entity.LossFunctionAble;
import com.mkproductions.jnn.cpu.entity.Tensor;

public enum LossFunction implements LossFunctionAble {
    MEAN_SQUARED_ERROR {
        @Override
        public Tensor getLossFunctionTensor(Tensor prediction, Tensor target) {
            // Ensure input matrices have the same dimensions
            Tensor.validateTensors(prediction, target);
            // Calculate the squared error for each element in the matrix
            return Tensor.tensorMapping(prediction, (flatIndex, _) -> Math.pow(prediction.getData().get(flatIndex) - target.getData().get(flatIndex), 2));
        }

        @Override
        public Tensor getDerivativeTensor(Tensor prediction, Tensor target) {
            Tensor.validateTensors(prediction, target);
            return Tensor.tensorMapping(prediction, ((flatIndex, _) -> -2 * (prediction.getData().get(flatIndex) - target.getData().get(flatIndex))));
        }
    }, // Mean Squared Error
    MEAN_ABSOLUTE_ERROR {
        @Override
        public Tensor getLossFunctionTensor(Tensor prediction, Tensor target) {
            Tensor.validateTensors(prediction, target);
            return Tensor.tensorMapping(prediction, (flatIndex, _) -> {
                var error = prediction.getData().get(flatIndex) - target.getData().get(flatIndex);
                return Math.abs(error);
            });
        }

        @Override
        public Tensor getDerivativeTensor(Tensor prediction, Tensor target) {
            Tensor.validateTensors(prediction, target);
            return Tensor.tensorMapping(prediction, (flatIndex, _) -> {
                double error = target.getData().get(flatIndex) - prediction.getData().get(flatIndex);
                return error >= 0 ? 1 : -1;
            });
        }
    }, // Mean Absolute Error.
    // // TODO: Implement this loss function with the proper derivative.
    //    SMOOTH_MEAN_ABSOLUTE_ERROR {
    //        @Override
    //        public Tensor getLossFunctionTensor(Tensor prediction, Tensor target) {
    //            // Ensure input matrices have the same dimensions
    //            if (prediction.getRowCount() != target.getRowCount() || prediction.getColumnCount() != target.getColumnCount()) {
    //                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
    //            }
    //
    //            return Tensor.matrixMapping(new Tensor(prediction.getRowCount(), prediction.getColumnCount()), (FlatIndex, column, value) -> {
    //                double actual = target.getEntry(FlatIndex, column);
    //                double forecast = prediction.getEntry(FlatIndex, column);
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
    //        public Tensor getDerivativeTensor(Tensor predictions, Tensor targets) {
    //            return Tensor.matrixMapping(new Tensor(predictions.getRowCount(), predictions.getColumnCount()), (FlatIndex, column, prediction) -> {
    //                double target = targets.getEntry(FlatIndex, column);
    //                double diff = target - prediction;
    //                double absDiff = Math.abs(diff);
    //                double absTarget = Math.abs(target);
    //                double absPrediction = Math.abs(prediction);
    //                double absSum = absTarget + absPrediction;
    //                return (2 * sign(diff) * (absSum - absDiff)) / (absSum * absSum);
    //            });
    //        }
    //    },
    LOG_COSH {
        @Override
        public Tensor getLossFunctionTensor(Tensor prediction, Tensor target) {
            Tensor.validateTensors(prediction, target);
            // ... (similar structure as other loss functions)
            return Tensor.tensorMapping(prediction, (flatIndex, _) -> {
                double error = -target.getData().get(flatIndex) + prediction.getData().get(flatIndex);
                return Math.log(Math.cosh(error));
            });
        }

        @Override
        public Tensor getDerivativeTensor(Tensor prediction, Tensor target) {
            Tensor.validateTensors(prediction, target);
            return Tensor.tensorMapping(prediction, (FlatIndex, _) -> {
                double error = -target.getData().get(FlatIndex) + prediction.getData().get(FlatIndex);
                return -Math.tan(error);
            });
        }
    }, // Logarithmic
    BINARY_CROSS_ENTROPY {
        @Override
        public Tensor getLossFunctionTensor(Tensor prediction, Tensor target) {
            // Check for dimension compatibility
            Tensor.validateTensors(prediction, target);
            // Calculate binary cross-entropy loss for each element
            return Tensor.tensorMapping(prediction, (flatIndex, _) -> {
                double y = target.getData().get(flatIndex);
                double p = prediction.getData().get(flatIndex);
                // Clip probabilities to prevent log(0)
                p = Math.max(p, 1e-15);
                p = Math.min(p, 1 - 1e-15);
                return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
            });
        }

        @Override
        public Tensor getDerivativeTensor(Tensor prediction, Tensor target) {
            // Ensure input matrices have the same dimensions
            Tensor.validateTensors(prediction, target);

            // Calculate the derivative of binary cross-entropy loss for each element
            return Tensor.tensorMapping(prediction, (flatIndex, _) -> (target.getData().get(flatIndex) - prediction.getData().get(flatIndex)));
        }
    }, // Binary Cross Entropy
    CATEGORICAL_CROSS_ENTROPY {
        @Override
        public Tensor getLossFunctionTensor(Tensor predictions, Tensor targets) {
            // Check for dimension compatibility
            Tensor.validateTensors(predictions, targets);
            // Clip predictions to prevent log(0)
            predictions = Tensor.clip(predictions, 1e-15, 1 - 1e-15);
            // Calculate categorical cross-entropy loss for each element
            return Tensor.tensorMapping(predictions, (flatIndex, value) -> targets.getData().get(flatIndex) * Math.log(value));
        }

        public Tensor getDerivativeTensor(Tensor predictions, Tensor targets) {
            // Ensure input matrices have the same dimensions
            Tensor.validateTensors(predictions, targets);
            // Calculate the derivative of categorical cross-entropy loss for each element
            return Tensor.tensorMapping(predictions, (a, value) -> {
                //                double y = targets.getEntry(a, b);
                //                double p = Math.max(Math.min(value, 1 - 1e-15), 1e-15);
                //                return (y / p - (1 - y) / (1 - p));
                targets.getData().set(a, targets.getData().get(a) - value);
                return targets.getData().get(a);
            });
        }
    }, // Categorical Cross Entropy
    SPARSE_CATEGORICAL_CROSS_ENTROPY {
        @Override
        public Tensor getLossFunctionTensor(Tensor predictions, Tensor trueLabels) {
            // Check for dimension compatibility
            if (Tensor.hasSameShape(predictions, trueLabels)) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            // Calculate sparse categorical cross-entropy loss for each element
            return Tensor.tensorMapping(new Tensor(predictions.getShape().toHeapArray()), (flatIndex, _) -> {
                double predictedProb = predictions.getEntry(flatIndex);
                predictedProb = Math.max(predictedProb, 1e-15); // Clip to prevent log(0)
                return -Math.log(predictedProb);
            });
        }

        @Override
        public Tensor getDerivativeTensor(Tensor predictions, Tensor trueLabels) {
            // Check for dimension compatibility
            for (int shapeIndex = 0; shapeIndex < predictions.getShape().getSize(); shapeIndex++) {
                if (predictions.getShape().toHeapArray()[shapeIndex] != trueLabels.getShape().toHeapArray()[shapeIndex]) {
                    throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
                }
            }
            // Calculate the derivative for each element
            if (Tensor.hasSameShape(predictions, trueLabels)) {
                throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
            }
            // Calculate sparse categorical cross-entropy loss for each element
            return Tensor.tensorMapping(new Tensor(predictions.getShape().toHeapArray()), (flatIndex, _) -> predictions.getEntry(flatIndex) - trueLabels.getEntry(flatIndex));
        }
    }, // Sparse Categorical Cross Entropy
}