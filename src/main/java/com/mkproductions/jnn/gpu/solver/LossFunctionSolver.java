package com.mkproductions.jnn.gpu.solver;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

public class LossFunctionSolver {
    //    public static void applyLossFunctionDerivative(Matrix2DDouble prediction, Matrix2DDouble target, Matrix2DDouble result, LossFunction lossFunction) {
    //        for (@Parallel int i = 0; i < target.getNumRows(); i++) {
    //            for (@Parallel int j = 0; j < target.getNumColumns(); j++) {
    //                double activatedValue = switch (lossFunction) {
    //                    case MEAN_ABSOLUTE_ERROR -> target.get(i, j) - prediction.get(i, j) >= 0 ? 1 : -1;
    //                    case MEAN_SQUARED_ERROR -> -2 * (prediction.get(i, j) - target.get(i, j));
    //                    case LOG_COSH -> TornadoMath.tanh(target.get(i, j) - prediction.get(i, j));
    //                    case BINARY_CROSS_ENTROPY ->
    //                            (-target.get(i, j) / prediction.get(i, j)) + ((1 - target.get(i, j)) / (1 - prediction.get(i, j)));
    //                    case CATEGORICAL_CROSS_ENTROPY -> -target.get(i, j) / prediction.get(i, j);
    //                };
    //                result.set(i, j, activatedValue);
    //            }
    //        }
    //    }
    //                                double activatedValue = switch (lossFunction) {
    //                                    case MEAN_ABSOLUTE_ERROR -> target.get(i, j) - prediction.get(i, j) >= 0 ? 1 : -1;
    //                                    case MEAN_SQUARED_ERROR -> -2 * (prediction.get(i, j) - target.get(i, j));
    //                                    case LOG_COSH -> TornadoMath.tanh(target.get(i, j) - prediction.get(i, j));
    //                                    case BINARY_CROSS_ENTROPY ->
    //                                            (-target.get(i, j) / prediction.get(i, j)) + ((1 - target.get(i, j)) / (1 - prediction.get(i, j)));
    //                                    case CATEGORICAL_CROSS_ENTROPY -> -target.get(i, j) / prediction.get(i, j);
    //                                };
    public static void applyLossFunctionDerivativeMAE(Matrix2DDouble prediction, Matrix2DDouble target, Matrix2DDouble result) {
        for (@Parallel int i = 0; i < target.getNumRows(); i++) {
            for (@Parallel int j = 0; j < target.getNumColumns(); j++) {
                result.set(i, j, target.get(i, j) - prediction.get(i, j) >= 0 ? 1 : -1);
            }
        }
    }

    public static void applyLossFunctionDerivativeCCE(Matrix2DDouble prediction, Matrix2DDouble target, Matrix2DDouble result) {
        for (@Parallel int i = 0; i < target.getNumRows(); i++) {
            for (@Parallel int j = 0; j < target.getNumColumns(); j++) {
                result.set(i, j, -target.get(i, j) / prediction.get(i, j));
            }
        }
    }

    public static void applyLossFunctionDerivativeMSE(Matrix2DDouble prediction, Matrix2DDouble target, Matrix2DDouble result) {
        for (@Parallel int i = 0; i < target.getNumRows(); i++) {
            for (@Parallel int j = 0; j < target.getNumColumns(); j++) {
                result.set(i, j, -2 * (prediction.get(i, j) - target.get(i, j)));
            }
        }
    }

    public static void applyLossFunctionDerivativeLogCosH(Matrix2DDouble prediction, Matrix2DDouble target, Matrix2DDouble result) {
        for (@Parallel int i = 0; i < target.getNumRows(); i++) {
            for (@Parallel int j = 0; j < target.getNumColumns(); j++) {
                result.set(i, j, TornadoMath.tanh(target.get(i, j) - prediction.get(i, j)));
            }
        }
    }

    public static void applyLossFunctionDerivativeBCE(Matrix2DDouble prediction, Matrix2DDouble target, Matrix2DDouble result) {
        for (@Parallel int i = 0; i < target.getNumRows(); i++) {
            for (@Parallel int j = 0; j < target.getNumColumns(); j++) {
                result.set(i, j, (-target.get(i, j) / prediction.get(i, j)) + ((1 - target.get(i, j)) / (1 - prediction.get(i, j))));
            }
        }
    }
}
