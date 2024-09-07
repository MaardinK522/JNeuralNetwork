package com.mkproductions.jnn.entity.solvers;

import com.mkproductions.jnn.entity.activationFunctions.ActivationFunctionManager;
import com.mkproductions.jnn.entity.lossFunctions.LossFunctionManager;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DDouble;

public class TaskGraphMatrixSolver {
    public static void solveAddition(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix1, Matrix2DDouble matrix2) {
        TaskGraphMatrixSolver.solveAddition(taskGraph, taskID, matrix1, matrix2, matrix1);
    }

    public static void solveAddition(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix1, Matrix2DDouble matrix2, Matrix2DDouble result) {
        if (matrix1.getNumRows() != matrix2.getNumRows() || matrix1.getNumColumns() != matrix2.getNumColumns() || matrix1.getNumRows() != result.getNumRows() || matrix1.getNumColumns() != result.getNumColumns())
            throw new RuntimeException("Matrices does not have the same number of rows and columns for performing addition.");
        taskGraph.task(taskID, MatrixOperationSolver::solveAddition, matrix1, matrix2, result);
    }

    public static void solveSubtraction(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix1, Matrix2DDouble matrix2) {
        TaskGraphMatrixSolver.solveSubtraction(taskGraph, taskID, matrix1, matrix2, matrix1);
    }

    public static void solveSubtraction(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix1, Matrix2DDouble matrix2, Matrix2DDouble result) {
        if (matrix1.getNumRows() != matrix2.getNumRows() || matrix1.getNumColumns() != matrix2.getNumColumns() || matrix1.getNumRows() != result.getNumRows() || matrix1.getNumColumns() != result.getNumColumns())
            throw new RuntimeException("Matrices does not have the same number of rows and columns for performing subtraction.");
        taskGraph.task(taskID, MatrixOperationSolver::solveSubtraction, matrix1, matrix2, result);
    }

    public static void solveMatrixScaling(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix, double scale) {
        TaskGraphMatrixSolver.solveMatrixScaling(taskGraph, taskID, matrix, scale, matrix);
    }

    public static void solveMatrixScaling(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix1, double scale, Matrix2DDouble result) {
        taskGraph.task(taskID, MatrixOperationSolver::solveMatrixScaling, matrix1, scale, result);
    }

    public static void solveElementWiseMultiplication(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix1, Matrix2DDouble matrix2) {
        TaskGraphMatrixSolver.solveElementWiseMultiplication(taskGraph, taskID, matrix1, matrix2, matrix1);
    }

    public static void solveElementWiseMultiplication(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix1, Matrix2DDouble matrix2, Matrix2DDouble result) {
        if (matrix1.getNumRows() != matrix2.getNumRows() || matrix1.getNumColumns() != matrix2.getNumColumns() || matrix1.getNumRows() != result.getNumRows() || matrix1.getNumColumns() != result.getNumColumns())
            throw new RuntimeException("Matrices does not have the same number of rows and columns for performing element wise multiplication.");
        taskGraph.task(taskID, MatrixOperationSolver::solveElementWiseMultiplication, matrix1, matrix2, result);
    }

    public static void solveMatrixMultiplication(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix1, Matrix2DDouble matrix2, Matrix2DDouble result) {
        if (matrix1.getNumColumns() != matrix2.getNumRows() || result.getNumRows() != matrix1.getNumRows() || result.getNumColumns() != matrix2.getNumColumns()) {
            System.out.println(matrix1);
            System.out.println(matrix2);
            throw new RuntimeException(STR."Given matrices does validates the matrix multiplication criteria.\{matrix1.getNumColumns()} != \{matrix2.getNumRows()}. Given result \{result}. output result\{matrix1.getNumRows()} * \{matrix2.getNumColumns()}");
        }
        taskGraph.task(taskID, MatrixOperationSolver::solveMatrixMultiplication, matrix1, matrix2, result);
    }

    public static void fill(TaskGraph taskGraph, String taskID, double value, Matrix2DDouble matrix) {
        taskGraph.task(taskID, MatrixOperationSolver::fill, value, matrix);
    }

    public static void initializeMatrixWithRandomNumbers(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix, long nanoTime) {
        taskGraph.task(taskID, MatrixOperationSolver::initializeMatrixWithRandomNumbers, matrix, nanoTime);
    }

    public static void transpose(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix, Matrix2DDouble result) {
        if (matrix.getNumRows() != result.getNumColumns() || matrix.getNumColumns() != result.getNumRows())
            throw new RuntimeException("Given result matrix does not have proper dimension.");
        taskGraph.task(taskID, MatrixOperationSolver::transpose, matrix, result);
    }

    public static void applyActivationFunction(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix, ActivationFunctionManager.NetworkActivationFunction activationFunction) {
        taskGraph.task(taskID, ActivationFunctionManager::applyActivationFunction, matrix, activationFunction);
    }

    public static void applyActivationFunctionDerivative(TaskGraph taskGraph, String taskID, Matrix2DDouble matrix, ActivationFunctionManager.NetworkActivationFunction activationFunction) {
        taskGraph.task(taskID, ActivationFunctionManager::applyActivationFunctionDerivative, matrix, activationFunction);
    }

    public static void calculateLossDerivative(TaskGraph taskGraph, String taskID, Matrix2DDouble prediction, Matrix2DDouble target, Matrix2DDouble result, LossFunctionManager.LossFunction lossFunction) {
        if (prediction.getNumRows() != target.getNumRows() || prediction.getNumColumns() != target.getNumColumns())
            throw new RuntimeException("Unable to calculate the loss function derivative of the given matrices.");
        taskGraph.task(taskID, LossFunctionManager::applyLossFunctionDerivative, prediction, target, result, lossFunction);
    }

    public static class MatrixOperationSolver {
        public static void solveAddition(Matrix2DDouble matrix1, Matrix2DDouble matrix2, Matrix2DDouble result) {
            for (@Parallel int i = 0; i < result.getNumRows(); i++)
                for (@Parallel int j = 0; j < result.getNumColumns(); j++)
                    result.set(i, j, matrix1.get(i, j) + matrix2.get(i, j));
        }

        public static void solveSubtraction(Matrix2DDouble matrix1, Matrix2DDouble matrix2, Matrix2DDouble result) {
//            throw new RuntimeException("Matrices do not have the same number of rows and columns");
            for (@Parallel int i = 0; i < result.getNumRows(); i++)
                for (@Parallel int j = 0; j < result.getNumColumns(); j++)
                    result.set(i, j, matrix1.get(i, j) - matrix2.get(i, j));
        }

        public static void solveMatrixScaling(Matrix2DDouble matrix1, double scale, Matrix2DDouble result) {
            for (@Parallel int i = 0; i < matrix1.getNumRows(); i++)
                for (@Parallel int j = 0; j < matrix1.getNumColumns(); j++)
                    result.set(i, j, scale * matrix1.get(i, j));
        }

        public static void solveElementWiseMultiplication(Matrix2DDouble matrix1, Matrix2DDouble matrix2, Matrix2DDouble result) {
            for (@Parallel int i = 0; i < result.getNumRows(); i++)
                for (@Parallel int j = 0; j < result.getNumColumns(); j++) {
                    double value = matrix1.get(i, j) * matrix2.get(i, j);
                    result.set(i, j, value);
                }
        }

        public static void solveMatrixMultiplication(Matrix2DDouble matrix1, Matrix2DDouble matrix2, Matrix2DDouble result) {

            for (@Parallel int i = 0; i < result.getNumRows(); i++) {
                for (@Parallel int j = 0; j < result.getNumColumns(); j++) {
                    double sum = 0.0f;
                    for (int k = 0; k < matrix2.getNumRows(); k++)
                        sum += matrix1.get(i, k) * matrix2.get(k, j);
                    result.set(i, j, sum);
                }
            }
        }

        public static void fill(double value, Matrix2DDouble matrix) {
            for (@Parallel int i = 0; i < matrix.getNumRows(); i++)
                for (@Parallel int j = 0; j < matrix.getNumColumns(); j++)
                    matrix.set(i, j, value);
        }

        public static void initializeMatrixWithRandomNumbers(Matrix2DDouble matrix, long nanoTime) {
            for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
                for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                    long seed = (long) (i + 1) * (j + 1) + nanoTime;
                    // generate a pseudo random number (you do need it twice)
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                    // this generates a number between 0 and 1 (with an awful entropy)
                    double value = (seed & 0x0FFFFFFF) / 268435455f;
                    matrix.set(i, j, value * 2 - 1);
                }
            }
        }

        public static void transpose(Matrix2DDouble matrix, Matrix2DDouble result) {
            for (@Parallel int i = 0; i < matrix.getNumRows(); i++) {
                for (@Parallel int j = 0; j < matrix.getNumColumns(); j++) {
                    result.set(i, j, matrix.get(j, i));
                }
            }
        }
    }
}

