package com.mkproductions.jnn.cpu.entity;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Random;

public class Matrix {
    private static final Logger log = LoggerFactory.getLogger(Matrix.class);
    public ArrayList<ArrayList<Double>> data;

    public Matrix(int rowCount, int columnCount) {
        this(new ArrayList<>());
        for (int row = 0; row < rowCount; row++) {
            this.data.add(row, new ArrayList<>());
            for (int column = 0; column < columnCount; column++)
                this.data.get(row).add(0.0);
        }
    }

    public Matrix(ArrayList<ArrayList<Double>> data) {
        this.data = new ArrayList<>();
        for (int row = 0; row < data.size(); row++) {
            this.data.add(row, new ArrayList<>());
            for (int column = 0; column < data.getFirst().size(); column++) {
                this.data.get(row).add(column, data.get(row).get(column));
            }
        }
    }

    public Matrix(double[][] data) {
        this(new ArrayList<>());
        for (int row = 0; row < data.length; row++) {
            this.data.add(row, new ArrayList<>());
            for (int column = 0; column < data[0].length; column++) {
                this.data.get(row).add(column, data[row][column]);
            }
        }
    }

    public static Matrix fromArray(ArrayList<Double> values) {
        var inputsArrayList = new ArrayList<ArrayList<Double>>();
        inputsArrayList.add(values);
        return new Matrix(inputsArrayList);
    }

    public static Matrix fromArray(double[] values) {
        ArrayList<ArrayList<Double>> inputsArrayList = new ArrayList<>();
        inputsArrayList.add(new ArrayList<>());
        for (double value : values) {
            inputsArrayList.getFirst().add(value);
        }
        return new Matrix(inputsArrayList);
    }

    public static double[] toFlatArray(Matrix matrix) {
        double[] flatArray = new double[matrix.getRow() * matrix.getColumnCount()];
        int i = 0;
        for (int a = 0; a < matrix.getRow(); a++) {
            for (int b = 0; b < matrix.getColumnCount(); b++) {
                flatArray[i] = matrix.getEntry(a, b);
                i++;
            }
        }
        return flatArray;
    }

    public static Matrix matrixMultiplication(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getColumnCount() != matrix2.getRow()) {
            log.error("Matrix1: \n{}", matrix1);
            log.error("Matrix2: {}", matrix2);

            throw new IllegalArgumentException("Matrix multiplication not possible. Cause" + matrix1.getColumnCount() + " != " + matrix2.getRow());
        }

        Matrix result = new Matrix(matrix1.getRow(), matrix2.getColumnCount());
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumnCount(); b++) {
                double sum = 0;
                for (int k = 0; k < matrix1.getColumnCount(); k++)
                    sum += matrix1.getEntry(a, k) * matrix2.getEntry(k, b);
                result.setEntry(a, b, sum);
            }
        return result;
    }

    public static Matrix convolute(Matrix matrix, Matrix filter) {
        // Input validation
        if (filter.getRow() > matrix.getRow() || filter.getColumnCount() > matrix.getColumnCount()) {
            throw new IllegalArgumentException(
                    "Filter dimensions (" + filter.getRow() + "," + filter.getColumnCount() + " cannot exceed image dimensions(" + matrix.getRow() + "," + matrix.getColumnCount() + ")");
        }

        // Calculating the output shape.
        int resultRowCount = matrix.getRow() - filter.getRow() + 1;
        int resultColumnCount = matrix.getColumnCount() - filter.getColumnCount() + 1;

        // Creating an empty result of the convolution.
        Matrix resultMatrix = new Matrix(resultRowCount, resultColumnCount);

        for (int a = 0; a < resultMatrix.getRow(); a++) {
            for (int b = 0; b < resultMatrix.getColumnCount(); b++) {
                double sum = 0;
                for (int c = 0; c < filter.getRow(); c++) {
                    for (int d = 0; d < filter.getColumnCount(); d++) {
                        sum += matrix.getEntry(a + c, b + d) * filter.getEntry(c, d);
                    }
                }
                resultMatrix.setEntry(a, b, sum);
            }
        }
        return resultMatrix;
    }

    public static Matrix createFromArrayToDiagonalMatrix(ArrayList<Double> column) {
        return Matrix.matrixMapping(new Matrix(column.size(), column.size()), (a, b, value) -> {
            if (a == b) {
                return value;
            }
            return 0.0;
        });
    }

    public static Matrix createFromDiagonalToColumnMatrix(Matrix result) {
        double[] results = new double[result.getRow()];
        for (int a = 0; a < results.length; a++) {
            results[a] = result.getEntry(a, a);
        }
        return Matrix.fromArray(results);
    }

    public static Matrix clip(Matrix matrix, double start, double end) {
        if (start >= end) {
            throw new IllegalArgumentException("Start value must be less than end value.");
        }
        return Matrix.matrixMapping(matrix, (_, _, value) -> Math.max(Math.min(value, end), start));
    }

    public static Matrix transpose(Matrix matrix) {
        Matrix result = new Matrix(matrix.getColumnCount(), matrix.getRow());
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix.getEntry(b, a));
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(this.getColumnCount(), this.getRow());
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, this.getEntry(b, a));
        return result;
    }

    public void randomize() {
        Random rand = new Random();
        for (int a = 0; a < this.getRow(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, rand.nextDouble() * 2 - 1);
    }

    public Matrix copy() {
        return new Matrix(this.data);
    }

    public void add(Matrix matrix) {
        if (this.getRow() != matrix.getRow() || this.getColumnCount() != matrix.getColumnCount()) {
            throw new IllegalArgumentException("Mismatch shape for performing addition operation.");
        }
        for (int a = 0; a < this.getRow(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) + matrix.getEntry(a, b));
    }

    public static Matrix add(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getRow() != matrix2.getRow() || matrix1.getColumnCount() != matrix2.getColumnCount()) {
            throw new IllegalArgumentException("Mismatch shape for performing addition operation.");
        }
        Matrix result = matrix1.copy();
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix1.getEntry(a, b) + matrix2.getEntry(a, b));
        return result;
    }

    public void subtract(Matrix matrix) {
        if (this.getRow() != matrix.getRow() || this.getColumnCount() != matrix.getColumnCount()) {
            throw new IllegalArgumentException("Mismatch shape for performing subtraction operation.");
        }
        for (int a = 0; a < this.getRow(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) - matrix.getEntry(a, b));
    }

    public static Matrix subtract(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getRow() != matrix2.getRow() || matrix1.getColumnCount() != matrix2.getColumnCount()) {
            throw new IllegalArgumentException("Mismatch shape for performing subtraction operation.");
        }
        Matrix result = matrix1.copy();
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix1.getEntry(a, b) - matrix2.getEntry(a, b));
        return result;
    }

    public void scalarMultiply(double scale) {
        for (int a = 0; a < this.getRow(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) * scale);
    }

    public static Matrix scalarMultiply(Matrix matrix, double scale) {
        Matrix result = matrix.copy();
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix.getEntry(a, b) * scale);
        return result;
    }

    public static Matrix elementWiseMultiply(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getRow() != matrix2.getRow() || matrix1.getColumnCount() != matrix2.getColumnCount()) {
            log.error("Matrix1: ({}, {}", matrix1.getRow(), matrix1.getColumnCount());
            log.error("Matrix2: ({}, {}", matrix2.getRow(), matrix2.getColumnCount());
            throw new IllegalArgumentException(" Mismatch shape for performing Hadamard product.");
        }

        Matrix result = matrix1.copy();
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix1.getEntry(a, b) * matrix2.getEntry(a, b));
        return result;
    }

    public void elementWiseMultiply(Matrix matrix) {
        if (this.getRow() != matrix.getRow() || this.getColumnCount() != matrix.getColumnCount()) {
            throw new IllegalArgumentException("Mismatch shape for performing  Hadamard product.");
        }
        for (int a = 0; a < this.getRow(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) * matrix.getEntry(a, b));
    }

    public void matrixMapping(MatrixFunctionAble function) {
        for (int a = 0; a < this.getRow(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, function.map(a, b, this.getEntry(a, b)));
    }

    public static Matrix matrixMapping(Matrix matrix, MatrixFunctionAble function) {
        Matrix result = matrix.copy();
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, function.map(a, b, matrix.getEntry(a, b)));
        return result;
    }

    public void setEntry(int row, int column, double value) {
        this.data.get(row).set(column, value);
    }

    public double getEntry(int row, int column) {
        return this.data.get(row).get(column);
    }

    public int getRow() {
        return data.size();
    }

    public int getColumnCount() {
        return data.getFirst().size();
    }

    public ArrayList<Double> getColumn(int columnIndex) {
        ArrayList<Double> column = new ArrayList<>();
        for (int a = 0; a < this.data.getFirst().size(); a++) {
            column.add(this.getEntry(a, columnIndex));
        }
        return column;
    }

    public ArrayList<Double> getRow(int rowIndex) {
        return new ArrayList<>(this.data.get(rowIndex));
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("\n");
        for (int a = 0; a < this.getRow(); a++) {
            stringBuilder.append("[");
            for (int b = 0; b < this.getColumnCount(); b++) {
                stringBuilder.append(this.getEntry(a, b));
                if (b != this.getColumnCount() - 1) {
                    stringBuilder.append(", ");
                }
            }
            stringBuilder.append("]").append("\n");
        }
        stringBuilder.append("Matrix rows: ").append(this.getRow()).append(" columns: ").append(this.getColumnCount());
        return stringBuilder.toString();
    }
}