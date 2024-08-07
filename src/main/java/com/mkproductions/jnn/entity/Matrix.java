package com.mkproductions.jnn.entity;

import java.util.Arrays;
import java.util.Random;

public class Matrix {
    private final double[][] data;

    public Matrix(int row, int column) {
        this.data = new double[row][column];
        for (int a = 0; a < row; a++)
            for (int b = 0; b < column; b++)
                this.data[a][b] = 0;
    }

    public Matrix(double[][] data) {
        this.data = new double[data.length][data[0].length];
        for (int a = 0; a < data.length; a++)
            System.arraycopy(data[a], 0, this.data[a], 0, data[0].length);
    }

    public static Matrix fromArray(double[] inputs) {
        return new Matrix(new double[][]{inputs});
    }

    public static double[] toFlatArray(Matrix matrix) {
        double[] flatArray = new double[matrix.getRowCount() * matrix.getColumnCount()];
        int i = 0;
        for (int a = 0; a < matrix.getRowCount(); a++) {
            for (int b = 0; b < matrix.getColumnCount(); b++) {
                flatArray[i] = matrix.getEntry(a, b);
                i++;
            }
        }
        return flatArray;
    }

    public static Matrix matrixMultiplication(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getColumnCount() != matrix2.getRowCount())
            throw new RuntimeException("Matrix multiplication not possible.\n Cause " + matrix1.getColumnCount() + " != " + matrix2.getRowCount());
        Matrix result = new Matrix(matrix1.getRowCount(), matrix2.getColumnCount());
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++) {
                double sum = 0;
                for (int k = 0; k < matrix1.getColumnCount(); k++)
                    sum += matrix1.getEntry(a, k) * matrix2.getEntry(k, b);
                result.setEntry(a, b, sum);
            }
        return result;
    }

    public static Matrix convolute(Matrix image, Matrix filter) {
        // Calculating the output shape.
        int resultRowCount = image.getRowCount() - filter.getRowCount() + 1;
        int resultColumnCount = image.getColumnCount() - filter.getColumnCount() + 1;

        // Creating an empty result of the convolution.
        Matrix resultMatrix = new Matrix(resultRowCount, resultColumnCount);

        for (int a = 0; a < resultMatrix.getRowCount(); a++) {
            for (int b = 0; b < resultMatrix.getColumnCount(); b++) {
                double sum = 0;
                for (int c = 0; c < filter.getRowCount(); c++) {
                    for (int d = 0; d < filter.getColumnCount(); d++) {
                        sum += image.getEntry(a + c, b + d) * filter.getEntry(c, d);
                    }
                }
                resultMatrix.setEntry(a, b, sum);
            }
        }
        return resultMatrix;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(this.getColumnCount(), this.getRowCount());
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, this.getEntry(b, a));
        return result;
    }

    public void randomize() {
        Random rand = new Random();
        for (int a = 0; a < this.getRowCount(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, rand.nextDouble() * 2 - 1);
    }

    public double[][] getData() {
        return this.data;
    }

    public Matrix copy() {
        return new Matrix(this.getData());
    }

    public void add(Matrix matrix) {
        if (this.getRowCount() != matrix.getRowCount() || this.getColumnCount() != matrix.getColumnCount())
            throw new RuntimeException("Mismatch shape for performing addition operation.");
        for (int a = 0; a < this.getRowCount(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) + matrix.getEntry(a, b));
    }

    public static Matrix add(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getRowCount() != matrix2.getRowCount() || matrix1.getColumnCount() != matrix2.getColumnCount())
            throw new RuntimeException("Mismatch shape for performing addition operation.");
        Matrix result = matrix1.copy();
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix1.getEntry(a, b) + matrix2.getEntry(a, b));
        return result;
    }


    public void subtract(Matrix matrix) {
        if (this.getRowCount() != matrix.getRowCount() || this.getColumnCount() != matrix.getColumnCount())
            throw new RuntimeException("Mismatch shape for performing subtraction operation.");
        for (int a = 0; a < this.getRowCount(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) - matrix.getEntry(a, b));
    }

    public static Matrix subtract(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getRowCount() != matrix2.getRowCount() || matrix1.getColumnCount() != matrix2.getColumnCount())
            throw new RuntimeException("Mismatch shape for performing subtraction operation.");
        Matrix result = matrix1.copy();
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix1.getEntry(a, b) - matrix2.getEntry(a, b));
        return result;
    }

    public void scalarMultiply(double scale) {
        for (int a = 0; a < this.getRowCount(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) * scale);
    }

    public static Matrix scalarMultiply(Matrix matrix, double scale) {
        Matrix result = matrix.copy();
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix.getEntry(a, b) * scale);
        return result;
    }

    public static Matrix elementWiseMultiply(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getRowCount() != matrix2.getRowCount() || matrix1.getColumnCount() != matrix2.getColumnCount())
            throw new RuntimeException("Mismatch shape for performing Hadamard product.");
        Matrix result = matrix1.copy();
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix1.getEntry(a, b) * matrix2.getEntry(a, b));
        return result;
    }

    public void elementWiseMultiply(Matrix matrix) {
        if (this.getRowCount() != matrix.getRowCount() || this.getColumnCount() != matrix.getColumnCount())
            throw new RuntimeException("Mismatch shape for performing  Hadamard product.");
        for (int a = 0; a < this.getRowCount(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) * matrix.getEntry(a, b));
    }

    public void matrixMapping(MapAble function) {
        for (int a = 0; a < this.getRowCount(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, function.map(a, b, this.getEntry(a, b)));
    }

    public static Matrix matrixMapping(Matrix matrix, MapAble function) {
        Matrix result = matrix.copy();
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, function.map(a, b, matrix.getEntry(a, b)));
        return result;
    }

    public void setEntry(int row, int column, double value) {
        this.data[row][column] = value;
    }

    public double getEntry(int row, int column) {
        return this.data[row][column];
    }

    public int getRowCount() {
        return data.length;
    }

    public int getColumnCount() {
        return data[0].length;
    }

    public double[] getRowCount(int row) {
        return this.data[row];
    }

    public double[] getColumn(int columnIndex) {
        double[] column = new double[this.getRowCount()];
        for (int a = 0; a < column.length; a++) {
            column[a] = this.getEntry(a, columnIndex);
        }
        return column;
    }

    public double[] getRow(int rowIndex) {
        double[] row = new double[this.getRowCount()];
        for (int a = 0; a < row.length; a++) {
            row[a] = this.getEntry(rowIndex, a);
        }
        return row;
    }

    public void printMatrix() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("\n");
        for (int a = 0; a < this.getRowCount(); a++) {
            stringBuilder.append("[");
            for (int b = 0; b < this.getColumnCount(); b++) {
                stringBuilder.append(this.getEntry(a, b));
                if (b != this.getColumnCount() - 1)
                    stringBuilder.append(", ");
            }
            stringBuilder.append("]");
            stringBuilder.append("\n");
        }
        stringBuilder.append("Matrix rows: ").append(this.getRowCount()).append(" columns: ").append(this.getColumnCount());
        System.out.println(stringBuilder);
    }
}
