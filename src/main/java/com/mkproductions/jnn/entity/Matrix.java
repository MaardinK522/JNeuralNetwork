package com.mkproductions.jnn.entity;

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
                this.setEntry(a, b, rand.nextGaussian());
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

    public double[] getColumnCount(int columnIndex) {
        double[] column = new double[this.getRowCount()];
        for (int a = 0; a < column.length; a++) {
            column[a] = this.getEntry(a, columnIndex);
        }
        return column;
    }

    public void printMatrix() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Matrix rows: ").append(this.getRowCount()).append(" columns: ").append(this.getColumnCount()).append("\n");
        for (int a = 0; a < this.getRowCount(); a++) {
            for (int b = 0; b < this.getColumnCount(); b++) {
                stringBuilder.append(this.getEntry(a, b)).append("  ");
            }
            stringBuilder.append("\n");
        }
        System.out.println(stringBuilder);
    }
}
