package com.mkproductions.jnn.entity;

public class Matrix {
    private final int row;
    private final int column;
    private final double[][] data;

    public Matrix(int row, int column) {
        this.row = row;
        this.column = column;
        this.data = new double[row][column];
        for (int a = 0; a < this.row; a++)
            for (int b = 0; b < this.column; b++)
                this.data[a][b] = 0;
    }

    public Matrix(double[][] data) {
        this.row = data.length;
        this.column = data[0].length;
        this.data = new double[this.row][this.column];
        for (int a = 0; a < this.row; a++)
            System.arraycopy(data[a], 0, this.data[a], 0, this.column);
    }

    public static Matrix fromArray(double[] inputs) {
        return new Matrix(new double[][]{inputs}).transpose();
    }

    public Matrix multiply(Matrix matrix) {
        if (this.getColumn() != matrix.getRow())
            throw new RuntimeException("Matrix multiplication not possible.\n Cause " + this.getColumn() + " != " + matrix.getRow());
        Matrix result = new Matrix(this.getRow(), matrix.getColumn());
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumn(); b++) {
                double sum = 0;
                for (int k = 0; k < this.getColumn(); k++) sum += this.getEntry(a, k) * matrix.getEntry(k, b);
                result.setEntry(a, b, sum);
            }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(this.getColumn(), this.getRow());
        for (int a = 0; a < result.getRow(); a++)
            for (int b = 0; b < result.getColumn(); b++)
                result.setEntry(a, b, this.getEntry(b, a));
        return result;
    }

    public void randomize() {
        for (int a = 0; a < this.getRow(); a++)
            for (int b = 0; b < this.getColumn(); b++)
                this.setEntry(a, b, Math.random() * 2 - 1);
    }

    public double[][] getData() {
        return this.data;
    }

    public Matrix copy() {
        return new Matrix(this.getData());
    }

    public Matrix add(Matrix matrix) {
        return matrixMapping(matrix, (int r, int c, double value) -> value + matrix.getEntry(r, c));
    }

    public Matrix subtract(Matrix matrix) {
        return matrixMapping(matrix, (int r, int c, double value) -> value - matrix.getEntry(r, c));
    }

    public Matrix scalarMultiply(double scalar) {
        return matrixMapping(this, (int r, int c, double value) -> value * scalar);
    }

    public static Matrix elementWiseMultiply(Matrix matrix1, Matrix matrix2) {
        return matrix1.matrixMapping(matrix2, (int r, int c, double value) -> matrix1.getEntry(r, c) * value);
    }

    public static Matrix matrixMapping(Matrix matrix, MapAble function) {
        Matrix result = new Matrix(matrix.getRow(), matrix.getColumn());
        for (int a = 0; a < matrix.getRow(); a++) {
            for (int b = 0; b < matrix.getColumn(); b++) {
                double value = function.map(a, b, matrix.getEntry(a, b));
                result.setEntry(a, b, value);
            }
        }
        return result;
    }

    public void setEntry(int row, int column, double value) {
        this.data[row][column] = value;
    }

    public double getEntry(int row, int column) {
        return this.data[row][column];
    }

    public int getRow() {
        return row;
    }

    public int getColumn() {
        return column;
    }

    public double[] getRow(int row) {
        return this.data[row];
    }

    public double[] getColumn(int columnIndex) {
        double[] column = new double[this.getRow()];
        for (int a = 0; a < column.length; a++) {
            column[a] = this.getEntry(a, columnIndex);
        }
        return column;
    }

    public void printMatrix() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Matrix rows: ").append(this.getRow()).append(" columns: ").append(this.getColumn()).append("\n");
        for (int a = 0; a < this.row; a++) {
            for (int b = 0; b < this.column; b++) {
                stringBuilder.append(this.getEntry(a, b)).append("  ");
            }
            stringBuilder.append("\n");
        }
        System.out.println(stringBuilder);
    }
}
