package com.mkproductions.jnn.cpu.entity;

import java.util.ArrayList;
import java.util.Random;

public class Matrix {
    private final ArrayList<ArrayList<Double>> data;

    public Matrix(int rowCount, int columnCount) {
        this.data = new ArrayList<>();
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
        this.data = new ArrayList<>();
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
        if (matrix1.getColumnCount() != matrix2.getRowCount()) {
            System.err.println("Matrix1: ");
            matrix1.printMatrix();
            System.err.println("Matrix2: ");
            matrix2.printMatrix();
            throw new RuntimeException(STR."Matrix multiplication not possible. Cause \{matrix1.getColumnCount()} != \{matrix2.getRowCount()}");
        }
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

    public static Matrix createFromArrayToDiagonalMatrix(ArrayList<Double> column) {
        return Matrix.matrixMapping(new Matrix(column.size(), column.size()), (a, b, value) -> {
            if (a == b) {
                return value;
            }
            return 0.0;
        });
    }

    public static Matrix createFromDiagonalToColumnMatrix(Matrix result) {
        double[] results = new double[result.getRowCount()];
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

    public ArrayList<ArrayList<Double>> getData() {
        return this.data;
    }

    public Matrix copy() {
        return new Matrix(this.getData());
    }

    public void add(Matrix matrix) {
        if (this.getRowCount() != matrix.getRowCount() || this.getColumnCount() != matrix.getColumnCount()) {
            throw new RuntimeException("Mismatch shape for performing addition operation.");
        }
        for (int a = 0; a < this.getRowCount(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) + matrix.getEntry(a, b));
    }

    public static Matrix add(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getRowCount() != matrix2.getRowCount() || matrix1.getColumnCount() != matrix2.getColumnCount()) {
            throw new RuntimeException("Mismatch shape for performing addition operation.");
        }
        Matrix result = matrix1.copy();
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix1.getEntry(a, b) + matrix2.getEntry(a, b));
        return result;
    }

    public void subtract(Matrix matrix) {
        if (this.getRowCount() != matrix.getRowCount() || this.getColumnCount() != matrix.getColumnCount()) {
            throw new RuntimeException("Mismatch shape for performing subtraction operation.");
        }
        for (int a = 0; a < this.getRowCount(); a++)
            for (int b = 0; b < this.getColumnCount(); b++)
                this.setEntry(a, b, this.getEntry(a, b) - matrix.getEntry(a, b));
    }

    public static Matrix subtract(Matrix matrix1, Matrix matrix2) {
        if (matrix1.getRowCount() != matrix2.getRowCount() || matrix1.getColumnCount() != matrix2.getColumnCount()) {
            throw new RuntimeException("Mismatch shape for performing subtraction operation.");
        }
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
        if (matrix1.getRowCount() != matrix2.getRowCount() || matrix1.getColumnCount() != matrix2.getColumnCount()) {
            throw new RuntimeException("Mismatch shape for performing Hadamard product.");
        }
        Matrix result = matrix1.copy();
        for (int a = 0; a < result.getRowCount(); a++)
            for (int b = 0; b < result.getColumnCount(); b++)
                result.setEntry(a, b, matrix1.getEntry(a, b) * matrix2.getEntry(a, b));
        return result;
    }

    public void elementWiseMultiply(Matrix matrix) {
        if (this.getRowCount() != matrix.getRowCount() || this.getColumnCount() != matrix.getColumnCount()) {
            throw new RuntimeException("Mismatch shape for performing  Hadamard product.");
        }
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
        this.data.get(row).set(column, value);
    }

    public double getEntry(int row, int column) {
        return this.data.get(row).get(column);
    }

    public int getRowCount() {
        return data.size();
    }

    public int getColumnCount() {
        return data.get(0).size();
    }

    public ArrayList<Double> getRowCount(int row) {
        return this.data.get(row);
    }

    public ArrayList<Double> getColumn(int columnIndex) {
        ArrayList<Double> column = new ArrayList<>();
        for (int a = 0; a < this.data.getFirst().size(); a++) {
            column.add(this.getEntry(a, columnIndex));
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
                if (b != this.getColumnCount() - 1) {
                    stringBuilder.append(", ");
                }
            }
            stringBuilder.append("]");
            stringBuilder.append("\n");
        }
        stringBuilder.append("Matrix rows: ").append(this.getRowCount()).append(" columns: ").append(this.getColumnCount());
        System.out.println(stringBuilder);
    }
}
