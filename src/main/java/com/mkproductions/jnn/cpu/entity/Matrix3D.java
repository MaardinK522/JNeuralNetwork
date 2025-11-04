package com.mkproductions.jnn.cpu.entity;

import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;

public class Matrix3D {
    private static final Logger log = LoggerFactory.getLogger(Matrix3D.class);
    ArrayList<ArrayList<ArrayList<Double>>> data;

    public int getDepthCount() {
        return data.size();
    }

    public int getRowCount() {
        return data.getFirst().size();
    }

    public int getColumnCount() {
        return data.getFirst().getFirst().size();
    }

    public Matrix3D(int row, int column, int depth) {
        this.data = new ArrayList<>();
        for (int a = 0; a < depth; a++) {
            this.data.add(new ArrayList<>());
            for (int b = 0; b < row; b++) {
                this.data.get(a).add(new ArrayList<>());
                for (int c = 0; c < column; c++) {
                    this.data.get(a).get(b).add(0.0);
                }
            }
        }
    }

    public Matrix3D(double[][][] data, int row, int column, int depth) {
        this.data = new ArrayList<>();
        for (int c = 0; c < depth; c++) {
            this.data.add(new ArrayList<>());
            for (int a = 0; a < row; a++) {
                this.data.get(c).add(new ArrayList<>());
                for (int b = 0; b < column; b++) {
                    this.data.get(c).get(a).add(data[c][a][b]);
                }
            }
        }
        //        this(this.data, row, column, depth);
    }

    public ArrayList<ArrayList<ArrayList<Double>>> getData() {
        return data;
    }

    public void setData(ArrayList<ArrayList<ArrayList<Double>>> data) {
        this.data = data;
    }

    public double getEntry(int row, int column, int depth) {
        return this.data.get(depth).get(row).get(column);
    }

    public void setEntry(int row, int column, int depth, double value) {
        this.data.get(depth).get(row).set(column, value);
    }

    public static double @NotNull [] toFlatArray(@NotNull Matrix3D matrix3D) {
        var data = matrix3D.getData();
        double[] flatArray = new double[data.size() * data.getFirst().size() * data.getFirst().getFirst().size()];
        for (int c = 0; c < matrix3D.getDepthCount(); c++) {
            for (int a = 0; a < matrix3D.getRowCount(); a++) {
                for (int b = 0; b < matrix3D.getColumnCount(); b++) {
                    flatArray[a * matrix3D.getColumnCount() * matrix3D.getDepthCount() + b * matrix3D.getDepthCount() + c] = matrix3D.getEntry(a, b, c);
                }
            }
        }
        return flatArray;
    }

    public static @NotNull Matrix3D fromFlatArray(double[] data, int row, int column, int depth) {
        var tensor = new Matrix3D(row, column, depth);
        for (int a = 0; a < row; a++) {
            for (int b = 0; b < column; b++) {
                for (int c = 0; c < depth; c++) {
                    tensor.setEntry(a, b, c, data[a * column * depth + b * depth + c]);
                }
            }
        }
        return tensor;
    }

    public void add(Matrix3D matrix3D) {
        mapToMatrix3D(this, (a, b, c, value) -> this.getEntry(a, b, c) + matrix3D.getEntry(a, b, c));
    }

    public static @NotNull Matrix3D add(Matrix3D matrix3D1, Matrix3D matrix3D2) {
        return Matrix3DMapping(matrix3D1, (a, b, c, value) -> value + matrix3D2.getEntry(a, b, c));
    }

    public void scalarAdd(double scale) {
        mapToMatrix3D(this, (a, b, c, value) -> this.getEntry(a, b, c) + scale);
    }

    public static @NotNull Matrix3D scalarAdd(Matrix3D matrix3D, double scale) {
        return Matrix3DMapping(matrix3D, (a, b, c, value) -> value + scale);
    }

    public void subtract(Matrix3D matrix3D) {
        mapToMatrix3D(this, (a, b, c, value) -> this.getEntry(a, b, c) - matrix3D.getEntry(a, b, c));
    }

    public static @NotNull Matrix3D subtract(Matrix3D matrix3D1, Matrix3D matrix3D2) {
        return Matrix3DMapping(matrix3D1, (a, b, c, value) -> value - matrix3D2.getEntry(a, b, c));
    }

    public void scalarSubtract(double scale) {
        mapToMatrix3D(this, (a, b, c, value) -> this.getEntry(a, b, c) - scale);
    }

    public static @NotNull Matrix3D scalarSubtract(Matrix3D matrix3D, double scale) {
        return Matrix3DMapping(matrix3D, (a, b, c, value) -> value - scale);
    }

    public Matrix3D scalarMultiply(double scale) {
        return Matrix3DMapping(this, (a, b, c, value) -> value * scale);
    }

    public static @NotNull Matrix3D scalarMultiply(Matrix3D matrix3D, double scale) {
        return Matrix3DMapping(matrix3D, (a, b, c, value) -> value * scale);
    }

    public void elementWiseMultiply(Matrix3D matrix) {
        mapToMatrix3D(this, (a, b, c, value) -> this.getEntry(a, b, c) * matrix.getEntry(a, b, c));
    }

    public static @NotNull Matrix3D elementWiseMultiply(@NotNull Matrix3D matrix1, @NotNull Matrix3D matrix2) {
        if (matrix1.getRowCount() != matrix2.getRowCount() || matrix1.getColumnCount() != matrix2.getColumnCount() || matrix1.getDepthCount() != matrix2.getDepthCount()) {
            log.error("Matrix1: ({},{},{})", matrix1.getRowCount(), matrix1.getColumnCount(), matrix1.getDepthCount());
            log.error("Matrix2: ({},{},{})", matrix2.getRowCount(), matrix2.getColumnCount(), matrix2.getDepthCount());
            throw new IllegalArgumentException("Mismatch shape for performing Hadamard product.");
        }
        return Matrix3DMapping(matrix1, (a, b, c, value) -> value * matrix2.getEntry(a, b, c));
    }

    public static Matrix3D convolve(Matrix3D input, Matrix3D[] filters, int stride, int padding) {
        // Apply padding
        Matrix3D paddedInput = addPadding(input, padding);

        int outputRows = (paddedInput.getRowCount() - filters[0].getRowCount()) / stride + 1;
        int outputColumns = (paddedInput.getColumnCount() - filters[0].getColumnCount()) / stride + 1;
        int outputDepth = filters.length; // One output channel per filter

        Matrix3D result = new Matrix3D(outputRows, outputColumns, outputDepth);

        for (int f = 0; f < filters.length; f++) {
            Matrix3D filter = filters[f];
            for (int i = 0; i < outputRows; i++) {
                for (int j = 0; j < outputColumns; j++) {
                    double sum = 0.0;
                    // Sum across all input channels
                    for (int c = 0; c < input.getDepthCount(); c++) {
                        for (int fi = 0; fi < filter.getRowCount(); fi++) {
                            for (int fj = 0; fj < filter.getColumnCount(); fj++) {
                                sum += paddedInput.getEntry(i * stride + fi, j * stride + fj, c) * filter.getEntry(fi, fj, c);
                            }
                        }
                    }
                    result.setEntry(i, j, f, sum);
                }
            }
        }
        return result;
    }

    public static Matrix3D maxPool(Matrix3D input, int poolSize, int stride) {
        int outputRows = (input.getRowCount() - poolSize) / stride + 1;
        int outputColumns = (input.getColumnCount() - poolSize) / stride + 1;

        Matrix3D result = new Matrix3D(outputRows, outputColumns, input.getDepthCount());

        for (int d = 0; d < input.getDepthCount(); d++) {
            for (int i = 0; i < outputRows; i++) {
                for (int j = 0; j < outputColumns; j++) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int pi = 0; pi < poolSize; pi++) {
                        for (int pj = 0; pj < poolSize; pj++) {
                            max = Math.max(max, input.getEntry(i * stride + pi, j * stride + pj, d));
                        }
                    }
                    result.setEntry(i, j, d, max);
                }
            }
        }
        return result;
    }

    public double[] flatten() {
        return toFlatArray(this);
    }

    private static Matrix3D addPadding(Matrix3D input, int padding) {
        if (padding == 0) {
            return input;
        }

        int newRows = input.getRowCount() + 2 * padding;
        int newCols = input.getColumnCount() + 2 * padding;
        Matrix3D padded = new Matrix3D(newRows, newCols, input.getDepthCount());

        for (int d = 0; d < input.getDepthCount(); d++) {
            for (int i = 0; i < input.getRowCount(); i++) {
                for (int j = 0; j < input.getColumnCount(); j++) {
                    padded.setEntry(i + padding, j + padding, d, input.getEntry(i, j, d));
                }
            }
        }
        return padded;
    }

    public static void mapToMatrix3D(@NotNull Matrix3D matrix3D, TensorFunctionAble function) {
        for (int c = 0; c < matrix3D.getDepthCount(); c++) {
            for (int a = 0; a < matrix3D.getRowCount(); a++) {
                for (int b = 0; b < matrix3D.getColumnCount(); b++) {
                    matrix3D.setEntry(a, b, c, function.map(a, b, c, matrix3D.getEntry(a, b, c)));
                }
            }
        }
    }

    public static @NotNull Matrix3D Matrix3DMapping(@NotNull Matrix3D matrix3D, TensorFunctionAble function) {
        Matrix3D result = new Matrix3D(matrix3D.getRowCount(), matrix3D.getColumnCount(), matrix3D.getDepthCount());
        for (int c = 0; c < result.getDepthCount(); c++) {
            for (int a = 0; a < result.getRowCount(); a++) {
                for (int b = 0; b < result.getColumnCount(); b++) {
                    result.setEntry(a, b, c, function.map(a, b, c, matrix3D.getEntry(a, b, c)));
                }
            }
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("\n");
        for (int c = 0; c < this.getDepthCount(); c++) {
            stringBuilder.append("[");
            for (int a = 0; a < this.getRowCount(); a++) {
                stringBuilder.append("[");
                for (int b = 0; b < this.getColumnCount(); b++) {
                    stringBuilder.append(this.getEntry(a, b, c));
                    if (b != this.getColumnCount() - 1) {
                        stringBuilder.append(", ");
                    }
                }
                stringBuilder.append("]");
            }
            stringBuilder.append("]").append("\n");
        }
        stringBuilder.append("Matrix rows: ").append(this.getRowCount()).append(" columns: ").append(this.getColumnCount()).append(" depth: ").append(this.getDepthCount());
        return stringBuilder.toString();
    }
}