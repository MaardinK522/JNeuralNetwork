package com.mkproductions.jnn.entity;


public final class SingleEntryThread extends Thread {
    private final double[] row;
    private final double[] column;
    private double summedResult = 0;

    public SingleEntryThread(double[] row, double[] column) {
        this.row = row;
        this.column = column;
    }

    public double[] row() {
        return this.row;
    }

    public double[] column() {
        return this.column;
    }

    @Override
    public void run() {
        super.run();
        for (int a = 0; a < row.length; a++) {
            summedResult += row[a] * column[a];
        }
    }

    public double getSummedResult() {
        return this.summedResult;
    }
}
