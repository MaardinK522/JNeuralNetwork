package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;

public class PoolingLayer extends Layer {
    private final int poolSize;
    private final int stride;
    private final PoolingLayerType poolingLayerType;
    private Tensor cachedMaxIndices;

    public PoolingLayer(int poolSize, int stride, PoolingLayerType poolingLayerType) {
        super(poolingLayerType == PoolingLayerType.AVG ? "Average Pooling" : "Max Pooling", ActivationFunction.NONE);
        this.poolSize = poolSize;
        this.stride = stride;
        this.poolingLayerType = poolingLayerType;
    }

    @Override
    public String toString() {
        return STR."PoolingLayer{poolSize=\{poolSize}, stride=\{stride}}";
    }

    public int getPoolSize() {
        return poolSize;
    }

    public int getStride() {
        return stride;
    }

    public PoolingLayerType getPoolingLayerType() {
        return poolingLayerType;
    }

    @Override
    public Tensor forward(Tensor input) {
        if (poolingLayerType == PoolingLayer.PoolingLayerType.MAX) {
            return maxPool(input, poolSize, stride);
        } else if (poolingLayerType == PoolingLayer.PoolingLayerType.AVG) {
            return averagePool(input, poolSize, stride);
        }
        throw new IllegalStateException("Unknown pooling type.");
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
        Tensor propagatedGradients = new Tensor(input.getShape());
        if (poolingLayerType == PoolingLayer.PoolingLayerType.MAX) {
            propagatedGradients = maxPoolingBackward(propagatedGradients);
        } else if (poolingLayerType == PoolingLayer.PoolingLayerType.AVG) {
            propagatedGradients = averagePoolBackward(propagatedGradients);
        }
        return new Tensor[]{null, null, propagatedGradients};
    }

    private Tensor maxPoolingBackward(Tensor input) {
        int C = input.getShape()[0];
        int H_in = input.getShape()[1];
        int W_in = input.getShape()[2];

        int H_out = (H_in - poolSize) / stride + 1;
        int W_out = (W_in - poolSize) / stride + 1;

        if (H_out <= 0 || W_out <= 0) {
            throw new IllegalArgumentException("Pooling resulted in zero or negative dimensions.");
        }
        Tensor output = new Tensor(C, H_in, W_in);
        this.cachedMaxIndices = new Tensor(C, H_out, W_out, 2);

        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {

                    double max = Double.NEGATIVE_INFINITY;
                    int max_r = -1, max_c = -1;

                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    for (int p_h = 0; p_h < poolSize; p_h++) {
                        for (int p_w = 0; p_w < poolSize; p_w++) {
                            int r = h_start + p_h;
                            int c_in = w_start + p_w;
                            double entry = input.getEntry(c, r, c_in);

                            if (entry > max) {
                                max = entry;
                                max_r = r;
                                max_c = c_in;
                            }
                        }
                    }
                    output.setEntry(max, c, h_out, w_out);
                    this.cachedMaxIndices.setEntry(max_r, c, h_out, w_out, 0);
                    this.cachedMaxIndices.setEntry(max_c, c, h_out, w_out, 1);
                }
            }
        }
        return output;
    }

    private Tensor averagePoolBackward(Tensor input) {
        int C = input.getShape()[0];
        int H_in = input.getShape()[1];
        int W_in = input.getShape()[2];
        int H_out = (H_in - poolSize) / stride + 1;
        int W_out = (W_in - poolSize) / stride + 1;
        if (H_out <= 0 || W_out <= 0) {
            throw new IllegalArgumentException("Pooling resulted in zero or negative dimensions.");
        }
        Tensor output = new Tensor(C, H_out, W_out);
        int poolArea = poolSize * poolSize;
        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {
                    double sum = 0.0;
                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    for (int p_h = 0; p_h < poolSize; p_h++) {
                        for (int p_w = 0; p_w < poolSize; p_w++) {
                            sum += input.getEntry(c, h_start + p_h, w_start + p_w);
                        }
                    }
                    output.setEntry(sum / poolArea, c, h_out, w_out);
                }
            }
        }
        return output;
    }

    private static Tensor maxPool(Tensor input, int poolSize, int stride) {
        if (input.getRank() != 3) {
            throw new IllegalArgumentException("Input for pooling must be a Rank-3 Tensor (C, H, W).");
        }

        int C = input.getShape()[0];
        int H_in = input.getShape()[1];
        int W_in = input.getShape()[2];

        int H_out = (H_in - poolSize) / stride + 1;
        int W_out = (W_in - poolSize) / stride + 1;

        if (H_out <= 0 || W_out <= 0) {
            throw new IllegalArgumentException("Pooling resulted in zero or negative dimensions.");
        }

        Tensor output = new Tensor(C, H_out, W_out);

        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {

                    double max = Double.NEGATIVE_INFINITY;

                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    for (int p_h = 0; p_h < poolSize; p_h++) {
                        for (int p_w = 0; p_w < poolSize; p_w++) {
                            double entry = input.getEntry(c, h_start + p_h, w_start + p_w);
                            max = Math.max(max, entry);
                        }
                    }

                    output.setEntry(max, c, h_out, w_out);
                }
            }
        }

        return output;
    }

    private static Tensor averagePool(Tensor input, int poolSize, int stride) {
        if (input.getRank() != 3) {
            throw new IllegalArgumentException("Input for pooling must be a Rank-3 Tensor (C, H, W).");
        }

        int C = input.getShape()[0];
        int H_in = input.getShape()[1];
        int W_in = input.getShape()[2];

        int H_out = (H_in - poolSize) / stride + 1;
        int W_out = (W_in - poolSize) / stride + 1;

        if (H_out <= 0 || W_out <= 0) {
            throw new IllegalArgumentException("Pooling resulted in zero or negative dimensions.");
        }

        Tensor output = new Tensor(C, H_out, W_out);

        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                for (int w_out = 0; w_out < W_out; w_out++) {

                    double sum = 0.0;
                    int count = 0;

                    int h_start = h_out * stride;
                    int w_start = w_out * stride;

                    for (int p_h = 0; p_h < poolSize; p_h++) {
                        for (int p_w = 0; p_w < poolSize; p_w++) {
                            double entry = input.getEntry(c, h_start + p_h, w_start + p_w);
                            sum += entry;
                            count++;
                        }
                    }

                    double avg = sum / count;
                    output.setEntry(avg, c, h_out, w_out);
                }
            }
        }

        return output;
    }

    public enum PoolingLayerType {
        MAX, AVG
    }
}