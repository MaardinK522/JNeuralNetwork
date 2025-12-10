package com.mkproductions.jnn.gpu.solver;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class TaskGraphTensorOperation {
    public static final DoubleArray SOFTMAX_SUM_BUFFER = new DoubleArray(1);

    public static void addition(DoubleArray arr1, DoubleArray arr2, DoubleArray arrResult, int N) {
        for (@Parallel int flatIndex = 0; flatIndex < N; flatIndex++) {
            arrResult.set(flatIndex, arr1.get(flatIndex) + arr2.get(flatIndex));
        }
    }

    public static void subtraction(DoubleArray arr1, DoubleArray arr2, DoubleArray arrResult, int N) {
        for (@Parallel int flatIndex = 0; flatIndex < N; flatIndex++) {
            arrResult.set(flatIndex, arr1.get(flatIndex) - arr2.get(flatIndex));
        }
    }

    public static void scalarMultiplication(DoubleArray arr1, DoubleArray scaleBuffer, DoubleArray arrResult, int N) {
        double scale = scaleBuffer.get(0);
        for (@Parallel int flatIndex = 0; flatIndex < N; flatIndex++) {
            arrResult.set(flatIndex, arr1.get(flatIndex) * scale);
        }
    }

    public static void elementWiseMultiplication(DoubleArray arr1, DoubleArray arr2, DoubleArray arrResult, int N) {
        for (@Parallel int flatIndex = 0; flatIndex < N; flatIndex++) {
            arrResult.set(flatIndex, arr1.get(flatIndex) * arr2.get(flatIndex));
        }
    }

    public static void elementWiseDivision(DoubleArray array1, DoubleArray array2, DoubleArray result, int N) {
        for (@Parallel int flatIndex = 0; flatIndex < array1.getSize(); flatIndex++) {
            result.set(flatIndex, array1.get(flatIndex) / array2.get(flatIndex));
        }
    }

    public static void fill(DoubleArray array, DoubleArray valueBuffer, int N) {
        double value = valueBuffer.get(0);
        for (@Parallel int flatIndex = 0; flatIndex < array.getSize(); flatIndex++) {
            array.set(flatIndex, value);
        }
    }

    public static void matrixMultiplication(DoubleArray A, DoubleArray B, DoubleArray C, int rowsA, int colsA, int colsB) {
        int resultSize = rowsA * colsB;
        for (@Parallel int idx = 0; idx < resultSize; idx++) {
            int i = idx / colsB;
            int j = idx % colsB;
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                int flatIndexA = i * colsA + k;
                int flatIndexB = k * colsB + j;
                sum += A.get(flatIndexA) * B.get(flatIndexB);
            }
            C.set(idx, sum);
        }
    }

    public static void transpose(DoubleArray input, DoubleArray result, int inputRows, int inputCols, int resultCols) {
        int totalSize = inputRows * inputCols;
        for (@Parallel int flatInputIndex = 0; flatInputIndex < totalSize; flatInputIndex++) {
            int inputRow = flatInputIndex / inputCols;
            int inputCol = flatInputIndex % inputCols;
            int flatResultIndex = (inputCol * resultCols) + inputRow;
            result.set(flatResultIndex, input.get(flatInputIndex));
        }
    }

    public static void clip(DoubleArray input, DoubleArray result, int totalSize, double start, double end) {
        for (@Parallel int i = 0; i < totalSize; i++) {
            double value = input.get(i);
            double clippedValue = TornadoMath.max(TornadoMath.min(value, end), start);
            result.set(i, clippedValue);
        }
    }

    public static void setSlice(DoubleArray hostData, DoubleArray sliceData, int sliceSize, int[] hostStrides, int[] sliceStrides, int[] startIndices, int hostRank, int sliceRank) {
        for (@Parallel int flatSliceIndex = 0; flatSliceIndex < sliceSize; flatSliceIndex++) {
            int flatHostIndex = 0;
            int dimOffset = hostRank - sliceRank;
            int tempIndex = flatSliceIndex;
            for (int d = 0; d < sliceRank; d++) {
                int hostDimIndex = d + dimOffset;
                int sliceCoordinate = tempIndex / sliceStrides[d];
                tempIndex %= sliceStrides[d];
                int hostCoordinate = startIndices[hostDimIndex] + sliceCoordinate;
                flatHostIndex += hostCoordinate * hostStrides[hostDimIndex];
            }
            for (int d = 0; d < dimOffset; d++) {
                flatHostIndex += startIndices[d] * hostStrides[d];
            }
            hostData.set(flatHostIndex, sliceData.get(flatSliceIndex));
        }
    }

    public static void getSlice(DoubleArray input, DoubleArray result, int sliceSize, int[] inputStrides, int[] resultStrides, int resultRank, int axis, int start) {
        for (@Parallel int flatResultIndex = 0; flatResultIndex < sliceSize; flatResultIndex++) {
            int flatInputIndex = 0;
            int tempIndex = flatResultIndex;
            for (int d = 0; d < resultRank; d++) {
                int resultCoord = tempIndex / resultStrides[d];
                tempIndex %= resultStrides[d];
                int inputCoord = resultCoord;
                if (d == axis) {
                    inputCoord += start;
                }
                flatInputIndex += inputCoord * inputStrides[d];
            }
            result.set(flatResultIndex, input.get(flatInputIndex));
        }
    }

    public static void correlate2D(DoubleArray inputData, DoubleArray deltaData, DoubleArray gradData, int paddedWidth, int deltaHeight, int deltaWidth, int filterSize, int stride) {
        int gradSize = filterSize * filterSize;
        for (@Parallel int flatIndex = 0; flatIndex < gradSize; flatIndex++) {
            int fr = flatIndex / filterSize;
            int fc = flatIndex % filterSize;
            double sum = 0.0;
            for (int ho = 0; ho < deltaHeight; ho++) {
                for (int wo = 0; wo < deltaWidth; wo++) {
                    int h_in_start = ho * stride + fr;
                    int w_in = wo * stride + fc;
                    int flatInputIndex = (h_in_start * paddedWidth) + w_in;
                    int flatDeltaIndex = (ho * deltaWidth) + wo;
                    double valI = inputData.get(flatInputIndex);
                    double valD = deltaData.get(flatDeltaIndex);
                    sum += valI * valD;
                }
            }
            gradData.set(flatIndex, sum);
        }
    }

    public static void getAppliedTensorToActivationFunction(DoubleArray array, DoubleArray result, int activationFunctionIndex, int N) {
        for (@Parallel int index = 0; index < N; index++) {
            double value = array.get(index);
            if (activationFunctionIndex == 0) {
                result.set(index, 0.0);
            } else if (activationFunctionIndex == 1) {
                result.set(index, 1.0 / (1.0 + Math.exp(-value)));
            } else if (activationFunctionIndex == 2) {
                result.set(index, Math.max(0.0, value));
            } else if (activationFunctionIndex == 3) {
                result.set(index, value);
            } else if (activationFunctionIndex == 4) {
                result.set(index, TornadoMath.tanh(value));
            }
        }
    }

    public static void getAppliedTensorToDerivativeOfActivationFunction(DoubleArray array, DoubleArray result, int activationFunctionIndex, int N) {
        for (@Parallel int index = 0; index < N; index++) {
            double value = array.get(index);
            if (activationFunctionIndex == 0) {
                result.set(index, 0.0);
            } else if (activationFunctionIndex == 1) {
                result.set(index, value * (1.0 - value));
            } else if (activationFunctionIndex == 2) {
                double resultantValue = 0;
                if (value > 0) resultantValue = 1;
                result.set(index, resultantValue);
            } else if (activationFunctionIndex == 3) {
                result.set(index, 1.0);
            } else if (activationFunctionIndex == 4) {
                result.set(index, 1 - (value * value));
            }
        }
    }

    public static void softmaxExpKernel(DoubleArray inputData, DoubleArray resultData, int totalSize) {
        for (@Parallel int i = 0; i < totalSize; i++) {
            resultData.set(i, TornadoMath.exp(inputData.get(i)));
        }
    }

    public static void softmaxSumKernel(DoubleArray inputExp, DoubleArray sumResult, int totalSize) {
        for (@Reduce int i = 0; i < totalSize; i++) {
            sumResult.set(0, sumResult.get(0) + inputExp.get(i));
        }
    }

    public static void softmaxDivideKernel(DoubleArray inputExp, DoubleArray sumInput, DoubleArray output, int totalSize) {
        double sum = sumInput.get(0);
        for (@Parallel int i = 0; i < totalSize; i++) {
            output.set(i, inputExp.get(i) / sum);
        }
    }

    public static void softmaxDerivativeKernel(DoubleArray softmaxOutput, DoubleArray jacobianResult, int totalSize) {
        int totalElements = totalSize * totalSize;
        for (@Parallel int flatIndex = 0; flatIndex < totalElements; flatIndex++) {
            int i = flatIndex / totalSize;
            int j = flatIndex % totalSize;
            double Si = softmaxOutput.get(i);
            double Sj = softmaxOutput.get(j);
            double delta_ij = (i == j) ? 1.0 : 0.0;
            double J_ij = Si * (delta_ij - Sj);
            jacobianResult.set(flatIndex, J_ij);
        }
    }

    public static void randomize(DoubleArray arr, int N, long nanoTime) {
        for (@Parallel int flatIndex = 0; flatIndex < N; flatIndex++) {
            arr.set(flatIndex, getRandomValue(flatIndex, 0, nanoTime));
        }
    }

    public static double getRandomValue(int i, int j, long nanoTime) {
        long seed = (long) (i + 1) * (j + 1) + nanoTime;
        // Generate two rounds of pseudo random number (using LCG logic)
        // The mask & ((1L << 48) - 1) ensures the result stays within 48 bits.
        seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
        seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
        // 1. Generate number between [0, 1]
        // 268435455 is 2^28-1, which corresponds to the 0x0FFFFFFF mask.
        double originalValue = (seed & 0x0FFFFFFF) / 268435455.0; // Use DoubleArray literal for high precision
        // 2. Transform the range from [0, 1] to [-1, 1]
        return (originalValue * 2.0) - 1.0;
    }

    public static void convForwardKernel(DoubleArray input, DoubleArray weights, DoubleArray bias, DoubleArray output, int inputChannels, int inputHeight, int inputWidth, int filterSize, int outputHeight, int outputWidth, int stride, int padding) {
        int totalOutputElements = output.getSize();
        for (@Parallel int idx = 0; idx < totalOutputElements; idx++) {
            int pixelsPerMap = outputHeight * outputWidth;
            int filterIndex = idx / pixelsPerMap;
            int rem = idx % pixelsPerMap;
            int outY = rem / outputWidth;
            int outX = rem % outputWidth;
            double sum = 0.0;
            for (int c = 0; c < inputChannels; c++) {
                int inputChannelOffset = c * (inputHeight * inputWidth);
                int weightChannelOffset = filterIndex * (inputChannels * filterSize * filterSize) + c * (filterSize * filterSize);
                for (int ky = 0; ky < filterSize; ky++) {
                    for (int kx = 0; kx < filterSize; kx++) {
                        int inY = outY * stride + ky - padding;
                        int inX = outX * stride + kx - padding;
                        if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth) {
                            double valI = input.get(inputChannelOffset + inY * inputWidth + inX);
                            double valW = weights.get(weightChannelOffset + ky * filterSize + kx);
                            sum += valI * valW;
                        }
                    }
                }
            }
            sum += bias.get(filterIndex);
            output.set(idx, sum);
        }
    }

    /**
     * Max Pooling Forward Pass Kernel.
     * Parallelizes over: Output Indices (Channel, OutputY, OutputX).
     * Caches the indices (max_r, max_c) into the indices buffer.
     */
    public static void maxPoolForwardKernel(DoubleArray input, DoubleArray output, IntArray cachedMaxIndices, int C, int Hin, int Win, int Hout, int Wout, int poolSize, int stride) {
        int totalOutputElements = C * Hout * Wout;
        int pixelsPerMap = Hout * Wout;
        for (@Parallel int idx = 0; idx < totalOutputElements; idx++) {
            int c = idx / pixelsPerMap;
            int rem = idx % pixelsPerMap;
            int outY = rem / Wout;
            int outX = rem % Wout;
            double max = Double.NEGATIVE_INFINITY;
            int maxR = -1, maxC = -1;
            int hStart = outY * stride;
            int wStart = outX * stride;
            for (int p_h = 0; p_h < poolSize; p_h++) {
                for (int p_w = 0; p_w < poolSize; p_w++) {
                    int r = hStart + p_h;
                    int c_in = wStart + p_w;
                    int flatInputIndex = (c * (Hin * Win)) + (r * Win) + c_in;
                    double entry = input.get(flatInputIndex);
                    if (entry > max) {
                        max = entry;
                        maxR = r;
                        maxC = c_in;
                    }
                }
            }
            output.set(idx, max);
            int indexBase = idx * 2;
            cachedMaxIndices.set(indexBase, maxR);
            cachedMaxIndices.set(indexBase + 1, maxC);
        }
    }

    /**
     * Average Pooling Forward Pass Kernel.
     * Parallelizes over: Output Indices (Channel, OutputY, OutputX).
     */
    public static void avgPoolForwardKernel(DoubleArray input, DoubleArray output, int C, int Hin, int Win, int Hout, int Wout, int poolSize, int stride) {
        int totalOutputElements = C * Hout * Wout;
        int pixelsPerMap = Hout * Wout;
        int poolArea = poolSize * poolSize;
        for (@Parallel int idx = 0; idx < totalOutputElements; idx++) {
            int c = idx / pixelsPerMap;
            int rem = idx % pixelsPerMap;
            int outY = rem / Wout;
            int outX = rem % Wout;
            double sum = 0.0;
            int hStart = outY * stride;
            int wStart = outX * stride;
            for (int p_h = 0; p_h < poolSize; p_h++) {
                for (int p_w = 0; p_w < poolSize; p_w++) {
                    int r = hStart + p_h;
                    int c_in = wStart + p_w;
                    int flatInputIndex = (c * (Hin * Win)) + (r * Win) + c_in;
                    sum += input.get(flatInputIndex);
                }
            }
            output.set(idx, sum / poolArea);
        }
    }

    // Mean Square Error Kernels
    public static void meanSquaredErrorKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            double diff = output.get(idx) - target.get(idx);
            error.set(idx, diff * diff);
        }
    }

    public static void meanSquaredErrorDerivativeKernel(DoubleArray output, DoubleArray target, DoubleArray error, DoubleArray delta, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            double diff = output.get(idx) - target.get(idx);
            error.set(idx, 2 * diff * delta.get(idx));
        }
    }

    // Mean Absolute Error Kernel
    public static void meanAbsoluteErrorKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            double diff = TornadoMath.abs(output.get(idx) - target.get(idx));
            error.set(idx, diff);
        }
    }

    public static void meanSquaredErrorDerivativeKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (int idx = 0; idx < N; idx++) {
            double difference = target.get(idx) - output.get(idx);
            error.set(idx, difference >= 0 ? 1 : -1);
        }
    }

    // Log(cosh( ))
    public static void logCosKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            double difference = output.get(idx) - target.get(idx);
            error.set(idx, Math.log(Math.cos(difference)));
        }
    }

    public static void logCosDerivativeKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            double difference = target.get(idx) - output.get(idx);
            error.set(idx, -Math.tan(difference));
        }
    }

    // Binary Cross Entropy
    public static void binaryCrossEntropyKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            double p = output.get(idx);
            double y = target.get(idx);

            p = TornadoMath.max(p, 1e-15);
            p = TornadoMath.min(p, 1 - 1e-15);
            error.set(idx, -(y * TornadoMath.log(p) + (1 - y) * TornadoMath.log(1 - p)));
        }
    }

    public static void binaryCrossEntropyDerivativeKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            error.set(idx, target.get(idx) - output.get(idx));
        }
    }

    // Categorical Cross Entropy
    public static void categoricalCrossEntropyKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            double p = output.get(idx);
            p = TornadoMath.max(p, 1e-15);
            p = TornadoMath.min(p, 1 - 1e-15);
            error.set(idx, -target.get(idx) * TornadoMath.log(p));
        }
    }

    public static void categoricalCrossEntropyDerivativeKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            error.set(idx, output.get(idx) - target.get(idx));
        }
    }

    // Sparse Categorical Cross Entropy
    public static void sparseCategoricalCrossEntropyKernel(DoubleArray output, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            double p = output.get(idx);
            p = TornadoMath.max(p, 1e-15);
            error.set(idx, -TornadoMath.log(p));
        }
    }

    public static void sparseCategoricalCrossEntropyDerivativeKernel(DoubleArray output, DoubleArray target, DoubleArray error, int N) {
        for (@Parallel int idx = 0; idx < N; idx++) {
            error.set(idx, output.get(idx) - target.get(idx));
        }
    }
}