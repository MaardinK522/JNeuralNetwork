package com.mkproductions.jnn.gpu.gpu_layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.gpu.solver.TaskGraphTensorOperation;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class Pool extends GeneralLayer {
    private final int poolSize;
    private final int stride;
    private final PoolingType poolingLayerType;
    private IntArray cachedMaxIndices;

    public Pool(int poolSize, int stride, PoolingType poolingLayerType) {
        super(poolingLayerType == PoolingType.AVG ? "Average Pooling" : "Max Pooling", ActivationFunction.NONE);
        this.poolSize = poolSize;
        this.stride = stride;
        this.poolingLayerType = poolingLayerType;
    }

    public int getPoolSize() {
        return poolSize;
    }

    public int getStride() {
        return stride;
    }

    public PoolingType getPoolingLayerType() {
        return poolingLayerType;
    }


    @Override
    public Tensor forward(Tensor input) {
        if (input.getRank() != 3) {
            throw new IllegalArgumentException("Input for pooling must be a Rank-3 Tensor (C, H, W).");
        }
        this.input.copy(input.getData(), input.getShape());
        this.feedForwardExecutionPlan.execute();
        return output;
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
        this.setCurrentGradients(gradients);
        this.forward(input);
        return new Tensor[]{null, null, this.currentGradients};
    }

    @Override
    protected TornadoExecutionPlan[] prepareTaskGraphs() {
        int C = input.getShape().toHeapArray()[0];
        int Hin = input.getShape().toHeapArray()[1];
        int Win = input.getShape().toHeapArray()[2];
        int Hout = (Hin - poolSize) / stride + 1;
        int Wout = (Win - poolSize) / stride + 1;
        if (Hout <= 0 || Wout <= 0) {
            throw new IllegalArgumentException("Pooling resulted in zero or negative dimensions.");
        }
        // Preparing an execution plan for feed forward.
        TaskGraph feedForwardTaskGraph = new TaskGraph(STR."feedForwardTaskGraph:\{layerIndex}");
        feedForwardTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, input.getData(), output.getData(), cachedMaxIndices);
        if (poolingLayerType == PoolingType.MAX) {
            feedForwardTaskGraph.task(STR."avgPool:\{layerIndex}", TaskGraphTensorOperation::avgPoolForwardKernel, input.getData(), output.getData(), C, Hin, Win, Hout, Wout, this.poolSize, this.stride);
        } else if (poolingLayerType == PoolingType.AVG) {
            feedForwardTaskGraph.task(STR."maxPool:\{layerIndex}", TaskGraphTensorOperation::maxPoolForwardKernel, input.getData(), output.getData(), cachedMaxIndices, C, Hin, Win, Hout, Wout, this.poolSize, this.stride);
        } else {
            throw new IllegalStateException("Unknown pooling type.");
        }
        feedForwardTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, output.getData());
        this.feedForwardExecutionPlan = new TornadoExecutionPlan(feedForwardTaskGraph.snapshot());
        // Preparing an execution plan for back propagation.
        TaskGraph backPropagationTaskGraph = new TaskGraph(STR."backPropagationTaskGraph:\{layerIndex}");
        return new TornadoExecutionPlan[]{this.feedForwardExecutionPlan, backPropagationExecutionPlan};
    }

    public enum PoolingType {
        AVG, MAX
    }
}