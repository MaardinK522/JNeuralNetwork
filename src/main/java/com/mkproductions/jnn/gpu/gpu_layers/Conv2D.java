package com.mkproductions.jnn.gpu.gpu_layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.gpu.solver.TaskGraphTensorOperation;
import org.jetbrains.annotations.NotNull;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

public class Conv2D extends GeneralLayer {
    private final int filterSize;
    private final int numberOfFilters;
    private final int stride;
    private final int padding;

    public Conv2D(int filterSize, int numberOfFilters, int stride, int padding, ActivationFunction activationFunction) {
        super("Convolution", activationFunction);
        this.filterSize = filterSize;
        this.numberOfFilters = numberOfFilters;
        this.stride = stride;
        this.padding = padding;
    }

    public int getFilterSize() {
        return filterSize;
    }

    public int getNumberOfFilters() {
        return numberOfFilters;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }


    @Override
    public @NotNull String toString() {
        return STR."ConvolutionLayer{filterSize=\{filterSize}, activationFunction=\{getActivationFunction()}}";
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input.copy(input.getData(), input.getShape());
        this.feedForwardExecutionPlan.execute();
        this.output.copy(output.getData(), output.getShape());
        return output;
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
        this.setCurrentGradients(gradients);
        this.forward(input);
        this.backPropagationExecutionPlan.execute();
        return new Tensor[]{this.weightsGradients, this.biasesGradients, this.currentGradients};
    }

    @Override
    protected TornadoExecutionPlan[] prepareTaskGraphs() {
        // Preparing a feed forward an execution plan.
        TaskGraph feedForwardTaskGraph = new TaskGraph(STR."feedForwardPropagationTaskGraph:\{layerIndex}");
        TaskGraph backPropagationTaskGraph = new TaskGraph(STR."backPropagationTaskGraph:\{layerIndex}");

        int inputChannels = input.getShape().toHeapArray()[0];
        int inputHeight = input.getShape().toHeapArray()[1];
        int inputWidth = input.getShape().toHeapArray()[2];
        int filterSize = weights.getShape().toHeapArray()[2];
        int outputHeight = output.getShape().toHeapArray()[1];
        int outputWidth = output.getShape().toHeapArray()[2];
        feedForwardTaskGraph.task(STR."convolution:\{this.layerIndex}", TaskGraphTensorOperation::convForwardKernel, input.getData(), weights.getData(), bias.getData(), rawOutput.getData(), inputChannels, inputHeight, inputWidth, filterSize, outputHeight, outputWidth, stride, padding);
        feedForwardTaskGraph.task(STR."applyingActivation:\{layerIndex}", TaskGraphTensorOperation::getAppliedTensorToActivationFunction, rawOutput.getData(), output.getData(), getActivationFunction().getIndex(), rawOutput.getData().getSize());
        return new TornadoExecutionPlan[]{new TornadoExecutionPlan(feedForwardTaskGraph.snapshot()), new TornadoExecutionPlan(backPropagationTaskGraph.snapshot())};
    }
}