package com.mkproductions.jnn.gpu.gpu_layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

public class Flatten extends GeneralLayer {
    public Flatten() {
        super("Flatten", ActivationFunction.NONE);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.reshape(input.getData().getSize(), 1);
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
        return new Tensor[]{null, null, gradients.reshape(input.getShape().toHeapArray())};
    }

    @Override
    protected TornadoExecutionPlan[] prepareTaskGraphs() {
        return new TornadoExecutionPlan[0];
    }
}