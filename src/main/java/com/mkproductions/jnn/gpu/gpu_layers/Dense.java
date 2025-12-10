package com.mkproductions.jnn.gpu.gpu_layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.gpu.solver.TaskGraphTensorOperation;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class Dense extends GeneralLayer {
    private final int numberOfNeurons;
    private Tensor softmaxJacobian;

    public Dense(int numberOfNeurons, ActivationFunction activation) {
        super("Dense", activation);
        this.numberOfNeurons = numberOfNeurons;
    }

    @Override
    public String toString() {
        return STR."DenseLayer{numberOfNodes=\{this.numberOfNeurons}, activationFunction=\{getActivationFunction()}} \n\{this.getWeights()}";
    }

    public int getNumberOfNeurons() {
        return numberOfNeurons;
    }

    @Override
    public Tensor forward(Tensor input) {
        input.copy(this.input.getData(), input.getShape());
        this.feedForwardExecutionPlan.execute();
        return this.output;
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
        this.setCurrentGradients(gradients);
        this.forward(input);
        this.backPropagationExecutionPlan.execute();
        return new Tensor[]{this.weightsGradients, this.biasesGradients, this.previousLayerGradient};
    }

    @Override
    protected TornadoExecutionPlan[] prepareTaskGraphs() {
        // Preparing a feed forward an execution plan.
        TaskGraph feedForwardTaskGraph = new TaskGraph(STR."feedForwardPropagationTaskGraph:\{layerIndex}");
        TaskGraph backPropagationTaskGraph = new TaskGraph(STR."backPropagationTaskGraph:\{layerIndex}");

        feedForwardTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.input.getData(), weights.getData(), bias.getData(), output.getData());
        feedForwardTaskGraph.task(STR."weightedInput:\{layerIndex}", TaskGraphTensorOperation::matrixMultiplication, this.input.getData(), weights.getData(), output.getData(), this.input.getShape().toHeapArray()[1], this.input.getShape().toHeapArray()[0], weights.getShape().toHeapArray()[0]);
        feedForwardTaskGraph.task(STR."addingBias:\{layerIndex}", TaskGraphTensorOperation::addition, output.getData(), this.bias.getData(), output.getData(), output.getData().getSize());
        feedForwardTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, output.getData());
        this.feedForwardExecutionPlan = new TornadoExecutionPlan(feedForwardTaskGraph.snapshot());

        // Preparing a back propagation an execution plan.
        backPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, input.getData(), output.getData(), currentGradients.getData(), weights.getData(), weightsGradients.getData(), biasesGradients.getData(), this.inputTranspose.getData(), this.weightsTranspose.getData());
        backPropagationTaskGraph.task(STR."inputT:\{getLayerIndex()}", TaskGraphTensorOperation::transpose, this.input.getData(), this.inputTranspose.getData(), this.input.getShape().toHeapArray()[1], this.input.getShape().toHeapArray()[0], this.inputTranspose.getShape().toHeapArray()[0]);
        backPropagationTaskGraph.task(STR."weightsT:\{getLayerIndex()}", TaskGraphTensorOperation::transpose, this.weights.getData(), this.weightsTranspose.getData(), this.weights.getShape().toHeapArray()[1], this.weights.getShape().toHeapArray()[0], this.weightsTranspose.getShape().toHeapArray()[0]);
        if (this.getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            biasesGradients.copy(currentGradients.getData(), currentGradients.getShape());
        } else {
            backPropagationTaskGraph.task(STR."denseDerivative:\{getLayerIndex()}", TaskGraphTensorOperation::getAppliedTensorToDerivativeOfActivationFunction, output.getData(), biasesGradients.getData(), this.getActivationFunction().getIndex(), output.getData().getSize());
            backPropagationTaskGraph.task(STR."dL_dZ_EWM:\{getLayerIndex()}", TaskGraphTensorOperation::elementWiseMultiplication, biasesGradients.getData(), currentGradients.getData(), biasesGradients.getData(), biasesGradients.getData().getSize());
        }
        backPropagationTaskGraph.task(STR."deltaWeights:\{layerIndex}", TaskGraphTensorOperation::matrixMultiplication, biasesGradients.getData(), this.inputTranspose.getData(), this.weightsGradients.getData(), biasesGradients.getShape().toHeapArray()[1], biasesGradients.getShape().toHeapArray()[0], this.inputTranspose.getShape().toHeapArray()[0]);
        backPropagationTaskGraph.task(STR."previousGradients:\{layerIndex}", TaskGraphTensorOperation::matrixMultiplication, this.weightsTranspose.getData(), biasesGradients.getData(), previousLayerGradient.getData(), this.weightsTranspose.getShape().toHeapArray()[1], this.weightsTranspose.getShape().toHeapArray()[0], biasesGradients.getShape().toHeapArray()[0]);
        backPropagationTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, weightsGradients.getData(), biasesGradients.getData(), previousLayerGradient.getData());
        this.backPropagationExecutionPlan = new TornadoExecutionPlan(backPropagationTaskGraph.snapshot());
        return new TornadoExecutionPlan[]{this.feedForwardExecutionPlan, this.backPropagationExecutionPlan};
    }
}