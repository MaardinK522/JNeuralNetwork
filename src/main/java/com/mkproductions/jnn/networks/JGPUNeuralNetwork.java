package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.gpu.gpu_layers.*;
import com.mkproductions.jnn.gpu.solver.TaskGraphTensorOperation;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;

import java.security.SecureRandom;
import java.util.Random;

import static com.mkproductions.jnn.gpu.solver.TaskGraphTensorOperation.SOFTMAX_SUM_BUFFER;

public class JGPUNeuralNetwork {
    private final GeneralLayer[] networkLayers;
    private static final Random random = new SecureRandom();
    private LossFunction lossFunction;
    private final JNetworkOptimizer optimizer;
    private double learningRate;
    private double momentumFactorBeta1;
    private double momentumFactorBeta2;
    private double epsilon = 1e-8;
    private boolean debugMode = false;
    private final Tensor finalOutputTensor;
    private final Tensor targetTensor;
    private final Tensor lossTensor;
    private final Tensor deactivatedTensor;
    private final Tensor gradients;
    private Tensor weightsGradient;
    private Tensor biasesGradient;
    private TaskGraph calculateLossTaskGraph;

    public JGPUNeuralNetwork(int[] inputShapeDimension, LossFunction lossFunction, JNetworkOptimizer optimizer, GeneralLayer... networkLayers) {
        this.networkLayers = networkLayers;
        if (this.networkLayers.length == 0) {
            throw new IllegalArgumentException("Network must contain at least one layer.");
        }
        int inputDepth;
        int inputWidth;
        int inputHeight;
        if (inputShapeDimension.length > 0) {
            if (inputShapeDimension.length == 1) {
                inputDepth = 1;
                inputHeight = 1;
                inputWidth = inputShapeDimension[0];
            } else if (inputShapeDimension.length == 3) {
                inputDepth = inputShapeDimension[0];
                inputHeight = inputShapeDimension[1];
                inputWidth = inputShapeDimension[2];
            } else {
                throw new IllegalArgumentException("Input shape must be length 1 (Dense) or 3 (Conv).");
            }
        } else {
            throw new IllegalArgumentException("Input shape must contain at least one dimension.");
        }
        this.learningRate = 0.01;
        int currentInputDepth = inputDepth;
        int currentInputWidth = inputWidth;
        int currentInputHeight = inputHeight;
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            GeneralLayer layer = this.networkLayers[layerIndex];
            switch (layer) {
                case Conv2D convolutionLayer -> {
                    int outputHeight = (currentInputHeight + 2 * convolutionLayer.getPadding() - convolutionLayer.getFilterSize()) / convolutionLayer.getStride() + 1;
                    int outputWidth = (currentInputWidth + 2 * convolutionLayer.getPadding() - convolutionLayer.getFilterSize()) / convolutionLayer.getStride() + 1;
                    if (outputWidth < 1 || outputHeight < 1) {
                        throw new IllegalArgumentException(STR."Convolution at index \{layerIndex} resulted in zero or negative dimensions.");
                    }
                    convolutionLayer.setWeights(new Tensor(convolutionLayer.getNumberOfFilters(), currentInputDepth, convolutionLayer.getFilterSize(), convolutionLayer.getFilterSize()));
                    convolutionLayer.setBias(new Tensor(convolutionLayer.getNumberOfFilters()));
                    int[] inputShape = {currentInputDepth, currentInputWidth, currentInputHeight};
                    currentInputDepth = convolutionLayer.getNumberOfFilters();
                    currentInputWidth = outputWidth;
                    currentInputHeight = outputHeight;
                    int[] outputShape = {currentInputDepth, currentInputWidth, currentInputHeight};
                    layer.initLayerParameters(layerIndex, this.momentumFactorBeta1, this.momentumFactorBeta2, this.epsilon, this.learningRate, inputShape, outputShape);
                }

                case Pool poolingLayer -> {
                    int outputHeight = (currentInputHeight - poolingLayer.getPoolSize()) / poolingLayer.getStride() + 1;
                    int outputWidth = (currentInputWidth - poolingLayer.getPoolSize()) / poolingLayer.getStride() + 1;
                    if (outputWidth < 1 || outputHeight < 1) {
                        throw new IllegalArgumentException(STR."Pooling at index \{layerIndex} resulted in zero or negative dimensions.");
                    }
                    int[] inputShape = {currentInputDepth, currentInputWidth, currentInputHeight};
                    currentInputWidth = outputWidth;
                    currentInputHeight = outputHeight;
                    int[] outputShape = {currentInputDepth, currentInputWidth, currentInputHeight};
                    layer.initLayerParameters(layerIndex, this.momentumFactorBeta1, this.momentumFactorBeta2, this.epsilon, this.learningRate, inputShape, outputShape);
                }

                case Flatten _ -> {
                    if (currentInputHeight == 1 && currentInputDepth == 1) {
                        throw new IllegalArgumentException(STR."Redundant FlattenLayer at index \{layerIndex}. Input is already flat (Dense/1D).");
                    }
                    currentInputWidth = currentInputWidth * currentInputHeight * currentInputDepth;
                    currentInputHeight = 1;
                    currentInputDepth = 1;
                }

                case Dense denseLayer -> {
                    if (currentInputDepth > 1 || currentInputHeight > 1) {
                        throw new IllegalArgumentException(STR."DenseLayer at index \{layerIndex} can only accept 1D input. A FlattenLayer is required after Convolution/Pooling.");
                    }
                    denseLayer.setWeights(new Tensor(denseLayer.getNumberOfNeurons(), currentInputWidth));
                    denseLayer.setBias(new Tensor(denseLayer.getNumberOfNeurons(), 1));
                    int[] inputShape = {currentInputDepth, currentInputWidth, currentInputHeight};
                    currentInputWidth = denseLayer.getNumberOfNeurons();
                    currentInputHeight = 1;
                    currentInputDepth = 1;
                    int[] outputShape = {currentInputDepth, currentInputWidth, currentInputHeight};
                    layer.initLayerParameters(layerIndex, this.momentumFactorBeta1, this.momentumFactorBeta2, this.epsilon, this.learningRate, inputShape, outputShape);
                }

                case null, default -> {
                    System.err.println(this.networkLayers[layerIndex]);
                    throw new IllegalArgumentException("Unsupported layer type detected in network architecture.");
                }
            }
//            if (!(layer instanceof Flatten)) {
//                layer.initLayerParameters(layerIndex, this.momentumFactorBeta1, this.momentumFactorBeta2, this.epsilon, this.learningRate);
//            }
        }
        if (this.networkLayers[this.networkLayers.length - 1].getBias().getData().getSize() <= 1 && this.networkLayers[this.networkLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            throw new IllegalArgumentException("When applied softmax, the last layer must have at least 2 nodes.");
        } else if ((this.lossFunction == LossFunction.SPARSE_CATEGORICAL_CROSS_ENTROPY || this.lossFunction == LossFunction.CATEGORICAL_CROSS_ENTROPY) && !this.networkLayers[this.networkLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            throw new IllegalArgumentException("SCCE and CCE loss functions requires a softmax or sigmoid activation function at the end of the network.");
        }
        this.finalOutputTensor = new Tensor(this.networkLayers[this.networkLayers.length - 1].getBias().getShape().toHeapArray());
        this.targetTensor = new Tensor(this.networkLayers[this.networkLayers.length - 1].getBias().getShape().toHeapArray());
        this.lossTensor = new Tensor(this.networkLayers[this.networkLayers.length - 1].getBias().getShape().toHeapArray());
        this.deactivatedTensor = new Tensor(this.networkLayers[this.networkLayers.length - 1].getBias().getShape().toHeapArray());
        this.gradients = new Tensor(this.networkLayers[this.networkLayers.length - 1].getBias().getShape().toHeapArray());

        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.prepareLossTaskGraph();
    }

    public Tensor[] predict(Tensor inputs) {
        return forwardPropagation(inputs);
    }

    private Tensor[] forwardPropagation(Tensor inputs) {
        Tensor[] outputs = new Tensor[this.networkLayers.length];
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            inputs = outputs[layerIndex] = this.networkLayers[layerIndex].forward(layerIndex == 0 ? inputs : outputs[layerIndex - 1]);
        }
        return outputs;
    }

    private void backPropagation(Tensor inputTensor, Tensor targetTensor) {
        Tensor[] outputTensors = this.forwardPropagation(inputTensor);
        this.finalOutputTensor.copy(outputTensors[this.networkLayers.length - 1].getData(), outputTensors[this.networkLayers.length - 1].getShape());
        this.targetTensor.copy(targetTensor.getData(), targetTensor.getShape());
        Tensor currentGradient = null;
        for (int layerIndex = this.networkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            GeneralLayer layer = this.networkLayers[layerIndex];
            if (layerIndex == this.networkLayers.length - 1) {
                calculateLastLayerGradients();
                currentGradient.copy(gradients.getData(), gradients.getShape());
            }
            Tensor[] backwardData = layer.backward(inputTensor, currentGradient);
            this.weightsGradient.copy(backwardData[0].getData(), backwardData[0].getShape());
            this.biasesGradient.copy(backwardData[1].getData(), backwardData[1].getShape());
            currentGradient.copy(backwardData[2].getData(), backwardData[2].getShape());
            if (!(layer instanceof Pool) && !(layer instanceof Flatten)) {
                layer.setWeightsGradients(weightsGradient);
                layer.setBiasesGradients(biasesGradient);
                switch (this.optimizer) {
                    case SGD -> layer.backPropagationSGD();
                    case SGD_MOMENTUM -> layer.backPropagationSGDWithMomentum();
                    case RMS_PROP -> layer.backPropagationRMSPropagation();
                    case ADA_GARD -> layer.backPropagationAdaGrad();
                    case ADAM -> layer.backPropagationAdam();
                }
            }
        }
    }


    private void calculateLastLayerGradients() {
        new TornadoExecutionPlan(this.calculateLossTaskGraph.snapshot()).execute();
    }

    private void prepareLossTaskGraph() {
        this.calculateLossTaskGraph = new TaskGraph("CalculatingGradients.");
        calculateLossTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, finalOutputTensor.getData(), targetTensor.getData(), lossTensor.getData(), deactivatedTensor.getData(), gradients.getData());
        if (((this.lossFunction.equals(LossFunction.CATEGORICAL_CROSS_ENTROPY) || this.lossFunction.equals(LossFunction.SPARSE_CATEGORICAL_CROSS_ENTROPY)) && this.networkLayers[this.networkLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX))) {
            calculateLossTaskGraph.task("lastLayerSubtraction", TaskGraphTensorOperation::subtraction, finalOutputTensor.getData(), targetTensor.getData(), gradients.getData(), finalOutputTensor.getData().getSize());
        } else {
            String taskID = "calculateLoss:";
            int N = finalOutputTensor.getData().getSize();
            // 1. Calculating Loss function derivative
            switch (this.lossFunction) {
                case LOG_COSH -> calculateLossTaskGraph.task(taskID, TaskGraphTensorOperation::logCosDerivativeKernel, finalOutputTensor.getData(), targetTensor.getData(), lossTensor.getData(), N);
                case MEAN_SQUARED_ERROR ->
                        calculateLossTaskGraph.task(taskID, TaskGraphTensorOperation::meanSquaredErrorDerivativeKernel, finalOutputTensor.getData(), targetTensor.getData(), lossTensor.getData(), N);
                case MEAN_ABSOLUTE_ERROR ->
                        calculateLossTaskGraph.task(taskID, TaskGraphTensorOperation::meanAbsoluteErrorKernel, finalOutputTensor.getData(), targetTensor.getData(), lossTensor.getData(), N);
                case BINARY_CROSS_ENTROPY ->
                        calculateLossTaskGraph.task(taskID, TaskGraphTensorOperation::binaryCrossEntropyDerivativeKernel, finalOutputTensor.getData(), targetTensor.getData(), lossTensor.getData(), N);
                case CATEGORICAL_CROSS_ENTROPY ->
                        calculateLossTaskGraph.task(taskID, TaskGraphTensorOperation::categoricalCrossEntropyDerivativeKernel, finalOutputTensor.getData(), targetTensor.getData(), lossTensor.getData(), N);
                case SPARSE_CATEGORICAL_CROSS_ENTROPY ->
                        calculateLossTaskGraph.task(taskID, TaskGraphTensorOperation::sparseCategoricalCrossEntropyDerivativeKernel, finalOutputTensor.getData(), targetTensor.getData(), lossTensor.getData(), N);
                case null, default -> throw new IllegalArgumentException(STR."Unsupported lossFunction: \{this.lossFunction}");
            }
            // 2. Calculating derivative of the activation function.
            calculateLossTaskGraph.task("applyDerivative:", TaskGraphTensorOperation::getAppliedTensorToDerivativeOfActivationFunction, finalOutputTensor.getData(), deactivatedTensor.getData(), this.networkLayers[this.networkLayers.length - 1].getActivationFunction().getIndex(), finalOutputTensor.getData().getSize());
            // 3. Calculating gradients.
            calculateLossTaskGraph.task("lastLayerGradients:", TaskGraphTensorOperation::elementWiseMultiplication, lossTensor.getData(), deactivatedTensor.getData(), gradients.getData(), lossTensor.getData().getSize());
        }
        calculateLossTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, gradients.getData());
    }

    public void train(Tensor[] trainingInputs, Tensor[] trainingOutputs, int epochs) {
        if (trainingInputs.length != trainingOutputs.length) {
            throw new IllegalArgumentException("Training inputs and outputs must be of the same length.");
        } else if (trainingOutputs[0].getData().getSize() != this.networkLayers[this.networkLayers.length - 1].getBias().getData().getSize()) {
            System.err.println(STR."Input: Tensor: \{trainingInputs[0]}");
            System.err.println(STR."Output: Tensor: \{trainingOutputs[0]}");
            System.err.println(STR."Output bias: \{this.networkLayers[this.networkLayers.length - 1].getBias()}");
            throw new IllegalArgumentException("Training outputs must match the number of nodes in the last layer.");
        }
        int lastProgress = -1;
        for (int epoch = 0; epoch < epochs; epoch++) {
            long start = System.currentTimeMillis();
            int progress;
            for (int trainingIndex = 0; trainingIndex < trainingInputs.length; trainingIndex++) {
                progress = (trainingIndex * 100) / trainingInputs.length;
                Tensor trainingInput = trainingInputs[trainingIndex];
                Tensor trainingOutput = trainingOutputs[trainingIndex];
                this.backPropagation(trainingInput, trainingOutput);
                if (this.debugMode && progress != lastProgress) {
                    lastProgress = progress;
                    int filled = progress / 2;
                    int empty = 50 - filled;
                    String filledBar = "#".repeat(filled);
                    String emptyBar = " ".repeat(empty);
                    System.out.printf("\rTraining progress: %d%% [%s%s]", progress, filledBar, emptyBar);
                    if (progress >= 100) {
                        System.out.println();
                    }
                }
            }
            long end = System.currentTimeMillis() - start;
            System.out.println(STR."Epoch count: \{epoch}, Time taken: \{end}ms");
        }
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void printData() {
        for (GeneralLayer layer : this.networkLayers) {
            System.out.println(layer);
        }
    }

    public Tensor processInputs(Tensor inputs) {
        return this.forwardPropagation(inputs)[this.networkLayers.length - 1];
    }

    public static TaskGraph applyActivationFunctionToTensor(TaskGraph taskGraph, String taskID, Tensor tensor, Tensor result, ActivationFunction activationFunction) {
        Tensor.validateTensors(tensor, result);
        taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, tensor.getData());
        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, result.getData());
        int activationID = activationFunction.getIndex();
        if (activationID == 2) {
            if (tensor.getRank() != 2 || tensor.getShape().toHeapArray()[1] != 1) {
                throw new IllegalArgumentException("Softmax requires a Rank 2 Tensor with exactly one column (N x 1).");
            }
            DoubleArray sumBufferData = SOFTMAX_SUM_BUFFER;
            int totalSize = tensor.getData().getSize();
            sumBufferData.set(0, 0.0);
            taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, sumBufferData);
            taskGraph.task(STR."\{taskID}_EXP", TaskGraphTensorOperation::softmaxExpKernel, tensor.getData(), result.getData(), totalSize);
            taskGraph.task(STR."\{taskID}_SUM", TaskGraphTensorOperation::softmaxSumKernel, result.getData(), sumBufferData, totalSize);
            return taskGraph.task(STR."\{taskID}_DIVIDE", TaskGraphTensorOperation::softmaxDivideKernel, result.getData(), sumBufferData, result.getData(), totalSize);
        } else {
            return taskGraph.task(taskID, TaskGraphTensorOperation::getAppliedTensorToActivationFunction, tensor.getData(), result.getData(), activationID, tensor.getData().getSize());
        }
    }
}