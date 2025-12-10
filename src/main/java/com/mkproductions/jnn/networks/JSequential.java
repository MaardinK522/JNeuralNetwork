package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.layers.ConvolutionLayer;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.layers.FlattenLayer;
import com.mkproductions.jnn.cpu.layers.Layer;
import com.mkproductions.jnn.cpu.layers.PoolingLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;

import java.security.SecureRandom;
import java.util.Random;
import java.util.stream.IntStream;


public class JSequential {
    private final Layer[] networkLayers;
    private static final Random random = new SecureRandom();
    private LossFunction lossFunction;
    private JNetworkOptimizer optimizer;
    private double learningRate;
    private double momentumFactorBeta1;
    private double momentumFactorBeta2;
    private double epsilon = 1e-8;
    private boolean debugMode = false;

    public JSequential(int[] inputShape, LossFunction lossFunction, JNetworkOptimizer optimizer, Layer... networkLayers) {
        this.networkLayers = networkLayers;
        int inputDepth;
        int inputWidth;
        int inputHeight;
        if (inputShape.length > 0) {
            if (inputShape.length == 1) {
                inputDepth = 1;
                inputHeight = 1;
                inputWidth = inputShape[0];
            } else if (inputShape.length == 3) {
                inputDepth = inputShape[0];
                inputHeight = inputShape[1];
                inputWidth = inputShape[2];
            } else {
                throw new IllegalArgumentException("Input shape must be length 1 (Dense) or 3 (Conv).");
            }
        } else {
            throw new IllegalArgumentException("Input shape must contain at least one dimension.");
        }
        this.learningRate = 0.01;
        this.momentumFactorBeta1 = 0.9;
        momentumFactorBeta2 = 0.999;
        int currentInputDepth = inputDepth;
        int currentInputWidth = inputWidth;
        int currentInputHeight = inputHeight;
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            Layer layer = this.networkLayers[layerIndex];
            switch (layer) {
                case ConvolutionLayer convolutionLayer -> {
                    int outputHeight = (currentInputHeight + 2 * convolutionLayer.getPadding() - convolutionLayer.getFilterSize()) / convolutionLayer.getStride() + 1;
                    int outputWidth = (currentInputWidth + 2 * convolutionLayer.getPadding() - convolutionLayer.getFilterSize()) / convolutionLayer.getStride() + 1;

                    if (outputWidth < 1 || outputHeight < 1) {
                        throw new IllegalArgumentException(STR."Convolution at index \{layerIndex} resulted in zero or negative dimensions.");
                    }

                    convolutionLayer.setWeights(new Tensor(convolutionLayer.getNumberOfFilters(), currentInputDepth, convolutionLayer.getFilterSize(), convolutionLayer.getFilterSize()));
                    convolutionLayer.setBias(new Tensor(convolutionLayer.getNumberOfFilters()));

                    currentInputDepth = convolutionLayer.getNumberOfFilters();
                    currentInputWidth = outputWidth;
                    currentInputHeight = outputHeight;
                }

                case PoolingLayer poolingLayer -> {
                    int outputHeight = (currentInputHeight - poolingLayer.getPoolSize()) / poolingLayer.getStride() + 1;
                    int outputWidth = (currentInputWidth - poolingLayer.getPoolSize()) / poolingLayer.getStride() + 1;

                    if (outputWidth < 1 || outputHeight < 1) {
                        throw new IllegalArgumentException(STR."Pooling at index \{layerIndex} resulted in zero or negative dimensions.");
                    }

                    currentInputWidth = outputWidth;
                    currentInputHeight = outputHeight;
                }

                case FlattenLayer _ -> {
                    if (currentInputHeight == 1 && currentInputDepth == 1) {
                        throw new IllegalArgumentException(STR."Redundant FlattenLayer at index \{layerIndex}. Input is already flat (Dense/1D).");
                    }
                    currentInputWidth = currentInputWidth * currentInputHeight * currentInputDepth;
                    currentInputHeight = 1;
                    currentInputDepth = 1;
                }

                case DenseLayer denseLayer -> {
                    // LOGIC FIX:
                    // Since we initialized Height/Depth to 1 for the 'int[]{2}' case,
                    // currentInputHeight is 1, currentInputDepth is 1.
                    // This check (1 > 1 || 1 > 1) returns FALSE, so no exception is thrown.
                    if (currentInputDepth > 1 || currentInputHeight > 1) {
                        throw new IllegalArgumentException(STR."DenseLayer at index \{layerIndex} can only accept 1D input. A FlattenLayer is required after Convolution/Pooling.");
                    }

                    // Weights: (Output, Input). currentInputWidth holds the node count.
                    denseLayer.setWeights(new Tensor(denseLayer.getNumberOfNodes(), currentInputWidth));
                    denseLayer.setBias(new Tensor(denseLayer.getNumberOfNodes(), 1));

                    currentInputWidth = denseLayer.getNumberOfNodes();
                    currentInputHeight = 1;
                    currentInputDepth = 1;
                }

                case null, default -> {
                    System.err.println(this.networkLayers[layerIndex]);
                    throw new IllegalArgumentException("Unsupported layer type detected in network architecture.");
                }
            }

            if (!(layer instanceof FlattenLayer) && !(layer instanceof PoolingLayer)) {
                layer.initLayerParameters();
            }
        }
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        if (this.networkLayers.length == 0) {
            throw new IllegalArgumentException("Network must contain at least one layer.");
        }
        if (this.networkLayers[this.networkLayers.length - 1].getBias().getData().getSize() <= 1 && this.networkLayers[this.networkLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            throw new IllegalArgumentException("When applied softmax, the last layer must have at least 2 nodes.");
        } else if ((this.lossFunction == LossFunction.SPARSE_CATEGORICAL_CROSS_ENTROPY || this.lossFunction == LossFunction.CATEGORICAL_CROSS_ENTROPY) && !this.networkLayers[this.networkLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            throw new IllegalArgumentException("SCCE and CCE loss functions requires a softmax or sigmoid activation function at the end of the network.");
        }
    }

    public Tensor[] predict(Tensor inputTensor) {
        return forwardPropagation(inputTensor);
    }

    private Tensor[] forwardPropagation(Tensor inputTensor) {
        Tensor[] outputTensors = new Tensor[this.networkLayers.length];
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            inputTensor = outputTensors[layerIndex] = this.networkLayers[layerIndex].forward(layerIndex == 0 ? inputTensor : outputTensors[layerIndex - 1]);
        }
        return outputTensors;
    }

    private void backPropagation(Tensor inputTensor, Tensor targetTensor) {
        Tensor[] outputTensors = this.forwardPropagation(inputTensor);
        Tensor finalOutputTensor = outputTensors[this.networkLayers.length - 1];
        Tensor currentGradient = null;
        for (int layerIndex = this.networkLayers.length - 1; layerIndex >= 0; layerIndex--) {
            Layer layer = this.networkLayers[layerIndex];
            if (layerIndex == this.networkLayers.length - 1) {
                if (((this.lossFunction.equals(LossFunction.CATEGORICAL_CROSS_ENTROPY) || this.lossFunction.equals(LossFunction.SPARSE_CATEGORICAL_CROSS_ENTROPY)) && this.networkLayers[this.networkLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX))) {
                    currentGradient = Tensor.subtract(finalOutputTensor, targetTensor);
                } else {
                    Tensor lossTensor = this.lossFunction.getDerivativeTensor(finalOutputTensor, targetTensor);
                    Tensor deActivatedTensor = getDeactivatedTensor(finalOutputTensor, layer.getActivationFunction());
                    currentGradient = Tensor.elementWiseMultiplication(lossTensor, deActivatedTensor);
                }
            }
            Tensor[] backwardData = layer.backward(layerIndex == 0 ? inputTensor : outputTensors[layerIndex - 1], currentGradient);
            Tensor weightsGradient = backwardData[0];
            Tensor biasesGradient = backwardData[1];
            currentGradient = backwardData[2];
            if (!(layer instanceof PoolingLayer) && !(layer instanceof FlattenLayer)) {
                switch (this.optimizer) {
                    case SGD -> layer.backPropagationSGD(this.learningRate, weightsGradient, biasesGradient);
                    case SGD_MOMENTUM -> layer.backPropagationSGDWithMomentum(this.learningRate, weightsGradient, biasesGradient, this.momentumFactorBeta1);
                    case RMS_PROP -> layer.backPropagationRMSPropagation(this.learningRate, weightsGradient, biasesGradient, this.momentumFactorBeta1, this.epsilon);
                    case ADA_GARD -> layer.backPropagationAdaGrad(this.learningRate, weightsGradient, biasesGradient, this.epsilon);
                    case ADAM -> layer.backPropagationAdam(this.learningRate, weightsGradient, biasesGradient, this.momentumFactorBeta1, this.momentumFactorBeta2, this.epsilon);
                }
            }
        }
    }

    public void train(Tensor[] trainingInputs, Tensor[] trainingOutputs, int epochs) {
        if (trainingInputs.length != trainingOutputs.length) {
            throw new IllegalArgumentException("Training inputs and outputs must be of the same length.");
        } else if (trainingOutputs[0].getShape().toHeapArray()[0] != this.networkLayers[this.networkLayers.length - 1].getBias().getData().getSize()) {
            System.err.println(STR."Input: Tensor: \{trainingOutputs[0]}");
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

    public Tensor processInputs(Tensor inputs) {
        return this.forwardPropagation(inputs)[this.networkLayers.length - 1];
    }

    private int argMax(Tensor tensor) {
        if (tensor == null || tensor.getData().getSize() == 0) {
            return -1;
        }
        double maxVal = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for (int i = 0; i < tensor.getData().getSize(); i++) {
            if (tensor.getData().get(i) > maxVal) {
                maxVal = tensor.getData().get(i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public double calculateAccuracy(Tensor[] inputs, Tensor[] targets) {
        if (inputs.length != targets.length || inputs.length == 0) {
            return 0.0;
        }
        int correctPredictions = 0;
        int totalSamples = inputs.length;
        for (int i = 0; i < totalSamples; i++) {
            Tensor input = inputs[i];
            Tensor target = targets[i];
            Tensor finalOutput = this.processInputs(input);
            int predictedClassIndex = argMax(finalOutput);
            int trueClassIndex = argMax(target);
            if (predictedClassIndex == trueClassIndex) {
                correctPredictions++;
            }
        }
        return (double) correctPredictions / totalSamples;
    }

    public static Tensor getActivatedTensors(Tensor tensor, ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (tensor.getShape().toHeapArray()[1] != 1) {
                throw new IllegalArgumentException("Unable to apply softmax due to more column existing in the given Tensor.");
            }
            Tensor eRasiedTensor = Tensor.tensorMapping(tensor, (_, value) -> Math.exp(value));
            double sum = IntStream.range(0, eRasiedTensor.getShape().toHeapArray()[0]).mapToDouble(a -> eRasiedTensor.getEntry(a, 0)).sum();
            return Tensor.tensorMapping(eRasiedTensor, ((_, value) -> value / sum));
        }
        return Tensor.tensorMapping(tensor, activationFunction.getEquation());
    }

    public static Tensor getDeactivatedTensor(Tensor activatedTensor, ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (activatedTensor.getShape().toHeapArray()[1] != 1) {
                throw new IllegalArgumentException("Softmax derivative expects a single vector output.");
            }
            int n = Math.max(activatedTensor.getShape().toHeapArray()[0], activatedTensor.getShape().toHeapArray()[1]);
            Tensor result = new Tensor(n, n);
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    double entry = a == b ? (activatedTensor.getEntry(a, b) * (1 - activatedTensor.getEntry(a, b))) : -activatedTensor.getEntry(a, b) * activatedTensor.getEntry(b, a);
                    result.setEntry(entry, a, b);
                }
            }
            return result;
        }
        return Tensor.tensorMapping(activatedTensor, activationFunction.getDerivative());
    }

    public static void randomize(Tensor tensor) {
        for (int i = 0; i < tensor.getData().getSize(); i++) {
            double value = -1 + (random.nextDouble() * 2);
            tensor.getData().set(i, value);
        }
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public JSequential setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        return this;
    }

    public JNetworkOptimizer getOptimizer() {
        return optimizer;
    }

    public JSequential setOptimizer(JNetworkOptimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    public double getMomentumFactorBeta1() {
        return momentumFactorBeta1;
    }

    public void setMomentumFactorBeta1(double momentumFactorBeta1) {
        this.momentumFactorBeta1 = momentumFactorBeta1;
    }

    public double getMomentumFactorBeta2() {
        return momentumFactorBeta2;
    }

    public void setMomentumFactorBeta2(double momentumFactorBeta2) {
        this.momentumFactorBeta2 = momentumFactorBeta2;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public boolean isDebugMode() {
        return debugMode;
    }

    public JSequential setDebugMode(boolean debugMode) {
        this.debugMode = debugMode;
        return this;
    }
}