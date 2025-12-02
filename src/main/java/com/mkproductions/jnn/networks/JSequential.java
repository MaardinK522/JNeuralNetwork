package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.LossFunctionAble;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.layers.ConvolutionLayer;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.cpu.layers.FlattenLayer;
import com.mkproductions.jnn.cpu.layers.Layer;
import com.mkproductions.jnn.cpu.layers.PoolingLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.jetbrains.annotations.NotNull;

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
        if (this.networkLayers[this.networkLayers.length - 1].getBias().getData().length <= 1 && this.networkLayers[this.networkLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            throw new IllegalArgumentException("When applied softmax, the last layer must have at least 2 nodes.");
        } else if ((this.lossFunction == LossFunction.SPARSE_CATEGORICAL_CROSS_ENTROPY || this.lossFunction == LossFunction.CATEGORICAL_CROSS_ENTROPY) && !this.networkLayers[this.networkLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            throw new IllegalArgumentException("SCCE and CCE loss functions requires a softmax or sigmoid activation function at the end of the network.");
        }
    }

    public Tensor[] forwardPropagation(Tensor inputTensor) {
        Tensor[] outputTensors = new Tensor[this.networkLayers.length];
        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
            inputTensor = outputTensors[layerIndex] = this.networkLayers[layerIndex].forward(layerIndex == 0 ? inputTensor : outputTensors[layerIndex - 1]);
        }
        return outputTensors;
    }

    //    public void forwardPropagation(Tensor input) {
//        if (input.getRank() != 3) {
//            throw new IllegalArgumentException("Input must be 3D tensor");
//        } else if (this.inputDepth != input.getShape()[0] || this.inputHeight != input.getShape()[1] || this.inputWidth != input.getShape()[2]) {
//            throw new IllegalArgumentException("Input dimensions do not match network architecture");
//        }
//
//        Tensor currentInput = input.copy();
//        for (int layerIndex = 0; layerIndex < this.networkLayers.length; layerIndex++) {
//            if (this.networkLayers[layerIndex] instanceof ConvolutionLayer convolutionLayer) {
//                Tensor kernel = this.kernelTensors[layerIndex];
//
//                int outputHeight = (currentInput.getShape()[1] - convolutionLayer.getFilterSize() + 2 * convolutionLayer.getPadding()) / convolutionLayer.getStride() + 1;
//                int outputWidth = (currentInput.getShape()[2] - convolutionLayer.getFilterSize() + 2 * convolutionLayer.getPadding()) / convolutionLayer.getStride() + 1;
//
//                if (outputHeight < 1 || outputWidth < 1) {
//                    throw new IllegalArgumentException("Convolution resulted in zero or negative dimensions.");
//                }
//                Tensor output = new Tensor(convolutionLayer.getNumberOfFilters(), outputHeight, outputWidth);
//                for (int filterNumber = 0; filterNumber < convolutionLayer.getNumberOfFilters(); filterNumber++) {
//                    Tensor filter = kernel.slice(0, filterNumber, filterNumber + 1).reshape( // Reshaping
//                            kernel.getShape()[1], // Depth,
//                            kernel.getShape()[2], // Rows,
//                            kernel.getShape()[3]  // Columns
//                    );
//                    // Accumulator for each convolution.
//                    Tensor accumulated = null;
//                    for (int d = 0; d < currentInput.getShape()[0]; d++) {
//                        Tensor inputSlice = currentInput.slice(0, d, d + 1).reshape(currentInput.getShape()[1], currentInput.getShape()[2]);
//                        Tensor filterSlice = filter.slice(0, d, d + 1).reshape(convolutionLayer.getFilterSize(), convolutionLayer.getFilterSize());
//                        Tensor conv2D = Tensor.convolve2D(inputSlice, filterSlice, convolutionLayer.getStride(), convolutionLayer.getPadding());
//                        if (accumulated == null) {
//                            accumulated = conv2D.copy();
//                        } else {
//                            accumulated.add(conv2D);
//                        }
//                    }
//                    assert accumulated != null;
//                    for (int y = 0; y < outputHeight; y++) {
//                        for (int x = 0; x < outputWidth; x++) {
//                            output.setEntry(accumulated.getEntry(y, x), filterNumber, y, x);
//                        }
//                    }
//                }
//                addBiasToConvOutput(output, this.biasTensors[layerIndex], this.outputTensors[layerIndex]);
//                this.rawOutputTensors[layerIndex] = output;
//                this.outputTensors[layerIndex].mapTensor(this.networkLayers[layerIndex].getActivationFunction().getEquation());
//                currentInput = this.outputTensors[layerIndex];
//            } else if (this.networkLayers[layerIndex] instanceof PoolingLayer poolingLayer) {
//                if (poolingLayer.getPoolingLayerType() == PoolingLayer.PoolingLayerType.MAX) {
//                    this.outputTensors[layerIndex] = maxPool(currentInput, poolingLayer.getPoolSize(), poolingLayer.getStride());
//                } else if (poolingLayer.getPoolingLayerType() == PoolingLayer.PoolingLayerType.AVG) {
//                    this.outputTensors[layerIndex] = averagePool(currentInput, poolingLayer.getPoolSize(), poolingLayer.getStride());
//                }
//                currentInput = this.outputTensors[layerIndex];
//            } else if (this.networkLayers[layerIndex] instanceof FlattenLayer) {
//                this.outputTensors[layerIndex] = currentInput.reshape(currentInput.getData().length);
//                currentInput = this.outputTensors[layerIndex];
//            } else if (this.networkLayers[layerIndex] instanceof DenseLayer denseLayer) {
//                int inputNode = currentInput.getShape()[0];
//                this.outputTensors[layerIndex] = Tensor.matrixMultiplication(this.outputTensors[layerIndex - 1].reshape(1, inputNode), this.kernelTensors[layerIndex]);
//                this.outputTensors[layerIndex] = Tensor.add(this.outputTensors[layerIndex], this.biasTensors[layerIndex].reshape(1, this.biasTensors[layerIndex].getShape()[1]));
//                this.rawOutputTensors[layerIndex] = this.outputTensors[layerIndex];
//                this.outputTensors[layerIndex] = getAppliedActivationTensors(this.outputTensors[layerIndex], denseLayer.getActivationFunction());
//                currentInput = this.outputTensors[layerIndex];
//            }
//            System.out.println(
//                    STR."\{this.networkLayers[layerIndex].getName()} Layer index: \{layerIndex},\n\tOutput shape: \{this.outputTensors[layerIndex]},\n\t Layer name:\{this.networkLayers[layerIndex].getActivationFunction()
//                            .name()}");
//        }
//        //        return this.outputTensors[this.networkLayers.length - 1];
//    }
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
                    Tensor outputLoss = this.lossFunction.getDerivativeTensor(finalOutputTensor, targetTensor);
                    Tensor outputdDerivativeTensor = getDeactivatedTensor(outputTensors[layerIndex], layer.getActivationFunction());
                    currentGradient = Tensor.elementWiseMultiplication(outputLoss, outputdDerivativeTensor);
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
        } else if (trainingOutputs[0].getShape()[0] != this.networkLayers[this.networkLayers.length - 1].getBias().getData().length) {
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
        if (tensor == null || tensor.getData().length == 0) {
            return -1;
        }
        double maxVal = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for (int i = 0; i < tensor.getData().length; i++) {
            if (tensor.getData()[i] > maxVal) {
                maxVal = tensor.getData()[i];
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

    public static Tensor getActivatedTensors(Tensor tensor, @NotNull ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (tensor.getShape()[1] != 1) {
                throw new IllegalArgumentException("Unable to apply softmax due to more column existing in the given Tensor.");
            }
            Tensor eRasiedTensor = Tensor.tensorMapping(tensor, (_, value) -> Math.exp(value));
            double sum = IntStream.range(0, eRasiedTensor.getShape()[0]).mapToDouble(a -> eRasiedTensor.getEntry(a, 0)).sum();
            return Tensor.tensorMapping(eRasiedTensor, ((flatIndex, value) -> value / sum));
        }
        return Tensor.tensorMapping(tensor, activationFunction.getEquation());
    }

    public static Tensor getDeactivatedTensor(Tensor activatedTensor, @NotNull ActivationFunction activationFunction) {
        if (activationFunction.name().equals(ActivationFunction.SOFTMAX.name())) {
            if (activatedTensor.getShape()[1] != 1) {
                throw new IllegalArgumentException("Softmax derivative expects a single vector output.");
            }
            int n = Math.max(activatedTensor.getShape()[0], activatedTensor.getShape()[1]);
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
        for (int i = 0; i < tensor.getData().length; i++) {
            double value = -1 + (random.nextDouble() * 2);
            tensor.getData()[i] = value;
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