package com.mkproductions.jnn.networks;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.LossFunctionAble;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.cpu.layers.DenseLayer;
import com.mkproductions.jnn.lossFunctions.LossFunction;
import com.mkproductions.jnn.optimzers.JNetworkOptimizer;
import org.jetbrains.annotations.NotNull;

import java.security.SecureRandom;
import java.util.Random;
import java.util.stream.IntStream;

public class JDenseSequential {
    private final int numberOfInputNode;
    private final DenseLayer[] netWorkDenseLayers;
    private final LossFunctionAble lossFunctionable;
    private final JNetworkOptimizer jNetworkOptimizer;
    private boolean debugMode = false;
    private double learningRate;
    private final double momentumFactorBeta1;
    private final double momentumFactorBeta2;
    private final double epsilon = 1e-8;
    private int epochCount = 0;
    private int adamSteps = 0;
    private final Random random = new SecureRandom();

    public JDenseSequential(LossFunctionAble lossFunctionable, JNetworkOptimizer jNetworkOptimizer, int numberOfInputNode, DenseLayer... netWorkDenseLayers) {
        this.lossFunctionable = lossFunctionable;
        this.jNetworkOptimizer = jNetworkOptimizer;
        this.learningRate = 0.01;
        this.numberOfInputNode = numberOfInputNode;
        this.netWorkDenseLayers = netWorkDenseLayers;
        for (int layerIndex = 0; layerIndex < this.netWorkDenseLayers.length; layerIndex++) {
            DenseLayer layer = this.netWorkDenseLayers[layerIndex];
            layer.setWeights(new Tensor(layer.getNumberOfNodes(), layerIndex == 0 ? this.numberOfInputNode : this.netWorkDenseLayers[layerIndex - 1].getNumberOfNodes()));
            layer.setBias(new Tensor(layer.getNumberOfNodes(), 1));
            layer.initLayerParameters();
            System.out.println(STR."Layer: \{layerIndex}, data: \{layer}");
        }
        this.momentumFactorBeta1 = 0.9;
        this.momentumFactorBeta2 = 0.999;
    }

    private Tensor[] forwardPropagation(Tensor inputTensor) {
        Tensor[] outputTensors = new Tensor[this.netWorkDenseLayers.length];
        for (int layerIndex = 0; layerIndex < this.netWorkDenseLayers.length; layerIndex++) {
//            System.out.println(STR."Layer: \{layerIndex}");
//            System.out.println(STR."Weights: \{this.netWorkDenseLayers[layerIndex].getWeights()}");
//            System.out.println(STR."Input tensor: \{inputTensor}");
            inputTensor = outputTensors[layerIndex] = this.netWorkDenseLayers[layerIndex].forward(inputTensor);
//            System.out.println(STR."Output tensor: \{outputTensors[layerIndex]}");
        }
//        System.out.println();
//        System.out.println();
//        System.out.println();
        return outputTensors;
    }


    private void backPropagation(Tensor inputTensor, Tensor targetTensor) {
        Tensor[] outputTensors = this.forwardPropagation(inputTensor);
        Tensor finalOutputTensor = outputTensors[this.netWorkDenseLayers.length - 1];
        Tensor currentGradient = null;
        for (int layerIndex = this.netWorkDenseLayers.length - 1; layerIndex >= 0; layerIndex--) {
            DenseLayer layer = this.netWorkDenseLayers[layerIndex];
            if (layerIndex == this.netWorkDenseLayers.length - 1) {
                if ((this.lossFunctionable.equals(LossFunction.CATEGORICAL_CROSS_ENTROPY) && this.netWorkDenseLayers[this.netWorkDenseLayers.length - 1].getActivationFunction().equals(ActivationFunction.SOFTMAX))) {
                    currentGradient = Tensor.subtract(finalOutputTensor, targetTensor);
                } else {
                    Tensor outputLoss = this.lossFunctionable.getDerivativeTensor(finalOutputTensor, targetTensor);
                    Tensor outputdDerivativeTensor = JSequential.getDeactivatedTensor(outputTensors[layerIndex], layer.getActivationFunction());
                    currentGradient = Tensor.elementWiseMultiplication(outputLoss, outputdDerivativeTensor);
                }
            }
            Tensor[] backwardData = layer.backward(layerIndex == 0 ? inputTensor : outputTensors[layerIndex - 1], currentGradient);
            Tensor weightsGradient = backwardData[0];
            Tensor biasesGradient = backwardData[1];
            switch (this.jNetworkOptimizer) {
                case SGD -> layer.backPropagationSGD(this.learningRate, weightsGradient, biasesGradient);
                case SGD_MOMENTUM -> layer.backPropagationSGDWithMomentum(this.learningRate, weightsGradient, biasesGradient, this.momentumFactorBeta1);
                case RMS_PROP -> layer.backPropagationRMSPropagation(this.learningRate, weightsGradient, biasesGradient, this.momentumFactorBeta1, this.epsilon);
                case ADA_GARD -> layer.backPropagationAdaGrad(this.learningRate, weightsGradient, biasesGradient, this.epsilon);
                case ADAM -> layer.backPropagationAdam(this.learningRate, weightsGradient, biasesGradient, this.momentumFactorBeta1, this.momentumFactorBeta2, this.epsilon);
            }
            currentGradient = backwardData[2];
        }
    }

    public void train(Tensor[] trainingInputs, Tensor[] trainingOutputs, int epochs) {
        int progress;
        int lastProgress = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Random index for training random data from the training data set.
            progress = (epoch * 100) / epochs;
            int randomIndex = this.random.nextInt(0, trainingInputs.length);
            //            System.out.println("Random index: " + randomIndex);
            Tensor trainingInput = trainingInputs[randomIndex];
            Tensor trainingOutput = trainingOutputs[randomIndex];
            this.backPropagation(trainingInput, trainingOutput);
            if (this.debugMode && progress != lastProgress) {
                lastProgress = progress;
                int filled = progress / 2;
                int empty = 50 - filled;
                String filledBar = "#".repeat(filled);
                String emptyBar = " ".repeat(empty);
                System.out.printf("\rTraining progress: %d%% [%s%s]", progress, filledBar, emptyBar);
                if (progress == 100) {
                    System.out.println();
                }
            }
            epochCount++;
        }
    }


    public Tensor processInputs(Tensor tensor) {
        return forwardPropagation(tensor)[this.netWorkDenseLayers.length - 1];
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}