package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.networks.JDenseSequential;
import com.mkproductions.jnn.networks.JNeuralNetwork;

import java.util.stream.IntStream;


public class DenseLayer extends Layer {
    private final int numberOfNodes;

    public DenseLayer(int numberOfNodes, ActivationFunction activation) {
        super("Dense", activation);
        this.numberOfNodes = numberOfNodes;
    }

    @Override
    public String toString() {
        return STR."Layer{numberOfNodes=\{numberOfNodes}, activationFunction=\{getActivationFunction()}} \n\{this.getWeights()}";
    }

    public int getNumberOfNodes() {
        return this.numberOfNodes;
    }

//    private boolean isBackwardCalled = false;

    @Override
    public Tensor forward(Tensor input) {
//        if (isBackwardCalled) {
//            System.out.println("Forward propagation for DenseLayer started");
//            System.out.println(STR."Inputs tensor: \{input}");
//            System.out.println(STR."Layer weights: \{this.getWeights()}");
//        }
        Tensor output = Tensor.add(Tensor.matrixMultiplication(this.getWeights(), input), this.getBias());
//        if (isBackwardCalled) {
//            System.out.println(STR."Layer outputs: \{output}");
//            isBackwardCalled = false;
//        }
        return JDenseSequential.getAppliedActivationFunctionTensor(output, this.getActivationFunction());
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
//        if (!this.isBackwardCalled) {
//            System.out.println("Started backward propagation for DenseLayer:");
//            this.isBackwardCalled = true;
//        }
        Tensor output = this.forward(input); //Tensor.add(Tensor.matrixMultiplication(this.getWeights(), input), this.getBias());
        Tensor deltaBiases;
        if (this.getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            deltaBiases = gradients.copy();
        } else {
            Tensor actionDerivative = JDenseSequential.getDeactivatedActivationFunctionTensor(output, this.getActivationFunction());
            deltaBiases = Tensor.elementWiseMultiplication(actionDerivative, gradients);
        }
        Tensor deltaWeights = Tensor.matrixMultiplication(deltaBiases, Tensor.transpose(input));
        Tensor gradient = Tensor.matrixMultiplication(Tensor.transpose(this.getWeights()), deltaBiases);
        return new Tensor[]{deltaWeights, deltaBiases, gradient};
    }
}