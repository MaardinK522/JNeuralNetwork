package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import com.mkproductions.jnn.networks.JSequential;

public class DenseLayer extends Layer {
    private final int numberOfNodes;

    public DenseLayer(int numberOfNodes, ActivationFunction activation) {
        super("Dense", activation);
        this.numberOfNodes = numberOfNodes;
    }

    @Override
    public String toString() {
        return STR."DenseLayer{numberOfNodes=\{numberOfNodes}, activationFunction=\{getActivationFunction()}} \n\{this.getWeights()}";
    }

    public int getNumberOfNodes() {
        return this.numberOfNodes;
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = Tensor.add(Tensor.matrixMultiplication(this.getWeights(), input), this.getBias());
        return JSequential.getActivatedTensors(output, this.getActivationFunction());
    }

    @Override
    public Tensor[] backward(Tensor input, Tensor gradients) {
        Tensor output = this.forward(input);
        Tensor deltaBiases;
        if (this.getActivationFunction().equals(ActivationFunction.SOFTMAX)) {
            deltaBiases = gradients.copy();
        } else {
            Tensor actionDerivative = JSequential.getDeactivatedTensor(output, this.getActivationFunction());
            deltaBiases = Tensor.elementWiseMultiplication(actionDerivative, gradients);
        }
        Tensor deltaWeights = Tensor.matrixMultiplication(deltaBiases, Tensor.transpose(input));
        Tensor gradient = Tensor.matrixMultiplication(Tensor.transpose(this.getWeights()), deltaBiases);
        return new Tensor[]{deltaWeights, deltaBiases, gradient};
    }
}