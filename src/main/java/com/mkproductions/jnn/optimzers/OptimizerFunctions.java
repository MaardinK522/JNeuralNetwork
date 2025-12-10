package com.mkproductions.jnn.optimzers;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class OptimizerFunctions {
    public static void applySGD(DoubleArray learningRate, DoubleArray weights, DoubleArray biases, DoubleArray weightsGradient, DoubleArray biasesGradient, int weightsSize, int biasesSize) {
        for (@Parallel int i = 0; i < weightsSize; i++) {
            weightsGradient.set(i, weightsGradient.get(i) * -learningRate.get(0));
            weights.set(i, weights.get(i) - weightsGradient.get(i));
        }
        for (@Parallel int i = 0; i < biasesSize; i++) {
            biasesGradient.set(i, biasesGradient.get(i) * -learningRate.get(0));
            biases.set(i, biases.get(i) - biasesGradient.get(i));
        }
    }

    public static void applySGDMomentum(DoubleArray learningRate, DoubleArray weights, DoubleArray biases, DoubleArray weightsGradient, DoubleArray biasesGradient, DoubleArray velocityWeights, DoubleArray velocityBiases, DoubleArray momentum, int weightsSize, int biasesSize) {
        for (@Parallel int i = 0; i < weightsSize; i++) {
            weightsGradient.set(i, weightsGradient.get(i) * -learningRate.get(0));
            weightsGradient.set(i, weightsGradient.get(i) * (1 - momentum.get(0)));
            velocityWeights.set(i, velocityWeights.get(i) * momentum.get(i));
            velocityWeights.set(i, velocityWeights.get(i) + weightsGradient.get(i));
            weights.set(i, weights.get(i) - velocityWeights.get(i));
        }
        for (@Parallel int i = 0; i < biasesSize; i++) {
            biasesGradient.set(i, biasesGradient.get(i) * -learningRate.get(0));
            biasesGradient.set(i, biasesGradient.get(i) * (1 - momentum.get(0)));
            velocityBiases.set(i, velocityBiases.get(i) * momentum.get(i));
            velocityBiases.set(i, velocityBiases.get(i) + biasesGradient.get(i));
            biases.set(i, biases.get(i) - velocityBiases.get(i));
        }
    }

    public static void applyAdaGrad(DoubleArray learningRate, DoubleArray weights, DoubleArray biases, DoubleArray weightsGradient, DoubleArray biasesGradient, DoubleArray velocityWeights, DoubleArray velocityBiases, DoubleArray epsilon, int weightsSize, int biasesSize) {
        for (@Parallel int i = 0; i < weightsSize; i++) {
            weightsGradient.set(i, weightsGradient.get(i) * -learningRate.get(0));
            double squaredWeightsGradients = TornadoMath.pow(weightsGradient.get(i), 2);
            velocityWeights.set(i, velocityWeights.get(i) + squaredWeightsGradients);
            double rootedVelocityWeights = learningRate.get(0) * weightsGradient.get(i) / (TornadoMath.sqrt(velocityWeights.get(i)) + epsilon.get(0));
            weights.set(i, weights.get(i) - rootedVelocityWeights);
        }
        for (@Parallel int i = 0; i < biasesSize; i++) {
            biasesGradient.set(i, biasesGradient.get(i) * -learningRate.get(0));
            double squaredBiasesGradients = TornadoMath.pow(biasesGradient.get(i), 2);
            velocityBiases.set(i, velocityBiases.get(i) + squaredBiasesGradients);
            double rootedVelocityBiases = learningRate.get(0) * biasesGradient.get(i) / (TornadoMath.sqrt(velocityBiases.get(i)) + epsilon.get(0));
            biases.set(i, biases.get(i) - rootedVelocityBiases);
        }
    }

    public static void applyRMSPropagation(DoubleArray learningRate, DoubleArray weights, DoubleArray biases, DoubleArray weightsGradient, DoubleArray biasesGradient, DoubleArray velocityWeights, DoubleArray velocityBiases, DoubleArray momentumBeta1, DoubleArray epsilon, int weightsSize, int biasesSize) {
        for (@Parallel int i = 0; i < weightsSize; i++) {
            weightsGradient.set(i, weightsGradient.get(i) * -learningRate.get(0));
            double squaredWeightsGradients = TornadoMath.pow(weightsGradient.get(i), 2);
            velocityWeights.set(i, velocityWeights.get(i) * momentumBeta1.get(0) + squaredWeightsGradients * (1 - momentumBeta1.get(0)));
            double rootedVelocityWeights = learningRate.get(0) * weightsGradient.get(i) / (TornadoMath.sqrt(velocityWeights.get(i)) + epsilon.get(0));
            weights.set(i, weights.get(i) - rootedVelocityWeights);
        }
        for (@Parallel int i = 0; i < biasesSize; i++) {
            biasesGradient.set(i, biasesGradient.get(i) * -learningRate.get(0));
            double squaredBiasesGradients = TornadoMath.pow(biasesGradient.get(i), 2);
            velocityBiases.set(i, velocityBiases.get(i) * momentumBeta1.get(0) + squaredBiasesGradients * (1 - momentumBeta1.get(0)));
            double rootedVelocityBiases = learningRate.get(0) * biasesGradient.get(i) / (TornadoMath.sqrt(velocityBiases.get(i)) + epsilon.get(0));
            biases.set(i, biases.get(i) - rootedVelocityBiases);
        }
    }

    public static void applyAdam(DoubleArray learningRate, DoubleArray weights, DoubleArray biases, DoubleArray weightsGradients, DoubleArray biasesGradients, DoubleArray velocityWeights, DoubleArray velocityBiases, DoubleArray momentumWeights, DoubleArray momentumBiases, DoubleArray beta1, DoubleArray beta2, DoubleArray epsilon, IntArray adamSteps, int weightsSize, int biasesSize) {
        double beta1T = TornadoMath.pow(beta1.get(0), adamSteps.get(0));
        double beta2T = TornadoMath.pow(beta2.get(0), adamSteps.get(0));
        for (@Parallel int i = 0; i < weightsSize; i++) {
            weightsGradients.set(i, weightsGradients.get(i) * -learningRate.get(0));
            momentumWeights.set(i, momentumWeights.get(i) * beta1.get(0) + weightsGradients.get(i) * (1 - beta1T));
            double squareWeight = TornadoMath.pow(weightsGradients.get(i), 2);
            velocityWeights.set(i, velocityWeights.get(i) * beta2.get(0) + squareWeight * (1 - beta2T));
            double momentumWeightHat = momentumWeights.get(i) / (1 - beta1T);
            double velocityWeightHat = velocityWeights.get(i) / (1 - beta2T);
            weights.set(i, weights.get(i) - (learningRate.get(0) * momentumWeightHat / (TornadoMath.sqrt(velocityWeightHat) + epsilon.get(0))));
        }
        for (@Parallel int i = 0; i < biasesSize; i++) {
            biasesGradients.set(i, biasesGradients.get(i) * -learningRate.get(0));
            momentumBiases.set(i, momentumBiases.get(i) * beta1.get(0) + biasesGradients.get(i) * (1 - beta1T));
            double squareBias = TornadoMath.pow(biasesGradients.get(i), 2);
            velocityBiases.set(i, velocityBiases.get(i) * beta2.get(0) + squareBias * (1 - beta2T));
            double momentumBiasHat = momentumBiases.get(i) / (1 - beta1T);
            double velocityBiasHat = velocityBiases.get(i) / (1 - beta2T);
            biases.set(i, biases.get(i) - (learningRate.get(0) * momentumBiasHat / (TornadoMath.sqrt(velocityBiasHat) + epsilon.get(0))));
        }
        adamSteps.set(0, adamSteps.get(0) + 1);
    }
}