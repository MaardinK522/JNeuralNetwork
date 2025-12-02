package com.mkproductions.jnn.cpu.layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;
import org.jetbrains.annotations.NotNull;

import java.security.SecureRandom;
import java.util.Random;
import java.util.function.Function;

/**
 * Represents an abstract neural network layer.
 * A layer consists of weights, a bias, an activation function, and a name. It defines the structure and behavior
 * that child classes must implement for performing forward and backward propagation. Layers can be customized
 * to represent different types of neural network operations, such as convolution, pooling, dense, etc.
 */
public abstract class Layer {
    /**
     * The activation function applied to the output of the layer.
     * It determines how the weighted sum of inputs (plus bias) is transformed before being passed to the next layer.
     * This is typically used to introduce non-linearity into the network, allowing it to model complex relationships in the data.
     * Common activation functions include ReLU, sigmoid, and tanh.
     * This variable is immutable and must be defined during the layer's initialization.
     */
    private final ActivationFunction activationFunction;

    /**
     * The designated name of the layer.
     * This variable serves as an identifier for the specific type of layer instance, such as "Dense" or "Flatten".
     * It is immutable and must be defined during the initialization of the layer.
     * The name helps in identifying the layer within the model architecture and distinguishing it when debugging or
     * analyzing the structure of a neural network.
     */
    private final String name;
    /**
     * Represents the weights of the layer.
     * The weights are trainable parameters that determine the strength of the connections between
     * neurons or units in the current layer and the previous layer.
     * They are initialized at the creation of the layer and updated during the training process
     * through optimization algorithms such as gradient descent.
     * The weights play a critical role in learning and adapting a neural network model to the provided data.
     */
    private Tensor weights;
    /**
     * Represents the bias tensor for the layer.
     * <p>
     * The bias tensor is an essential part in the computation performed by
     * the layer. It is added to the weighted input to introduce a trainable offset,
     * which enhances the flexibility and capacity of the model.
     * <p>
     * This tensor is typically updated during the training process based on the
     * optimization algorithm applied using the gradient of the loss function.
     */
    private Tensor bias;

    /**
     * Represents the tensor used to store velocity values for weight updates in
     * optimization algorithms that utilize momentum-based techniques.
     * <p>
     * This tensor is primarily used during the training phase to help accelerate
     * gradient descent and improve convergence by incorporating a fraction of
     * the previous update's direction and magnitude.
     */
    private Tensor velocityWeightsTensors;
    /**
     * Represents the tensor associated with the velocity term for the biases of this layer.
     * This tensor is primarily used in optimization algorithms, such as momentum-based
     * methods, to store the accumulated velocity or momentum term specifically for biases.
     * It helps in improving the efficiency of gradient-based learning by maintaining a
     * moving average of past gradients related to the biases.
     */
    private Tensor velocityBiasesTensors;
    /**
     * Represents the momentum tensor associated with the weights of the layer.
     * This tensor is used to store momentum values during optimization processes, such as
     * momentum-based variants of gradient descent (e.g., SGD with Momentum).
     */
    private Tensor momentumWeightsTensors;
    /**
     * Represents the momentum tensor associated with the biases of the layer.
     * This tensor is used during optimization processes like momentum-based gradient descent
     * to improve the convergence speed and stability of the training process.
     */
    private Tensor momentumBiasesTensors;
    /**
     * Represents the current step count for the Adam optimization algorithm.
     * This variable keeps track of the iteration number, which is used in the
     * computation of bias-corrected first and second moment estimates.
     * <p>
     * This counter is incremented during each training step that involves
     * the Adam optimizer and is critical in the calculation of parameter updates.
     */
    private int adamSteps = 0;
    private final Random random = new SecureRandom();

    /**
     * Constructs a new Layer with the specified name and activation function.
     *
     * @param name       The unique identifier for this layer.
     * @param activation The activation function to be applied to the layer's output.
     */
    public Layer(String name, ActivationFunction activation) {
        this.name = name;
        this.activationFunction = activation;
    }

    /**
     * Retrieves the current weight tensor of the layer.
     *
     * @return The tensor containing the layer's weights.
     */
    public Tensor getWeights() {
        return weights;
    }

    /**
     * Updates the weight tensor for this layer.
     *
     * @param weights The new tensor of weights to assign.
     */
    public void setWeights(Tensor weights) {
        this.weights = weights;
    }

    /**
     * Retrieves the current bias tensor of the layer.
     *
     * @return The tensor containing the layer's biases.
     */
    public Tensor getBias() {
        return bias;
    }

    /**
     * Updates the bias tensor for this layer.
     *
     * @param bias The new tensor of biases to assign.
     */
    public void setBias(Tensor bias) {
        this.bias = bias;
    }

    /**
     * Retrieves the name of the layer.
     *
     * @return A string representing the layer's name.
     */
    public String getName() {
        return this.name;
    }

    /**
     * Retrieves the activation function associated with this layer.
     *
     * @return The activation function used by the layer.
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    /**
     * Performs the forward propagation step for this layer.
     * This method takes an input tensor, applies the layer's operations (like dot product with weights and addition of bias),
     * and passes the result through the activation function.
     *
     * @param input The input tensor provided to this layer.
     * @return The output tensor resulting from the layer's computation.
     */
    public abstract Tensor forward(Tensor input);

    /**
     * Performs the backward propagation step for this layer.
     * This method calculates the gradients of the loss with respect to the input and the layer's parameters
     * (weights and biases), which are used for updating the model during training.
     *
     * @param input     The input tensor that was used during the corresponding forward pass.
     * @param gradients The gradient of the loss with respect to the output of this layer (propagated from the next layer).
     * @return An array of tensors representing the gradients to be propagated back to the previous layer.
     */
    public abstract Tensor[] backward(Tensor input, Tensor gradients);

    /**
     * Performs the backpropagation step for the layer using the Stochastic Gradient Descent (SGD) algorithm.
     * Updates the weights and biases of the layer based on the provided gradients and learning rate.
     *
     * @param learningRate The rate at which the model learns; used to scale the gradient updates.
     * @param deltaWeights The tensor containing the gradient of the loss with respect to the weights.
     * @param deltaBiases  The tensor containing the gradient of the loss with respect to the biases.
     */

    public void backPropagationSGD(double learningRate, Tensor deltaWeights, Tensor deltaBiases) {
        deltaWeights = Tensor.scalarMultiply(deltaWeights, -learningRate);
        deltaBiases = Tensor.scalarMultiply(deltaBiases, -learningRate);

        this.weights.subtract(deltaWeights);
        this.bias.subtract(deltaBiases);
    }

    /**
     * Performs the backpropagation step for the layer using the Stochastic Gradient Descent (SGD)
     * algorithm with momentum. This method updates the weights and biases of the layer based on
     * the provided gradients, learning rate, and momentum value to optimize parameter updates and
     * mitigate oscillations.
     *
     * @param learningRate The rate at which the model learns; used to scale the gradient updates.
     * @param deltaWeights The tensor containing the gradient of the loss with respect to the weights.
     * @param deltaBiases  The tensor containing the gradient of the loss with respect to the biases.
     * @param momentum     The momentum factor used to influence the gradient descent direction, helping
     *                     to speed up convergence and reduce oscillations.
     */
    public void backPropagationSGDWithMomentum(double learningRate, Tensor deltaWeights, Tensor deltaBiases, double momentum) {
        deltaWeights = Tensor.scalarMultiply(deltaWeights, -learningRate);
        deltaBiases = Tensor.scalarMultiply(deltaBiases, -learningRate);

        this.velocityWeightsTensors = Tensor.add(Tensor.scalarMultiply(this.velocityWeightsTensors, momentum), Tensor.scalarMultiply(deltaWeights, 1 - momentum));
        this.velocityBiasesTensors = Tensor.add(Tensor.scalarMultiply(this.velocityBiasesTensors, momentum), Tensor.scalarMultiply(deltaBiases, 1 - momentum));

        this.weights.subtract(this.velocityWeightsTensors);
        this.bias.subtract(this.velocityBiasesTensors);
    }

    /**
     * Performs the backpropagation step for the layer using the Adaptive Gradient (AdaGrad) optimization algorithm.
     * This method adjusts the weights and biases of the layer by updating parameter velocities and applying AdaGrad updates
     * with the specified learning rate and epsilon for numerical stability.
     *
     * @param learningRate The rate at which the model learns, used to scale the gradient updates.
     * @param deltaWeights The tensor containing the gradient of the loss with respect to the weights.
     * @param deltaBiases  The tensor containing the gradient of the loss with respect to the biases.
     * @param epsilon      A small constant added to the denominator to improve numerical stability and avoid division by zero.
     */
    public void backPropagationAdaGrad(double learningRate, Tensor deltaWeights, Tensor deltaBiases, double epsilon) {
        deltaWeights = Tensor.scalarMultiply(deltaWeights, -learningRate);
        deltaBiases = Tensor.scalarMultiply(deltaBiases, -learningRate);

        Tensor squaredDeltaWeightsTensor = Tensor.elementWiseMultiplication(deltaWeights, deltaWeights);
        Tensor squaredDeltaBiasTensor = Tensor.elementWiseMultiplication(deltaBiases, deltaBiases);

        this.velocityWeightsTensors.add(squaredDeltaWeightsTensor);
        this.velocityBiasesTensors.add(squaredDeltaBiasTensor);

        Tensor currentVelocityWeightsTensor = this.velocityWeightsTensors.copy();
        Tensor currentVelocityBiasesTensor = this.velocityBiasesTensors.copy();

        Tensor rootWithVelocityWeightsTensor = Tensor.tensorMapping(deltaWeights, (flatIndex, deltaWeight) -> learningRate * deltaWeight / (Math.sqrt(currentVelocityWeightsTensor.getData()[flatIndex]) + epsilon));
        Tensor rootWithVelocityBiasesTensor = Tensor.tensorMapping(deltaBiases, (flatIndex, deltaBias) -> learningRate * deltaBias / (Math.sqrt(currentVelocityBiasesTensor.getData()[flatIndex]) + epsilon));

        this.weights.subtract(rootWithVelocityWeightsTensor);
        this.bias.subtract(rootWithVelocityBiasesTensor);
    }

    /**
     * Performs the backpropagation step for the layer using the Root Mean Square Propagation (RMSProp)
     * optimization technique. This method adjusts the weights and biases of the layer by updating the
     * parameter velocities and applying the RMSProp updates with the specified learning rate, momentum,
     * and epsilon for stabilization.
     *
     * @param learningRate   The rate at which the model learns, used to scale the gradient updates.
     * @param deltaWeights   The tensor containing the gradient of the loss with respect to the weights.
     * @param deltaBiases    The tensor containing the gradient of the loss with respect to the biases.
     * @param momentum       The momentum factor used to smooth the gradient descent direction by
     *                       incorporating past gradients, helping to accelerate convergence and reduce oscillations.
     * @param epsilonRMSProp A small constant added to the denominator to improve numerical stability
     *                       and avoid division by zero during RMSProp updates.
     */
    public void backPropagationRMSPropagation(double learningRate, Tensor deltaWeights, Tensor deltaBiases, double momentum, double epsilonRMSProp) {
        deltaWeights = Tensor.scalarMultiply(deltaWeights, -learningRate);
        deltaBiases = Tensor.scalarMultiply(deltaBiases, -learningRate);

        Tensor squaredDeltaWeightsTensor = Tensor.elementWiseMultiplication(deltaWeights, deltaWeights);
        Tensor squaredDeltaBiasesTensor = Tensor.elementWiseMultiplication(deltaBiases, deltaBiases);

        this.velocityWeightsTensors = Tensor.add(Tensor.scalarMultiply(this.velocityWeightsTensors, momentum), Tensor.scalarMultiply(squaredDeltaWeightsTensor, 1 - momentum));
        this.velocityBiasesTensors = Tensor.add(Tensor.scalarMultiply(this.velocityBiasesTensors, momentum), Tensor.scalarMultiply(squaredDeltaBiasesTensor, 1 - momentum));

        Tensor velocityWeightsRootTenor = Tensor.tensorMapping(deltaWeights, (flatIndex, value) -> learningRate * value / (Math.sqrt(this.velocityWeightsTensors.getData()[flatIndex]) + epsilonRMSProp));
        Tensor velocityBiasesRootTenor = Tensor.tensorMapping(deltaBiases, (flatIndex, value) -> learningRate * value / (Math.sqrt(this.velocityBiasesTensors.getData()[flatIndex]) + epsilonRMSProp));

        this.weights.subtract(velocityWeightsRootTenor);
        this.bias.subtract(velocityBiasesRootTenor);
    }

    /**
     * Performs backpropagation using the Adam optimization algorithm to update the weights and biases
     * of the neural network. This method modifies the weights and biases by applying calculated
     * gradients, momentum, and velocity adjustments based on the Adam optimizer formula.
     *
     * @param learningRate The learning rate that controls how far the weights and biases are adjusted
     *                     during each optimization step.
     * @param deltaWeights The gradient of the loss function with respect to the weights, used to update
     *                     the weight parameters.
     * @param deltaBiases  The gradient of the loss function with respect to the biases, used to update
     *                     the bias parameters.
     * @param beta1        The exponential decay rate for the first moment estimates (momentum term).
     * @param beta2        The exponential decay rate for the second moment estimates (velocity term).
     * @param epsilon      A small constant added to the denominator for numerical stability during
     *                     division.
     */
    public void backPropagationAdam(double learningRate, Tensor deltaWeights, Tensor deltaBiases, double beta1, double beta2, double epsilon) {
        this.adamSteps++;
        final double beta_1_t = Math.pow(beta1, this.adamSteps);
        final double beta_2_t = Math.pow(beta2, this.adamSteps);

        deltaWeights = Tensor.scalarMultiply(deltaWeights, -learningRate);
        deltaBiases = Tensor.scalarMultiply(deltaBiases, -learningRate);

        this.momentumWeightsTensors = Tensor.add(Tensor.scalarMultiply(this.momentumWeightsTensors, beta1), Tensor.scalarMultiply(deltaWeights, 1 - beta1));
        this.momentumBiasesTensors = Tensor.add(Tensor.scalarMultiply(this.momentumBiasesTensors, beta1), Tensor.scalarMultiply(deltaBiases, 1 - beta1));

        Tensor squaredWeightsGradients = Tensor.elementWiseMultiplication(deltaWeights, deltaWeights);
        Tensor squaredBiasesGradients = Tensor.elementWiseMultiplication(deltaBiases, deltaBiases);

        this.velocityBiasesTensors = Tensor.add(Tensor.scalarMultiply(this.velocityBiasesTensors, beta2), Tensor.scalarMultiply(squaredBiasesGradients, 1 - beta2));
        this.velocityWeightsTensors = Tensor.add(Tensor.scalarMultiply(this.velocityWeightsTensors, beta2), Tensor.scalarMultiply(squaredWeightsGradients, 1 - beta2));

        Tensor momentumWeightsHat = Tensor.tensorMapping(this.momentumWeightsTensors, (_, mW) -> mW / (1 - beta_1_t));
        Tensor momentumBiasesHat = Tensor.tensorMapping(this.momentumBiasesTensors, (_, mB) -> mB / (1 - beta_1_t));

        Tensor velocityWeightsHat = Tensor.tensorMapping(this.velocityWeightsTensors, (_, vW) -> vW / (1 - beta_2_t));
        Tensor velocityBiasesHat = Tensor.tensorMapping(this.velocityBiasesTensors, (_, vB) -> vB / (1 - beta_2_t));

        Tensor rootWithVelocityWeightsTensor = Tensor.tensorMapping(momentumWeightsHat, (flatIndex, mW_Hat) -> learningRate * mW_Hat / (Math.sqrt(velocityWeightsHat.getData()[flatIndex]) + epsilon));
        Tensor rootWithVelocityBiasesTensor = Tensor.tensorMapping(momentumBiasesHat, (flatIndex, mN_hat) -> learningRate * mN_hat / (Math.sqrt(velocityBiasesHat.getData()[flatIndex]) + epsilon));

        this.weights.subtract(rootWithVelocityWeightsTensor);
        this.bias.subtract(rootWithVelocityBiasesTensor);
    }

    public void initLayerParameters() {
        // Randomizing weights and bias.
        randomize(weights);
        randomize(bias);

        // Initializing velocity and momentum for propagation.
        velocityWeightsTensors = weights.copyShape();
        velocityBiasesTensors = bias.copyShape();
        momentumWeightsTensors = weights.copyShape();
        momentumBiasesTensors = bias.copyShape();
        adamSteps = 0;
    }

    private void randomize(@NotNull Tensor tensor) {
        double stdDev = Math.sqrt(2.0 / tensor.getData().length);
        for (int a = 0; a < tensor.getData().length; a++) {
            double u1 = random.nextDouble();
            double u2 = random.nextDouble();
            double gaussian = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            tensor.getData()[a] = gaussian * stdDev;
        }
    }
}