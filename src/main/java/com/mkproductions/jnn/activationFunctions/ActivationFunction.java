package com.mkproductions.jnn.activationFunctions;

import com.mkproductions.jnn.cpu.entity.TensorMapAbleFunction;

/**
 * Enum ActivationFunction represents various activation functions
 * used in neural network calculations, including their equations
 * and corresponding derivatives for backpropagation.
 * <p>
 * Each activation function has:
 * - A name to identify it
 * - An equation to apply the activation
 * - A derivative function for gradient calculations
 */
public enum ActivationFunction {
    /**
     * The NONE activation function represents a non-operational or placeholder activation.
     * This activation function does not perform any meaningful calculation,
     * as both the equation and its derivative always return a constant value of 0.0.
     * <p>
     * Use this when no activation is required or as a default value.
     * <p>
     * Properties:
     * - Name: "none"
     * - Equation: Always returns 0.0
     * - Derivative: Always returns 0.0
     */
    NONE(((_, _) -> 0.0F), (_, _) -> 0.0F, 0), // NONE
    /**
     * The SIGMOID activation function is widely used in neural networks,
     * particularly in use cases involving binary classification.
     * It maps input values to an output range of [0, 1] following the equation:
     * <p>
     * f(x) = 1 / (1 + exp(-x))
     * <p>
     * where 'x' represents the input value.
     * <p>
     * The derivative of the sigmoid function is represented as:
     * <p>
     * f'(x) = f(x) * (1 - f(x))
     * <p>
     * This derivative is crucial during the backpropagation phase for calculating gradients.
     * <p>
     * Characteristics:
     * - Activation function: "sigmoid"
     * - Output range: [0, 1]
     * - Non-linear transformation
     * <p>
     * Use cases for the SIGMOID activation function include hidden layers in neural networks,
     * output neurons in binary classification models, and probabilistic output scenarios.
     */
    SIGMOID(((_, value) -> 1.0 / (1 + Math.exp(-value))), (_, y) -> y * (1 - y), 1), // Sigmoid activation function with derivative.
    /**
     * Represents an activation function called ReLU (Rectified Linear Unit).
     * <ul>
     * - Activation behavior: Returns the input value if it is greater than or equal to 0; otherwise, returns 0.
     * - Derivative behavior: Returns 1 if input is greater than or equal to 0; otherwise, returns 0.
     * </ul>
     * <p>
     * ReLU is commonly used in neural networks to introduce non-linearity, while maintaining computational efficiency.
     */
    RE_LU((_, x) -> Math.max(0, x), (_, y) -> (y < 0) ? 0 : 1, 2), // Rectified Linear Unit activation function with derivative.
    /**
     * Represents the linear activation function.
     * <p>
     * This activation function applies a linear transformation to its input, preserving the input value as the output:
     * Output = Input. Additionally, the derivative of the linear activation function is constant and equals 1.
     * <p>
     * Key properties:
     * - Function Name: "linear"
     * - Equation: Input -> Input
     * - Derivative: Constant value of 1
     * <p>
     * Used for scenarios where no non-linearity is required, or a simple identity mapping is sufficient.
     */
    LINEAR((_, x) -> x, (_, _) -> 1, 3), // Linear activation function with derivative.
    /**
     * Represents the hyperbolic tangent (tanh) activation function as an enumeration constant.
     * The tanh function is defined as:
     * tanh(x) = (e^x - e^-x) / (e^x + e^-x)
     * where e is the base of natural logarithms.
     * <p>
     * This activation function maps input values onto a range between -1 and 1.
     * It is often used in neural networks to introduce non-linearity, enabling the model
     * to learn complex patterns. It is also centered at zero, which can help with vanishing
     * gradient issues in deeper networks.
     * <p>
     * The derivative of the tanh function is given as:
     * tanh'(x) = 1 - tanh(x)^2
     * <p>
     * This enum constant defines the tanh function along with its derivative.
     * <p>
     * Parameters:
     * - activationFunctionName: A string name identifier for the tanh activation function.
     * - equation: A TensorMapAbleFunction instance implementing the tanh mathematical function.
     * - derivative: A TensorMapAbleFunction instance computing the derivative of the tanh function.
     */
    TAN_H((_, x) -> (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)), (_, y) -> 1 - (y * y), 4), // Hyper tangent activation function with derivative.
    /**
     * The SOFTMAX activation function is a commonly used activation function for neural network output layers
     * in classification problems. It is particularly useful in multiclass classification, as it converts raw
     * class scores into probabilities.
     * <p>
     * Properties:
     * - The function outputs a probability distribution over all classes.
     * - Sum of output probabilities is always equal to 1.
     * <p>
     * Equation Details:
     * - The SOFTMAX function is defined as:
     * softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
     * where x_i represents the input for the i-th class, and exp is the exponential function.
     * <p>
     * Usage:
     * - Ideal for tasks where the outputs represent mutually exclusive classes (e.g., multi-class classification).
     * <p>
     * Note:
     * - The SOFTMAX function is parameterized with placeholder values for its `equation` and `derivative`,
     * which are properly implemented to support the mathematical definition.
     */
    SOFTMAX(((_, _) -> 0.0F), (_, _) -> 0.0F, 5); // Soft max activation without function or derivative.
    /**
     * Represents the mathematical function applied to tensors within an activation function.
     * This variable serves as a functional contract that defines how input tensor data
     * is transformed or processed by the specified activation function in the parent enum.
     * <p>
     * The {@code equation} encapsulates the actual logic of the operation through a
     * {@link TensorMapAbleFunction}, enabling dynamic and modular application of various
     * mathematical mappings suitable for neural network activations.
     * <p>
     * By leveraging this interface, the activation function can process input tensors
     * element-by-element, based on custom mapping strategies defined for each specific
     * activation type (e.g., SIGMOID, ReLU, etc.).
     */
    final private TensorMapAbleFunction equation;
    /**
     * Represents the derivative function associated with an activation function.
     * This is a specific implementation of the {@link TensorMapAbleFunction} interface
     * that defines the derivative computation needed for backpropagation and gradient-based
     * learning in neural networks.
     * <p>
     * The {@code derivative} is responsible for computing the rate of change of the
     * activation function with respect to its input, at a specific index and value.
     * This operation is essential for optimizing the weights during the training process.
     * <p>
     * In the context of this class, the {@code derivative} is tied directly to the
     * corresponding activation function, ensuring consistent forward and backward passes
     * during computations.
     */
    final private TensorMapAbleFunction derivative;
    // Index for GPU support
    private final int index;

    /**
     * Constructs an ActivationFunction with a specified equation and its corresponding derivative.
     *
     * @param equation   the primary tensor mapping function representing the activation equation
     * @param derivative the tensor mapping function representing the derivative of the activation equation
     */
    ActivationFunction(TensorMapAbleFunction equation, TensorMapAbleFunction derivative, int index) {
        this.equation = equation;
        this.derivative = derivative;
        this.index = index;
    }

    /**
     * Retrieves the tensor mapping function that represents the activation equation of this activation function.
     *
     * @return the tensor mapping function defined as the primary equation for this activation function
     */
    public TensorMapAbleFunction getEquation() {
        return equation;
    }

    /**
     * Retrieves the tensor mapping function that represents the derivative
     * of the activation equation for this activation function.
     *
     * @return the tensor mapping function that computes the derivative of
     * the activation equation.
     */
    public TensorMapAbleFunction getDerivative() {
        return derivative;
    }

    public int getIndex() {
        return index;
    }
}