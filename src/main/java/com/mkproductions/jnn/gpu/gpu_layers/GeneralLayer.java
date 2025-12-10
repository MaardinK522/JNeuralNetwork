package com.mkproductions.jnn.gpu.gpu_layers;

import com.mkproductions.jnn.activationFunctions.ActivationFunction;
import com.mkproductions.jnn.cpu.entity.Tensor;

import com.mkproductions.jnn.optimzers.OptimizerFunctions;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.DoubleArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Represents an abstract neural network layer.
 * A layer consists of weights, a bias, an activation function, and a name. It defines the structure and behavior
 * that child classes must implement for performing forward and backward propagation. Layers can be customized
 * to represent different types of neural network operations, such as convolution, pooling, dense, etc.
 */
public abstract class GeneralLayer {
    private final ActivationFunction activationFunction;
    private final String name;
    // Layer data
    protected Tensor weights;
    protected Tensor weightsTranspose;
    protected Tensor bias;
    protected Tensor biasTranspose;
    // Velocity data
    protected Tensor velocityWeightsTensor;
    protected Tensor velocityBiasesTensors;
    // Momentum data
    protected Tensor momentumWeightsTensor;
    protected Tensor momentumBiasesTensors;
    // Layer delta data
    protected Tensor weightsGradients;
    protected Tensor biasesGradients;
    protected Tensor currentGradients;
    protected Tensor previousLayerGradient;
    // I/O
    protected Tensor input;
    protected Tensor inputTranspose;
    protected Tensor rawOutput;
    protected Tensor output;
    // Hyper parameters.
    protected final IntArray adamSteps = new IntArray();
    protected double layerIndex;
    protected DoubleArray momentumBeta1;
    protected DoubleArray momentumBeta2;
    protected DoubleArray learningRate;
    protected DoubleArray epsilon;

    protected TornadoExecutionPlan feedForwardExecutionPlan;
    protected TornadoExecutionPlan backPropagationExecutionPlan;

    protected TornadoExecutionPlan backPropagationSGDExecutionPlan;
    protected TornadoExecutionPlan backPropagationSGDWithMomentumExecutionPlan;
    protected TornadoExecutionPlan backPropagationAdaGradExecutionPlan;
    protected TornadoExecutionPlan backPropagationRMSPropagationExecutionPlan;
    protected TornadoExecutionPlan backPropagationAdamExecutionPlan;

    /**
     * Constructs a new Layer with the specified name and activation function.
     *
     * @param name       The unique identifier for this layer.
     * @param activation The activation function to be applied to the layer's output.
     */
    public GeneralLayer(String name, ActivationFunction activation) {
        this.name = name;
        this.activationFunction = activation;
    }

    public Tensor getInput() {
        return this.input;
    }

    public void setInput(Tensor input) {
        this.input.copy(input.getData(), input.getShape());
    }

    public Tensor getOutput() {
        return output;
    }

    public void setOutput(Tensor output) {
        this.output.copy(output.getData(), output.getShape());
    }

    public Tensor getWeights() {
        return weights;
    }

    public void setWeights(Tensor weights) {
        this.weights = weights;
        this.weightsTranspose = Tensor.transpose(weights);
    }

    public Tensor getBias() {
        return bias;
    }

    public void setBias(Tensor bias) {
        this.bias = bias;
        this.biasTranspose = Tensor.transpose(bias);
    }

    public double getLearningRate() {
        return learningRate.get(0);
    }

    public void setLearningRate(double learningRate) {
        this.learningRate.set(0, learningRate);
    }

    public double getMomentumBeta2() {
        return momentumBeta2.get(0);
    }

    public void setMomentumBeta2(double momentumBeta2) {
        this.momentumBeta2.set(0, momentumBeta2);
    }

    public double getMomentumBeta1() {
        return momentumBeta1.get(0);
    }

    public void setMomentumBeta1(double momentumBeta1) {
        this.momentumBeta1.set(0, momentumBeta1);
    }

    public double getLayerIndex() {
        return layerIndex;
    }

    public void setLayerIndex(double layerIndex) {
        this.layerIndex = layerIndex;
    }

    public int getAdamSteps() {
        return adamSteps.get(0);
    }

    public void setWeightsGradients(Tensor weightsGradients) {
        this.weightsGradients.copy(weightsGradients.getData(), weightsGradients.getShape());
    }

    public void setBiasesGradients(Tensor biasesGradients) {
        this.biasesGradients.copy(biasesGradients.getData(), biasesGradients.getShape());
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
     * @return An array of tensors representing the gradients to be propagated back to the previous layer.
     */
    public abstract Tensor[] backward(Tensor input, Tensor currentGraidents);

    public void backPropagationSGD() {
        this.backPropagationSGDExecutionPlan.execute();
    }

    public void backPropagationSGDWithMomentum() {
        this.backPropagationSGDWithMomentumExecutionPlan.execute();
    }

    public void backPropagationAdaGrad() {
        this.backPropagationAdaGradExecutionPlan.execute();
    }

    public void backPropagationRMSPropagation() {
        this.backPropagationRMSPropagationExecutionPlan.execute();
    }

    public void backPropagationAdam() {
        this.backPropagationAdamExecutionPlan.execute();
    }

    protected abstract TornadoExecutionPlan[] prepareTaskGraphs();

    public void initLayerParameters(int layerIndex, double momentumFactorBeta1, double momentumFactorBeta2, double epsilon, double learningRate, int[] inputShape, int[] outputShape) {
        this.layerIndex = layerIndex;
        this.momentumBeta1 = DoubleArray.fromArray(new double[]{momentumFactorBeta1});
        this.momentumBeta2 = DoubleArray.fromArray(new double[]{momentumFactorBeta2});
        this.learningRate = DoubleArray.fromArray(new double[]{learningRate});
        this.epsilon = DoubleArray.fromArray(new double[]{epsilon});
        // Initializing inputs and outputs.
        this.input = new Tensor(inputShape);
        this.inputTranspose = Tensor.transpose(input);
        this.output = new Tensor(outputShape);
        this.rawOutput = new Tensor(outputShape);
        this.currentGradients = new Tensor(outputShape);
        this.previousLayerGradient = new Tensor(inputShape);
        // Initializing velocity and momentum for propagation.
        velocityWeightsTensor = weights.copyShape();
        velocityBiasesTensors = bias.copyShape();
        momentumWeightsTensor = weights.copyShape();
        momentumBiasesTensors = bias.copyShape();
        weightsGradients = weights.copyShape();
        biasesGradients = bias.copyShape();
        // Preparing task graph for Stochastic Gradient Descent.
        TaskGraph backPropagationSGDTaskGraph = new TaskGraph(STR."backPropagationSGDTaskGraph:\{layerIndex}");
        backPropagationSGDTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), weightsGradients.getData().getSize(), biasesGradients.getData().getSize());
        backPropagationSGDTaskGraph.task(STR."sgd:\{layerIndex}", OptimizerFunctions::applySGD, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), weightsGradients.getData().getSize(), biasesGradients.getData().getSize());
        backPropagationSGDTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData());
        // Preparing task graph for Stochastic Gradient Descent with Momentum
        TaskGraph backPropagationSGDWithMomentumTaskGraph = new TaskGraph(STR."backPropagationSGDWithMomentumTaskGraph:\{layerIndex}");
        backPropagationSGDWithMomentumTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData(), momentumBeta1);
        backPropagationSGDWithMomentumTaskGraph.task(STR."sgd_momentum:\{layerIndex}", OptimizerFunctions::applySGDMomentum, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData(), momentumBeta1, weights.getData().getSize(), bias.getData().getSize());
        backPropagationSGDWithMomentumTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData());
        // Preparing task graph for Adaptive Gradients.
        TaskGraph backPropagationAdaGradTaskGraph = new TaskGraph(STR."backPropagationAdaGradTaskGraph\{layerIndex}");
        backPropagationAdaGradTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData(), epsilon);
        backPropagationAdaGradTaskGraph.task(STR."aga_grad:\{layerIndex}", OptimizerFunctions::applyAdaGrad, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData(), this.epsilon, weights.getData().getSize(), bias.getData().getSize());
        backPropagationAdaGradTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData());
        // Preparing task graph for Root Mean Square propagation.
        TaskGraph backPropagationRMSPropagationTaskGraph = new TaskGraph(STR."backPropagationRMSPropagationTaskGraph\{layerIndex}");
        backPropagationRMSPropagationTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData());
        backPropagationRMSPropagationTaskGraph.task(STR."rms:\{layerIndex}", OptimizerFunctions::applyRMSPropagation, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData(), momentumBeta1, this.epsilon, weights.getData().getSize(), bias.getData().getSize());
        backPropagationRMSPropagationTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData());
        // Preparing task graph for Adaptive Momentum's
        TaskGraph backPropagationAdamTaskGraph = new TaskGraph(STR."backPropagationAdamTaskGraph\{layerIndex}");
        backPropagationAdamTaskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData(), momentumWeightsTensor.getData(), momentumBiasesTensors.getData(), momentumBeta1, momentumBeta2, this.epsilon, adamSteps);
        backPropagationAdamTaskGraph.task(STR."adam:\{layerIndex}", OptimizerFunctions::applyAdam, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData(), momentumWeightsTensor.getData(), momentumBiasesTensors.getData(), momentumBeta1, momentumBeta2, this.epsilon, adamSteps, weights.getData().getSize(), bias.getData().getSize());
        backPropagationAdamTaskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, this.learningRate, weights.getData(), bias.getData(), weightsGradients.getData(), biasesGradients.getData(), velocityWeightsTensor.getData(), velocityBiasesTensors.getData(), momentumWeightsTensor.getData(), momentumBiasesTensors.getData(), momentumBeta1, momentumBeta2, this.epsilon, adamSteps);
        var executionPlans = this.prepareTaskGraphs();
        this.feedForwardExecutionPlan = executionPlans[0];
        this.backPropagationExecutionPlan = executionPlans[1];
        this.backPropagationSGDExecutionPlan = new TornadoExecutionPlan(backPropagationSGDTaskGraph.snapshot());
        this.backPropagationSGDWithMomentumExecutionPlan = new TornadoExecutionPlan(backPropagationSGDWithMomentumTaskGraph.snapshot());
        this.backPropagationAdaGradExecutionPlan = new TornadoExecutionPlan(backPropagationAdaGradTaskGraph.snapshot());
        this.backPropagationRMSPropagationExecutionPlan = new TornadoExecutionPlan(backPropagationRMSPropagationTaskGraph.snapshot());
        this.backPropagationAdamExecutionPlan = new TornadoExecutionPlan(backPropagationAdamTaskGraph.snapshot());
    }

    public void setCurrentGradients(Tensor currentGradient) {
        this.currentGradients.copy(currentGradient.getData(), currentGradient.getShape());
    }
}