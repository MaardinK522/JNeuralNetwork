package com.mkproductions.jnn.cpu.entity;

/**
 * This interface defines a functional contract for mapping operations over tensor data.
 * It provides a single method, {@code map}, which is intended to transform or process
 * a tensor value at a specific flat index. Implementations of this interface can define
 * various mapping logic based on the indexing and value of tensor elements.
 * <p>
 * Typical use cases for this interface include applying transformations, scaling values,
 * or adapting tensor data within computational workflows.
 */
public interface TensorMapAbleFunction {
    /**
     * Maps a given value, associated with a specific flat index, to a transformed
     * output. The mapping logic may depend on the value as well as the flat index
     * of the input, allowing for customized transformations or processing of tensor data.
     *
     * @param flatIndex the flat index of the element in the tensor. This represents
     *                  the position of the element in a linearized tensor structure.
     * @param value     the value of the tensor element at the specified flat index.
     * @return the transformed value resulting from the mapping operation.
     */
    double map(int flatIndex, double value);
}