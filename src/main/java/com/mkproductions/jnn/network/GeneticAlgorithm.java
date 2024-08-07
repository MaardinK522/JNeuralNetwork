package com.mkproductions.jnn.network;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;

public class GeneticAlgorithm<Chromosome extends GeneticAlgorithm.MutableChromosome, Gene extends GeneticAlgorithm.MutableGene> {
    protected final Random random;
    protected double mutationRate;
    protected double crossoverRate;
    int generationCount = -1;
    private final Function<Object[], Chromosome> initializer;
    private final int generationLength;

    public GeneticAlgorithm(int generationLength, Function<Object[], Chromosome> initializer) {
        this.initializer = initializer;
        this.generationLength = generationLength;
        this.random = new Random();
        this.mutationRate = 0.01;
        this.crossoverRate = 0.10;
    }

    public ArrayList<Chromosome> getRandomGeneration(Object[] parameters) {
        ArrayList<Chromosome> currentGeneration = new ArrayList<>();
        for (int i = 0; i < this.generationLength; i++)
            currentGeneration.add(initializer.apply(parameters));
        this.generationCount++;
        return currentGeneration;
    }

    public ArrayList<Chromosome> getNewGeneration(ArrayList<Chromosome> generation) {
        ArrayList<Chromosome> currentGeneration = new ArrayList<>();
        generation.sort((b1, b2) -> {
            if (b1.fitness < b2.fitness)
                return 1;
            else if (b1.fitness == b2.fitness)
                return 0;
            return -1;
        });
        Set<Chromosome> uniqueChromosome = new HashSet<>();
//        for
        this.generationCount++;
        return currentGeneration;
    }

    public int getGenerationCount() {
        return this.generationCount;
    }

    public void setMutationRate(double mutationRate) {
        this.mutationRate = mutationRate;
    }


    public abstract static class MutableGene {

        public abstract void mutate(Random random, double mutationRate);
    }

    public abstract static class MutableChromosome {
        public double fitness = 0;

        public abstract void mutateGene(Random random, double mutationRate);
    }
}
