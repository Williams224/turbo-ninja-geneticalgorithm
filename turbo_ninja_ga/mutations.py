import numpy as np


def add_mutations(individual):
    mutation_probability = 1.0 / len(individual.chromosone)
    mutation_posns = np.argwhere(
        np.random.uniform(size=len(individual.chromosone)) < mutation_probability
    )
    for mp in mutation_posns:
        individual.mutate_position(mp)
