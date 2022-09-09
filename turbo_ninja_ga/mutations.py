import numpy as np
import c_utils

def choose_mutation_posns(chromosone_length):
    mutation_probability = 1.0 / chromosone_length
    mutation_posns = np.argwhere(
        np.random.uniform(size=chromosone_length) < mutation_probability
    )
    return mutation_posns.reshape(mutation_posns.shape[0])


def add_mutations(individual):
    mutation_posns = c_utils.choose_mutation_posns(len(individual.chromosone))
    for mp in mutation_posns:
        individual.mutate_position(mp)
