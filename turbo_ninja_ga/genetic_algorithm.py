import numpy as np
from turbo_ninja_ga.mutations import add_mutations
from turbo_ninja_ga.fucks import uniform_fuck


def calc_iteration_stats(population):
    # sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
    fitness_scores = [p.fitness_score for p in population]
    stats_dict = {
        "fitness_mean": np.mean(fitness_scores),
        "fitness_median": np.median(fitness_scores),
        "best_solution_score": np.max(fitness_scores),
    }
    
    return stats_dict, np.argmax(fitness_scores)


class GeneticAlgorithm:
    def __init__(
        self,
        population,
        elitism_fraction=0.1,
        selection_method="rank",
        mutation_type="random",
        mutation_frac=0.1,
        fuck_method="uniform",
        crossover_top_frac=0.4,
    ):
        self.population = population
        self.pop_size = len(population)
        self.elitism_fraction = elitism_fraction
        self.crossover_top_frac = crossover_top_frac
        self.population_type = type(self.population[0])

    def run(self, n_iterations=100, early_stopping_rounds=100):
        best_best_score = 0.0
        for n in range(0, n_iterations):
            sorted_population = sorted(
                self.population, key=lambda x: x.fitness_score, reverse=True
            )
            # elite, not_elite = select_elite()
            # breeding_pairs = select_breeding_pairs(non_elite)
            # offspring = fuck(breeding_pairs, fuck_method)
            # mutation
            n_elite = int(self.pop_size * self.elitism_fraction)
            elite = sorted_population[:n_elite]
            n_children = self.pop_size - n_elite
            viable_parents = sorted_population[
                n_elite : n_elite + int(n_children * self.crossover_top_frac)
            ]
            breeding_pairs = np.random.choice(viable_parents, size=(n_children, 2))
            offspring = [
                self.population_type(uniform_fuck(a.chromosone, b.chromosone))
                for a, b in breeding_pairs
            ]
            self.population = elite + offspring
            self.pop_size = len(self.population)
            for i in self.population:
                add_mutations(i)
            iteration_stats, best_chromosone_index = calc_iteration_stats(self.population)
            best_chromosone = self.population[best_chromosone_index]
            if iteration_stats["best_solution_score"] > best_best_score:
                best_best_score = iteration_stats["best_solution_score"]
                self.best_best_chromosone = best_chromosone
            print(
                f"After {n} iterations stats are {iteration_stats}, best_best_score = {best_best_score}"
            )
            print(self.best_best_chromosone.chromosone)
