from audioop import add
import numpy as np
import pandas as pd
from more_itertools import grouper
from turbo_ninja_ga.fucks import uniform_fuck
from turbo_ninja_ga.mutations import add_mutations


class FPLTeam:

    full_data = []
    gk_data = []
    def_data = []
    mid_data = []
    fwd_data = []

    def __init__(self, data, from_chromosone=False):
        if len(data) != 15:
            raise ValueError(" gene list must be of length 15, the size of an FPL team")
        if from_chromosone:
            if len(self.full_data) == 0:
                raise RuntimeError(
                    "Need to add reference data to class before initialising any instances from chromosone"
                )

            self.ref_data = self.full_data.loc[data]
            self.chromosone = np.array(data)
        else:
            self.ref_data = data
            self.chromosone = self.encode_chromosone()

        self.fitness_score = self.calc_fitness()

        if len(self.gk_data) == 0:
            self.gk_data = self.full_data[self.full_data.element_type == "GK"]
        if len(self.def_data) == 0:
            self.def_data = self.full_data[self.full_data.element_type == "DEF"]
        if len(self.mid_data) == 0:
            self.mid_data = self.full_data[self.full_data.element_type == "MID"]
        if len(self.fwd_data) == 0:
            self.fwd_data = self.full_data[self.full_data.element_type == "FWD"]

        # print("Initialised")

    def encode_chromosone(self):
        return self.ref_data.sort_values(["element_type", "total_points"]).index.values

    def calc_fitness(self):
        self.team_cost = np.sum(self.ref_data["now_cost"])
        if self.team_cost > 1000:
            return 0.0
        if self.ref_data.index.nunique() != 15:
            return 0.0
        return np.sum(self.ref_data["total_points"])

    def mutate_position(self, x):
        if x > len(self.chromosone) - 1:
            raise ValueError(f"position {x} is out of range for mutation")
        if x < 2:
            self.chromosone[x] = np.random.choice(self.gk_data.index)
        elif x < 7:
            self.chromosone[x] = np.random.choice(self.def_data.index)
        elif x < 11:
            self.chromosone[x] = np.random.choice(self.mid_data.index)
        else:
            self.chromosone[x] = np.random.choice(self.fwd_data.index)

        self.ref_data = self.full_data.loc[self.chromosone]
        self.fitness_score = self.calc_fitness()


def generate_chromosone_df(gks, defs, mids, fwds):
    return pd.concat(
        [gks.sample(2), defs.sample(5), mids.sample(5), fwds.sample(3)], axis=0
    )


# def uniform_fuck(chromosone_A, chromosone_B):
#     if len(chromosone_A) != len(chromosone_B):
#         return ValueError(
#             "you wouldn't ask a great dane to fuck a chihuaua, so why are you trying to mate different length chromosones?!"
#         )
#     child_chromosone = np.where(
#         np.random.uniform(size=len(chromosone_A)) > 0.5, chromosone_A, chromosone_B
#     )
#     return FPLTeam(child_chromosone, from_chromosone=True)


def calc_iteration_stats(population):
    # sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
    fitness_scores = [p.fitness_score for p in population]
    stats_dict = {
        "fitness_mean": np.mean(fitness_scores),
        "fitness_median": np.median(fitness_scores),
        "best_solution_score": np.max(fitness_scores),
    }
    return stats_dict


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
                self.population_type(
                    uniform_fuck(a.chromosone, b.chromosone), from_chromosone=True
                )
                for a, b in breeding_pairs
            ]
            self.population = elite + offspring
            self.pop_size = len(self.population)
            for i in self.population:
                add_mutations(i)
            iteration_stats = calc_iteration_stats(self.population)
            if iteration_stats["best_solution_score"] > best_best_score:
                best_best_score = iteration_stats["best_solution_score"]
            print(
                f"After {n} iterations stats are {iteration_stats}, best_best_score = {best_best_score}"
            )


if __name__ == "__main__":

    df = pd.read_csv("/Users/TimothyW/Fun/genetic_algorithm/fpl_player_data.csv")

    gks = df[df.element_type == "GK"]
    defs = df[df.element_type == "DEF"]
    mids = df[df.element_type == "MID"]
    fwds = df[df.element_type == "FWD"]

    FPLTeam.full_data = df

    population = [
        FPLTeam(
            generate_chromosone_df(gks, defs, mids, fwds).index, from_chromosone=True
        )
        for _ in range(0, 100)
    ]

    print("done")

    GA = GeneticAlgorithm(population)

    GA.run()

    # GA.get_best_solution

    # GA.get_metrics
