from audioop import add
import numpy as np
import pandas as pd
from more_itertools import grouper
from turbo_ninja_ga.fucks import uniform_fuck
from turbo_ninja_ga.mutations import add_mutations


def fast_random_choice(l):
    return l[int(np.random.random() * len(l))]


class FPLTeam:

    full_data = {}
    gk_data = {}
    def_data = {}
    mid_data = {}
    fwd_data = {}

    def __init__(self, chromosone):
        if len(chromosone) != 15:
            raise ValueError(" gene list must be of length 15, the size of an FPL team")
        if len(self.full_data) == 0:
            raise RuntimeError(
                "Need to add reference data to class before initialising any instances from chromosone"
            )
        self.chromosone = chromosone
        self.fitness_score = self.calc_fitness()

    def calc_fitness(self):
        if len(set(self.chromosone)) != 15:
            return 0.0
        self.team_cost = np.sum(
            np.array([self.full_data[id]["now_cost"] for id in self.chromosone])
        )
        if self.team_cost > 1000:
            return 0.0
        return np.sum(
            np.array([self.full_data[id]["total_points"] for id in self.chromosone])
        )

    def mutate_position(self, pos):
        if pos > len(self.chromosone) - 1:
            raise ValueError(" positon out of range ")
        if pos < 2:
            self.chromosone[pos] = fast_random_choice(list(self.gk_data.keys()))
        elif pos < 7:
            self.chromosone[pos] = fast_random_choice(list(self.def_data.keys()))
        elif pos < 12:
            self.chromosone[pos] = fast_random_choice(list(self.mid_data.keys()))
        else:
            self.chromosone[pos] = fast_random_choice(list(self.fwd_data.keys()))
        self.fitness_score = self.calc_fitness()


def generate_chromosone(gks, defs, mids, fwds):
    gk_list = list(gks.sample(2).index)
    defs_list = list(defs.sample(5).index)
    mids_list = list(mids.sample(5).index)
    fwds_list = list(fwds.sample(3).index)
    return gk_list + defs_list + mids_list + fwds_list


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
                self.population_type(uniform_fuck(a.chromosone, b.chromosone))
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
    df["player_id"] = df.index

    gks = df[df.element_type == "GK"]
    defs = df[df.element_type == "DEF"]
    mids = df[df.element_type == "MID"]
    fwds = df[df.element_type == "FWD"]

    FPLTeam.full_data = df.to_dict(orient="index")
    FPLTeam.gk_data = gks.reindex(gks.player_id).to_dict(orient="index")
    FPLTeam.def_data = defs.reindex(defs.player_id).to_dict(orient="index")
    FPLTeam.mid_data = mids.reindex(mids.player_id).to_dict(orient="index")
    FPLTeam.fwd_data = fwds.reindex(fwds.player_id).to_dict(orient="index")

    population = [
        FPLTeam(generate_chromosone(gks, defs, mids, fwds)) for _ in range(0, 100)
    ]

    print("done")

    GA = GeneticAlgorithm(population)

    GA.run(1000)

    # GA.get_best_solution

    # GA.get_metrics
