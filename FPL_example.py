import numpy as np
import pandas as pd
from c_utils import list_sum
from turbo_ninja_ga.genetic_algorithm import GeneticAlgorithm
from turbo_ninja_ga.utils import fast_random_choice


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
        self.fitness_score = self.calc_fitness()  ##

    def calc_fitness(self):
        if len(set(self.chromosone)) != 15:
            return 0.0
        self.team_cost = list_sum(
            [self.full_data[id]["value"] for id in self.chromosone]
        )
        if self.team_cost > 1000:
            return 0.0
        return list_sum([self.full_data[id]["season_points"] for id in self.chromosone])

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


def decode_chromosonse(chromosone, full_data):
    names = [full_data[x]["name"] for x in chromosone]
    return names


if __name__ == "__main__":

    df = pd.read_csv("/Users/TimothyW/Fun/genetic_algorithm/lastest_gw_data.csv")
    df["season_points"] = (
        df.sort_values("GW")
        .groupby("element")["total_points"]
        .transform(lambda x: x.expanding(1).sum())
    )
    df["player_id"] = df.element
    df["element_type"] = df.position
    df.index = df["player_id"]
    df = df[df.GW == 4]

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

    print(decode_chromosonse(GA.best_best_chromosone.chromosone, FPLTeam.full_data))

    # GA.get_best_solution

    # GA.get_metrics
