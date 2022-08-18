import pandas as pd
import numpy as np


def create_dicts(df):
    gk_dict = {}
    def_dict = {}
    mid_dict = {}
    fwd_dict = {}
    for _, row in df.iterrows():
        if row["element_type"] == "GK":
            gk_dict[row["player_id"]] = (row["now_cost"], row["total_points"])
        elif row["element_type"] == "MID":
            mid_dict[row["player_id"]] = (row["now_cost"], row["total_points"])
        elif row["element_type"] == "DEF":
            def_dict[row["player_id"]] = (row["now_cost"], row["total_points"])
        elif row["element_type"] == "FWD":
            fwd_dict[row["player_id"]] = (row["now_cost"], row["total_points"])
        else:
            print("WTF")

    return gk_dict, def_dict, mid_dict, fwd_dict


def initialise_population(gk_dict, def_dict, mid_dict, fwd_dict, population_size=100):
    gks = np.random.choice(list(gk_dict.keys()), (population_size, 2))
    defs = np.random.choice(list(def_dict.keys()), (population_size, 5))
    mids = np.random.choice(list(mid_dict.keys()), (population_size, 5))
    fwds = np.random.choice(list(fwd_dict.keys()), (population_size, 3))

    population = np.hstack((gks, defs, mids, fwds))

    return population


def get_cost(x, merged_dict):
    return merged_dict[x][0]


vget_cost = np.vectorize(get_cost)


def get_points(x, merged_dict):
    return merged_dict[x][1]


vget_points = np.vectorize(get_points)


def nunique(a, axis):
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1


def score_solutions(population, merged_dict):
    costs = vget_cost(population, merged_dict)
    points = vget_points(population, merged_dict)

    chromosone_totalpoints = np.sum(points, axis=1)
    chromosone_cost = np.sum(costs, axis=1)

    unique_elements_row = nunique(population, 1)

    chromosone_scores = np.where(
        (chromosone_cost < 1000) & (unique_elements_row == 15),
        chromosone_totalpoints,
        0.0,
    )

    return (
        np.hstack(
            (population, np.reshape(chromosone_scores, (population.shape[0], 1)))
        ),
        (chromosone_cost > 1000).sum(),
        (unique_elements_row < 15).sum(),
    )


def fuck(parent_pair):
    # print(" -- - -- - -- ")
    # print(parent_pair)
    offspring = np.where(
        np.random.uniform(size=15) > 0.5, parent_pair[0], parent_pair[1]
    )
    # print(offspring)
    return offspring


def pair_parents(viable_parents):
    return lambda x: (viable_parents[x[0]], viable_parents[x[1]])


def get_new_chromo_mutation(chromo_pos, all_dicts):
    if chromo_pos[1] < 2:
        return np.random.choice(list(all_dicts[0].keys()))
    elif chromo_pos[1] < 7:
        return np.random.choice(list(all_dicts[1].keys()))
    elif chromo_pos[1] < 12:
        return np.random.choice(list(all_dicts[2].keys()))
    else:
        return np.random.choice(list(all_dicts[3].keys()))


def add_mutations(population, all_dicts):
    mutated_population = np.copy(population)
    chromo_length = population.shape[1]
    mutation_probability = 1.0 / float(chromo_length)
    mutation_pos = np.argwhere(
        np.random.uniform(size=population.shape) < mutation_probability
    )
    for chromo_mute_pose in mutation_pos:
        mutated_population[
            chromo_mute_pose[0], chromo_mute_pose[1]
        ] = get_new_chromo_mutation(chromo_mute_pose, all_dicts)

    return mutated_population


def breed(scored_population, elite_frac, crossover_top_frac, all_dicts):
    sorted_pop = scored_population[(-scored_population[:, 15]).argsort()]
    # 10% pass through
    n_sols = sorted_pop.shape[0]
    elite_n = int(n_sols * elite_frac)
    elite = sorted_pop[:elite_n, :15]
    n_children = n_sols - elite_n
    n_viable_parents = int(n_sols * crossover_top_frac)
    viable_parents = sorted_pop[elite_n : elite_n + n_viable_parents, :15]
    n_vp = viable_parents.shape[0]
    pairings_inds = np.random.choice(np.arange(0, n_vp), (n_children, 2))
    parent_pairs = np.apply_along_axis(pair_parents(viable_parents), 1, pairings_inds)
    offspring = np.array([fuck(x) for x in parent_pairs])
    new_population = np.vstack((elite, offspring))
    mutated_population = add_mutations(new_population, all_dicts)
    # print("X")

    return mutated_population


if __name__ == "__main__":

    df = pd.read_csv("/Users/TimothyW/Fun/genetic_algorithm/clean_clean_data.csv")

    all_dicts = create_dicts(df)

    merged_dict = {**all_dicts[0], **all_dicts[1], **all_dicts[2], **all_dicts[3]}

    population = initialise_population(*all_dicts)

    n_gens = 5000
    best_mean_score = 0
    best_mean_score_n = 0
    best_solution_score = 0
    best_solution_score_n = 0
    best_solution = None

    for n in range(0, n_gens):
        population_with_scores, n_over_budget, n_repeats = score_solutions(
            population, merged_dict
        )
        print(f"n over budget = {n_over_budget}, n with repeasts = {n_repeats}")
        mean_score = population_with_scores[:, 15].mean()
        print(
            f"n = {n},   mean score = {mean_score},   best_mean_score = {best_mean_score}"
        )
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_mean_score_n = n
        curr_best_solution_score = np.max(population_with_scores[:, 15])
        if curr_best_solution_score > best_solution_score:
            best_solution_score = curr_best_solution_score
            best_solution = population_with_scores[
                np.argmax(population_with_scores[:, 15])
            ]
            best_solution_score_n = n

        print(f"best solution score = {best_solution_score}")
        if n - best_solution_score_n > 10000:
            break

        population = breed(population_with_scores, 0.1, 0.4, all_dicts)

    print(df[df.player_id.isin(best_solution[:15])])
    print("DONE")
