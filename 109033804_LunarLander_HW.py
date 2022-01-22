
import gym
import random
import math
import pickle
import csv
import numpy as np
from statistics import mean, median

"""
#### System Requirement ####

[Windows]

> pip install gym
> pip install box2d
> pip install pyglet

[macOS]

> pip3 install gym[all]
> pip3 install box2d

If you occur an ImportError like "ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.", please run the following installation.
> pip3 install pyglet==1.5.11
"""

"""
#### Your Job ####
1. Crossover:
        Please implement your crossover operation in the `crossover(cls, parent1, parent2, xover_rate)` function.
        Arguments: parent1 (Chromosome), parent2 (Chromosome), xover_rate (float)
        Return:    two children in a tuple, i.e., (child1, child2); each child is an instance of Chromosome.

2. Mutation:
        Please implement your crossover operation in the `mutate(cls, chrm, mutate_rate)` function.
        Arguments: chrm (Chromosome), mutate_rate (float)
        Return:    None

3. k-tournament selection:
        Please implement the k-tournament selection in the `parent_selection(self, k)` function.
        Arguments: self (GA), k (int)
        Return:    a selected parent (Chromosome)

4. GA parameters:
        Please try different GA parameters and report an appropiate setting for this problem.
"""


#### GA Parameters ####
"""
[YOUR JOB]
Try and find a proper parameter setting.
"""
EXP_TIMES = 10
GA_POPULATION_SIZE = 50
GA_GENERATION      = 300
GA_K_TOURNAMENT    = 5
GA_CROSSOVER_RATE  = 0.9
GA_MUTATION_RATE   = 0.001
GA_CROSSOVER_TYPE  = 1       # 0: uniform crossover, 1: n-point corssover
GA_MUTATION_TYPE   = 1       # 0: uniform mutation, 1: nonuniform mutation (Gaussian)

#### Evaluation Parameter ####
"""
This parameter controls the simulation times in one evaluation
Since the simulation environment is randomly generated, the simulation should be run several times to get the reliable performance
Notice: You may lower this parameter to make the whole GA run faster in order to get the sense of how different parameters affect the performance,
but it is suggested that you set it to 20 to get a reliable output for homework assignment
"""
SIMULATIONS_PER_EVALUATION = 20

#### [DON'T CHANGE] Evironment Parameters ####
OBSV_DIM   = 6
ACTION_DIM = 2
RESOLUTION = 4
N_CASES    = RESOLUTION ** OBSV_DIM
CHRM_DIM   = N_CASES * ACTION_DIM
ACTION_LB  = -1
ACTION_UB  = 1

env = gym.make("LunarLanderContinuous-v2")


class Chromosome:
    def __init__(self, dim) -> None:
        self.gene = [random.uniform(ACTION_LB, ACTION_UB) for i in range(dim)]
        self.fitness = 0.0

    @classmethod
    def crossover(cls, parent1, parent2, xover_rate, type) -> tuple:
        """
        [YOUR JOB]
        The following code implements the uniform crossover.
        Please implement your crossover operator and replace the following code with yours.
        Hint: You may try "n-point crossover", "whole arithmetic crossover", "blend crossover", etc.
        """
        dim = len(parent1.gene)
        child1, child2 = Chromosome(dim), Chromosome(dim)

        # uniform crossover
        if random.random() < xover_rate:

            # uniform crossover
            if type == 0:
                for i in range(dim):
                    if random.random() < 0.5:
                        child1.gene[i] = parent2.gene[i]
                        child2.gene[i] = parent1.gene[i]
                    else:
                        child1.gene[i] = parent1.gene[i]
                        child2.gene[i] = parent2.gene[i]


            # n-point corssover
            elif type == 1:
                # xpointNumber = random.randint(1, 8000)
                # print(xpointNumber)
                xpointNumber = 100;
                xpoints = sorted(random.sample(range(dim), xpointNumber))
                xpoints.append(dim)
                point = 0
                flag = 0

                for i in range(dim):
                    if i < xpoints[point]:
                        if flag == 0:
                            child1.gene[i] = parent1.gene[i]
                            child2.gene[i] = parent2.gene[i]
                        elif flag == 1:
                            child1.gene[i] = parent2.gene[i]
                            child2.gene[i] = parent1.gene[i]

                    elif i == xpoints[point]:
                        i -= 1
                        point += 1
                        if flag == 0:
                            flag = 1
                        else:
                            flag = 0
                    else:
                        print("something wrong")
        else:
            child1 = parent1
            child2 = parent2

        return child1, child2

    @classmethod
    def mutate(cls, chrm, mutate_rate, type, sigma) -> None:
        """
        [YOUR JOB]
        The following code implements the random resetting mutation.
        Please implement your crossover operator and replace the following code with yours.
        Hint: You may try "Gaussian mutation", "creep mutation", etc.
        """
        dim = len(chrm.gene)

        if type == 0:
            for i in range(dim):
                if random.random() < mutate_rate:
                    chrm.gene[i] = random.uniform(ACTION_LB, ACTION_UB)

        elif type == 1:
            for i in range(dim):
                if random.random() < mutate_rate:
                    temp = chrm.gene[i]
                    chrm.gene[i] += np.random.normal(0, sigma)    # mean and standard deviation
                    while abs(chrm.gene[i]) > 1:
                        chrm.gene[i] = temp
                        chrm.gene[i] += np.random.normal(0, sigma)

        return


class GA:
    def __init__(self, pop_size, k, xover_rate, mutate_rate):
        #### Validate arguments ####
        if (not isinstance(pop_size, int)) or  (pop_size < 1):
            raise ValueError("`pop_size` can only be positive integer.")

        if (not isinstance(k, int)) or (k < 1) or (k > pop_size):
            raise ValueError("`k` for tournament selection can only an integer between 1 and `pop_size`.")

        #### Variable declaration ####
        global CHRM_DIM
        self.k   = k
        self.pop = []
        self.pop_size    = pop_size
        self.xover_rate  = xover_rate
        self.xover_type  = GA_CROSSOVER_TYPE
        self.mut_type = GA_MUTATION_TYPE
        self.mutate_rate = mutate_rate
        self.best_so_far = None

        #### Initialize variables ####
        self.pop = [Chromosome(dim=CHRM_DIM) for i in range(pop_size)]
        self.best_so_far = self.pop[0]

        #### Evaluate the initial population ####
        for i, chrm in enumerate(self.pop):
            print(f"Evaluating {(i/self.pop_size)*100: 3.1f}%", end="\r")
            chrm.fitness = evaluate(chrm.gene)

            if chrm.fitness > self.best_so_far.fitness:
                self.best_so_far = chrm

    def evolve(self, sigma):
        offspring = []

        #### Reproduction ####
        while len(offspring) < self.pop_size:

            #### Parent selection ####
            # p1 = self.parent_selection(k=GA_K_TOURNAMENT)
            # p2 = self.parent_selection(k=GA_K_TOURNAMENT)

            p1, p2 = self.parent_selection(k=GA_K_TOURNAMENT)


            #### Crossover ####
            c1, c2 = Chromosome.crossover(p1, p2, self.xover_rate, self.xover_type)

            #### Mutation ####
            Chromosome.mutate(c1, self.mutate_rate, self.mut_type, sigma)
            Chromosome.mutate(c2, self.mutate_rate, self.mut_type, sigma)

            offspring += [c1, c2]

        #### Evaluate offspring ####
        for i, chrm in enumerate(offspring):
            print(f"Evaluating {(i/self.pop_size)*100: 3.1f}%", end="\r")
            chrm.fitness = evaluate(chrm.gene)

            if chrm.fitness > self.best_so_far.fitness:
                self.best_so_far = chrm

        #### Survival selection ####
        """
        [OPTIONAL]
        You may play with different survival selection strategies.
        I've implemented the (mu +lambda) and (mu, lambda) strategies for you as follows.
        """
        ## (mu + lambda) ##
        intermediate_pop = self.pop + offspring
        self.survival_selection(pool=intermediate_pop, n_survivor=self.pop_size)

        ## (mu, lambda) ##
        # self.survivors(pool=offspring, n_survivor=self.pop_size)

    def parent_selection(self, k) -> Chromosome:
        """
        [YOUR JOB]
        The following code implements the random parent selection.
        Please implement your parent selection strategy and replace the following code with yours.
        """
        # return self.pop[random.randint(0, (self.pop_size-1))]

        index = []
        while (len(index) < 2):
            tournament = random.sample(range(len(self.pop)), k)
            index.append(max(tournament))

            if len(index) == 2:
                if index[0] == index[1]:
                    index.pop()

        p1 = self.pop[index[0]]
        p2 = self.pop[index[1]]

        return p1, p2

    def survival_selection(self, pool, n_survivor) -> list:
        """
        I've implement this function for you. No need to change.
        """
        # If `pool` is the offspring only, this performs (mu, lambda) selection.
        # If `pool` is the union of the population and offspring, this performs (mu + lambda) selection.

        if n_survivor == len(pool):
            self.pop = pool
        else:
            sorted_pool = sorted(pool, key=lambda chrm:chrm.fitness, reverse=True)
            self.pop    = sorted_pool[:n_survivor]


#### [DON'T CHANGE] Please don't change any part of this function. ####
def evaluate(gene:list, repeat=SIMULATIONS_PER_EVALUATION, mode="mean", display=False) -> float:
    rewards = []

    for i in range(repeat):
        env.reset()
        action = [0, 0]
        epsiodic_reward = 0.0
        if display:
            while True:
                env.render()
                obsv, reward, done, _ = env.step(action)
                epsiodic_reward += reward
                action = get_action(obsv, gene)
                if done:
                    break
            rewards.append(epsiodic_reward)
        else:
            while True:
                obsv, reward, done, _ = env.step(action)
                epsiodic_reward += reward
                action = get_action(obsv, gene)
                if done:
                    break
            rewards.append(epsiodic_reward)

    if mode == "mean":
        return mean(rewards)

    elif mode == "median":
        return median(rewards)

    elif mode == "min":
        return min(rewards)

    elif mode =="max":
        return max(rewards)

    else:
        raise ValueError("The argument `mode` can be 'mean', 'median', 'min', or 'max', but '{mode}' was given.")


#### [DON'T CHANGE] Please don't change any part of this function. ####
def get_action(observation:list, gene:list) -> list:
    obsv_dim  = 6
    obsv_grid = [
        [0.5,  0.0, -0.5],      # x position
        [0.7,  0.1, -0.5],      # x volecity
        [0.5,  0.0, -0.5],      # y position
        [0.0, -0.5, -1.0],      # y volecity
        [1.0,  0.0, -1.0],      # angle
        [2.0,  0.0, -2.0],      # angular volecity
    ]

    obsv  = observation[:obsv_dim]
    level = [None] * obsv_dim

    for i in range(obsv_dim):
        if obsv[i] > obsv_grid[i][0]:
            level[i] = 0
        elif obsv[i] > obsv_grid[i][1]:
            level[i] = 1
        elif obsv[i] > obsv_grid[i][2]:
            level[i] = 2
        else:
            level[i] = 3

    policy_i = 0
    for i in range(obsv_dim):
        policy_i += level[i] * math.pow(4, i)

    policy_i = int(policy_i) * 2
    action   = [gene[policy_i], gene[policy_i+1]]

    return action

####################################################


#### Initialize a GA instance ####
with open('adaptivefinal.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(["GA_POPULATION_SIZE", str(GA_POPULATION_SIZE)])
    writer.writerow(["GA_GENERATION", str(GA_GENERATION)])
    writer.writerow(["GA_K_TOURNAMENT", str(GA_K_TOURNAMENT)])
    writer.writerow(["GA_CROSSOVER_RATE", str(GA_CROSSOVER_RATE)])
    writer.writerow(["GA_MUTATION_RATE", str(GA_MUTATION_RATE)])
    writer.writerow(["GA_CROSSOVER_TYPE", str(GA_CROSSOVER_TYPE)])
    writer.writerow(["GA_MUTATION_TYPE", str(GA_MUTATION_TYPE)])
    writer.writerow(["SIMULATIONS_PER_EVALUATION", str(SIMULATIONS_PER_EVALUATION)])

    for t in range(EXP_TIMES):
        writer.writerow(["EXP Times", str(t)])
        ga = GA(
            pop_size=GA_POPULATION_SIZE,
            k=GA_K_TOURNAMENT,
            xover_rate=GA_CROSSOVER_RATE,
            mutate_rate=GA_MUTATION_RATE,
        )

        AnytimeBehavior = []

        print(f"Exp_times: {t: 2d}, Generation {0: 2d}, best fitness = {ga.best_so_far.fitness:4.2f}")
        evaluate(ga.best_so_far.gene, display=False)
        AnytimeBehavior.append(ga.best_so_far.fitness)

#### Evolve until termination criterion is met. #####
        for i in range(GA_GENERATION):
            if (i < 100):
                ga.evolve(0.6)
            else:
                ga.evolve(0.3)
            print(f"Exp_times: {t: 2d}, Generation {(i+1): 2d}, best fitness = {ga.best_so_far.fitness:4.2f}")
            evaluate(ga.best_so_far.gene, display=False)
            AnytimeBehavior.append(ga.best_so_far.fitness)

#### Store the best solution ####
# with open('best_gene.pickle', 'wb') as f:
#         pickle.dump(ga.best_so_far.gene, f)

        for i in range(len(AnytimeBehavior)):
            writer.writerow([str(AnytimeBehavior[i])])


# LunarLander_HW.py
# 目前顯示的是「LunarLander_HW.py」。
