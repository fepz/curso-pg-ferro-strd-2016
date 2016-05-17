"""
Implements the algorithm discussed at "An Evolutionary Method to Solve the Real-Time Scheduling Problem"
"""

import os
import sys
import random
from tabulate import tabulate
from deap import tools, base, creator
from argparse import ArgumentParser
from fileutils import get_rts_from_xmlfile


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Generate CPPs")
    parser.add_argument("xmlfile", help="XML file with RTS", type=str)
    parser.add_argument("xmlid", help="STR in file to process", type=int, default=1)
    return parser.parse_args()


def print_results(i, rts, cpus):
    """ pretty print """
    print("Chromosome {0} -- allocation result:".format(i))
    result_t = []
    for cpu_id, cpu in cpus.items():
        result_t.append((cpu_id, ', '.join(str(t["id"]) for t in cpu["tasks"])))
    print(tabulate(result_t, ["cpu", "tasks"], "psql"))


def print_population(population):
    """ pretty print """
    print("Final population: ")
    population_t = []
    for i in population:
        population_t.append((i, i.fitness.values))
    print(tabulate(population_t, ["chromosome", "fitness"], "psql"))


def genetic(rts, cpus):

    def func_X(a, b):
        """ length of messages transmitted from a towards b or from b towards a """
        comm_load = 0
        if "p" in a:
            for p in a["p"]:
                if p["id"] == b["id"]:
                    comm_load += p["payload"]
        # This part consideres incoming msgs from other tasks (so that task is the successor, b->a)
        # if "p" in b:
        #    for p in b["p"]:
        #        if p["id"] == a["id"]:
        #            comm_load += p["payload"]
        return comm_load

    def func_Xc(cpu_h, cpu_k):
        """ length of all messages (in bytes) to be transmitted between processors h and k through the network """
        summ = 0
        for task_h in cpu_h["tasks"]:
            for task_j in cpu_k["tasks"]:
                summ += func_X(task_h, task_j)
        return summ

    def func_Y(i, rts):
        """ load of the communication control network that the task i produces """
        comm_load = 0
        other_tasks = [t for t in rts if t is not i]
        for j in other_tasks:
            comm_load += func_X(i, j)
        return comm_load

    def func_Vp(cpus):
        """ total amount of information to be transferred over the network """
        summ = 0
        for cpu in cpus.values():
            other_cpus = [c for c in cpus.values() if c is not cpu]
            for other_cpu in other_cpus:
                summ += func_Xc(cpu, other_cpu)
        return summ

    def func_B(rts):
        """ Total amount of data to be transferred between predecessor and successors throught the network """
        summ = 0
        for task in rts:
            summ += func_Y(task, rts)
        return summ

    def func_cost_p(rts, cpus):
        return func_Vp(cpus) / func_B(rts)

    def get_cpu_alloc(individual):
        cpus_alloc = dict()
        for cpu_id in cpus.keys():
            cpus_alloc[cpu_id] = {"tasks": [], "uf": 0}  # tasks assigned to this cpu

        # A stack is assembled containing the tasks ordered by the value of the gene in decreasing order.
        task_stack = []
        for task_id, gene in enumerate(individual):
            task_stack.append((gene, rts[task_id]))
        task_stack.sort(key=lambda t: t[0], reverse=True)  # sort by gene value

        # clear previous task assignation
        #for cpu in cpus_alloc.values():
        #    cpu["tasks"].clear()

        # aux list  -- for easy sorting
        cpu_stack = [cpu for cpu in cpus_alloc.values()]

        # partition
        for _, max_task in task_stack:
            if "cpu" in max_task:
                cpu_id = max_task["cpu"]
                cpus_alloc[cpu_id]["tasks"].append(max_task)
            else:
                # create auxiliary stack with all task j that communicate with i
                aux_stack = []

                # add the succesors
                if "p" in max_task:
                    for p in max_task["p"]:
                        for task in rts:
                            if task["id"] == p["id"]:
                                aux_stack.append((func_Y(task, rts), task))

                # add other tasks that communicate with the task (the task will be the succesor)
                # for task in [t for t in rts if t is not max_task]:
                #    if "p" in task:
                #        for p in task["p"]:
                #           if p["id"] == max_task["id"]:
                #                aux_stack.append((func_Y(task, rts), task))

                cpu_a = None

                # order by func_y
                if aux_stack:
                    aux_stack.sort(key=lambda t: t[0], reverse=True)
                    aux_max_task = aux_stack[0]

                    # find the cpu at which the aux_max_task is allocated
                    for cpu in cpus_alloc.values():
                        if aux_max_task in cpu["tasks"]:
                            cpu_a = cpu

                # if not aux_stack or cpu_a is None:
                if cpu_a is None:
                    # update uf factors and allocate task to cpu with min uf
                    for cpu in cpus_alloc.values():
                        cpu["uf"] = sum([t["uf"] for t in cpu["tasks"]])
                    cpu_stack.sort(key=lambda c: c["uf"])
                    cpu_stack[0]["tasks"].append(max_task)
                else:
                    cpu_a["tasks"].append(max_task)

        # return the task allocation performed using the chromosome
        return cpus_alloc

    def cost(individual):
        # apply the cost function to the chromosome based in the cpu allocation produced
        return func_cost_p(rts, get_cpu_alloc(individual))

    def init_population(individual, rts, n):
        # generate initial population
        p_list = []

        # generate first chromosome
        chromosome = []
        for task in rts:
            g = func_Y(task, rts) * int(random.uniform(0, 1) * len(rts))
            chromosome.append(g)
        p_list.append(chromosome)

        # remaining chromosomes
        for _ in range(n - 1):
            new_chromosome = []
            nu = max(chromosome) / 10
            for g1 in chromosome:
                g2 = abs(g1 + int(random.uniform(-nu, nu)))
                new_chromosome.append(g2)
            p_list.append(new_chromosome)

        return [individual(c) for c in p_list]

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # Defines each individual as a list
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Initialize the population
    toolbox.register("population", init_population, creator.Individual, rts)

    # Applies a gaussian mutation of mean mu and standard deviation sigma on the input individual. The indpb argument
    # is the probability of each attribute to be mutated.
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.01)

    # Use map to pass values
    toolbox.register("evaluate", cost)

    # Generate the initial population (first generation)
    population = toolbox.population(n=6)

    # Evaluate the first generation
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)

    for _ in range(120):  # generations

        # Select the k worst individuals among the input individuals.
        population_worst = tools.selWorst(population, int(len(population) / 2))

        # Perform a roulette selection and apply a crossover to the selected individuals
        for _ in range(len(population_worst)):
            pair = tools.selRoulette(population_worst, 2)  # roulette
            tools.cxOnePoint(pair[0], pair[1])  # one point crossover

        # Mutate
        for c in population_worst:
            toolbox.mutate(c)
            del c.fitness.values  # delete the fitness value

        # Evaluate again the entire population
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)

    # print the final population
    print_population(population)

    # memory constraint verification
    for i, ind in enumerate(population):
        valid_cpu = True
        ch_cpus = get_cpu_alloc(ind)
        for cpuid, cpu in ch_cpus.items():
            if cpus[cpuid]["capacity"] < sum([t["r"] for t in cpu["tasks"]]):
                valid_cpu = False
        if valid_cpu:
            print_results(i, rts, ch_cpus)
        else:
            print("Chromosome {0} -- Invalid assignation found.".format(i))


def main():
    args = get_args()

    # verify that the file exists
    if not os.path.isfile(args.xmlfile):
        print("Can't find {0} XML file.".format(args.xmlfile))
        sys.exit(1)

    # read the rts from the specified file
    rts = get_rts_from_xmlfile(args.xmlid, args.xmlfile)

    # complete missing task info
    for task in rts:
        if "uf" not in task:
            task["uf"] = task["c"] / task["t"]
        if "d" not in task:
            task["d"] = task["t"]

    cpus0 = {0: {"capacity": 15, "tasks": [], "id": 0},
             1: {"capacity": 15, "tasks": [], "id": 1}}

    cpus1 = {0: {"capacity": 300, "tasks": []},
             1: {"capacity": 1400, "tasks": []},
             2: {"capacity": 320, "tasks": []},
             3: {"capacity": 730, "tasks": []},
             4: {"capacity": 445, "tasks": []},
             5: {"capacity": 550, "tasks": []}}

    cpus = cpus1

    # run darwin, run
    genetic(rts, cpus)


if __name__ == '__main__':
    main()
