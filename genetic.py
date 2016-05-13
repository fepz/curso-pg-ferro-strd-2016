"""
Implements the algorithm discussed at "An Evolutionary Method to Solve the Real-Time Scheduling Problem"
"""

import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import random
from deap import tools
from argparse import ArgumentParser
from fileutils import get_rts_from_xmlfile


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Generate CPPs")
    parser.add_argument("xmlfile", help="XML file with RTS", type=str)
    parser.add_argument("xmlid", help="STR in file to process", type=int, default=1)
    return parser.parse_args()


def print_results(rts, cpus):
    """ pretty print """
    print("Allocation result:")
    for cpu_id, cpu in cpus.items():
        print("Cpu {0}:".format(cpu_id))
        for task in cpu["tasks"]:
            print(" - Task {0}".format(task["id"]))


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

    def func_Y(i, rts):
        """ load of the communication control network that the task i produces """
        comm_load = 0
        other_tasks = [t for t in rts if t is not i]
        for j in other_tasks:
            comm_load += func_X(i, j)
        return comm_load

    def func_Xc(cpu_h, cpu_k):
        """ length of all messages (in bytes) to be transmitted between processors h and k through the network """
        summ = 0
        for task_h in cpu_h["tasks"]:
            for task_j in cpu_k["tasks"]:
                summ += func_X(task_h, task_j)
        return summ

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

    # initialization
    n_p = 128  # number of generations
    n_g = 16  # population
    pc = 0.5  # crossover probability
    pm = 0.001  # mutation probability
    m = len(rts)  # tasks
    n = len(cpus)  # processors
    B = func_B(rts) # total of data to be transferred between predecessor and successor

    W = 70  # network bandwith in bytes/slot
    Dmin = min([t["d"] for t in rts])  # minimum deadline
    WDmin = W * Dmin

    # generate first chromosome
    chromosome1 = []
    for task in rts:
        g = func_Y(task, rts) * int(random.uniform(0,1) * m)
        chromosome1.append(g)

    # the dict stores the cpus tasks lists
    chromosomes = [(chromosome1, dict())]

    # generate remain population
    for _ in range(n_g - 1):
        new_chromosome = []
        nu = max(chromosome1) / 10
        for g1 in chromosome1:
            g2 = g1 + random.uniform(-nu, nu)
            new_chromosome.append(g2)
        chromosomes.append((new_chromosome, dict()))

    # initialize the dict associated with each chromosome
    for _, cpus_alloc in chromosomes:
        for cpu_id in cpus.keys():
            cpus_alloc[cpu_id] = {"tasks":[], "uf":0}  # tasks assigned to this cpu

    # aux for reordering and other stuff
    chromosomes_stack = []

    # do generations
    for _ in range(n_p):

        chromosomes_stack.clear()

        # A stack is assembled containing the tasks ordered by the value of its allele in decreasing order.
        for item in chromosomes:
            chromosome, cpus_alloc = item[0], item[1]

            task_stack = []
            for task_id, allele in enumerate(chromosome):
                task_stack.append((allele, rts[task_id]))
            task_stack.sort(key=lambda allele: allele[0], reverse=True)

            # clear previous task assignation
            for cpu in cpus_alloc.values():
                cpu["tasks"].clear()

            # aux -- for easy sorting
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
                    #for task in [t for t in rts if t is not max_task]:
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

                    #if not aux_stack or cpu_a is None:
                    if cpu_a is None:
                        # update uf factors and allocate task to cpu with min uf
                        for cpu in cpus_alloc.values():
                            cpu["uf"] = sum([t["uf"] for t in cpu["tasks"]])
                        cpu_stack.sort(key=lambda c: c["uf"])

                        cpu_stack[0]["tasks"].append(max_task)
                    else:
                        cpu_a["tasks"].append(max_task)

            # apply the cost function to the chromosomes
            chromosomes_stack.append((func_cost_p(rts, cpus_alloc), chromosome))

        # evaluation, cost and fitness

        # elistist selection -- for the sake of simplicity, separate the first two chromosomes as the 'best'
        chromosomes_stack.sort(key=lambda c: c[0])  # order by ascending value
        chromosomes_stack = chromosomes_stack[2:]

        # apply roulette selection on the rest -- an apply crossover
        sum_fitness = sum([c[0] for c in chromosomes_stack])  # sum of all fitness values (cost function result)
        chromosomes_stack = [(c[0] / sum_fitness, c[1]) for c in chromosomes_stack]  # normalization

        # calculate probabilities
        sum_prob = 0
        probs = []
        for fitness, c in chromosomes_stack:
            prob = sum_prob + fitness
            probs.append(prob)
            sum_prob += fitness

        # perform crossover
        for _ in range(int(n_g / 2)):
            cs = []  # selected chromosomes
            for s in range(2):
                r = random.uniform(0,1)
                for idx, p in enumerate(probs):
                    if r < p:
                        cs.append(chromosomes_stack[idx][1])
            # uses radint() function for selecting a crossover point
            tools.cxOnePoint(cs[0], cs[1])

        # mutate
        for c in chromosomes_stack:
            tools.mutGaussian(c[1], pm, pm, pm)

    # memory constraint verification
    for idx, chromosome in enumerate(chromosomes):
        print("Chromosome {0}".format(idx))
        valid_cpu = True
        ch_cpus = chromosome[1]
        for cpuid, cpu in ch_cpus.items():
            if cpus[cpuid]["capacity"] < sum([t["r"] for t in cpu["tasks"]]):
                valid_cpu = False
        if valid_cpu:
            print_results(rts, chromosome[1])
        else:
            print(" -- Invalid assignation found.")

    # bandwidth constraint
    bw_ok = B <= WDmin

    return


def main():
    args = get_args()

    if not os.path.isfile(args.xmlfile):
        print("Can't find {0} XML file.".format(args.xmlfile))
        sys.exit(1)

    rts = get_rts_from_xmlfile(args.xmlid, args.xmlfile)

    for task in rts:
        if "uf" not in task:
            task["uf"] = task["c"] / task["t"]
        if "d" not in task:
            task["d"] = task["t"]

    cpus = {0: {"capacity": 300, "tasks": []},
            1: {"capacity": 1400, "tasks": []},
            2: {"capacity": 320, "tasks": []},
            3: {"capacity": 730, "tasks": []},
            4: {"capacity": 445, "tasks": []},
            5: {"capacity": 550, "tasks": []}}

    cpus1 = {0: {"capacity": 10000, "tasks": [], "id":0},
            1: {"capacity": 10000, "tasks": [], "id":1},
            2: {"capacity": 10000, "tasks": [], "id":2},
            3: {"capacity": 12000, "tasks": [], "id":3},
            4: {"capacity": 7000, "tasks": [], "id":4},
            5: {"capacity": 7000, "tasks": [], "id":5},
            6: {"capacity": 12000, "tasks": [], "id":6},
            7: {"capacity": 10000, "tasks": [], "id":7}}

    cpusz = {0: {"capacity": 15, "tasks": [], "id": 0},
            1: {"capacity": 15, "tasks": [], "id": 1}}

    genetic(rts, cpus)


if __name__ == '__main__':
    main()
