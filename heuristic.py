"""
Implements the algorithm discussed at "A Heuristic Approach to the Multitask-Multiprocessor Assignment Problem using
the Empty-Slots Method and Rate Monotonic Scheduling"
"""

import os
import math
import sys
import networkx as nx
import itertools
from argparse import ArgumentParser
import random
from fileutils import get_rts_from_xmlfile, get_rts_from_pot_file
from rtsutils import verify_with_empty_slot
from tabulate import tabulate


def heuristic(rts, cpus):

    def last_starting_time(task, tasks):
        """ Calculate the last possible starting time for executing task """
        def get_workload(t, tasks):
            """ Calculate the worload required for tasks in [0,t] -- uses floor() """
            workload = 0
            for task in tasks:
                workload += (task["c"] * math.ceil(t / task["t"]))
            return workload

        def get_empty_slots(t, tasks):
            """ Calculates the amount of empty slots left by the execution of tasks in [0, t] """
            workload = 0
            for task in tasks:
                workload += (task["c"] * math.floor(t / task["t"]))

            last_tasks = []
            min_last_i = t
            for task in sorted([t for t in tasks], key=lambda  t: t["id"]):
                last_i = math.floor(t / task["t"]) * task["t"]
                if last_i < t:
                    last_tasks.append(task)
                if last_i < min_last_i:
                    min_last_i = last_i

            add_c = 0
            for task in sorted([t for t in tasks], key=lambda  t: t["id"]):
                add_c += task["c"]

            if min_last_i + add_c > t:
                return workload + (t - min_last_i)
            return workload + add_c

        d = task["d"]  # deadline
        c = task["c"]  # period
        n = d - get_empty_slots(d, tasks)  # slots left empty in [0, d]

        t = 1
        while True:
            t_tmp = n - c + 1 + get_workload(t, tasks)  # << --- removed +1
            if t_tmp == t:
                break
            t = t_tmp

        return t

    def allocation_test(cpu_capacity, tasks):
        # Verify resource constraint of pre-allocated tasks on each cpu
        cap = 0
        for task in tasks:
            cap += task["r"]
        if cap > cpu_capacity:
            return False

        # Verify schedubility if the resource constraints are ok
        if not verify_with_empty_slot(tasks)[0]:
            return False

        return True

    def get_precedences_list(rts):
        precedences = []
        for task in rts:
            if "p" in task:
                for p in task["p"]:
                    precedences.append((task["id"], p["id"], p["payload"]))
        return precedences

    def step4(cpu_stack, comm_stack, task_stack):
        """ For each processor 1, 2, . . . , n of the processor-stack, the communication-stack is polled top-down until
        finding the first pair containing a task allocated to this processor. The allocability test is performed on the
        companion task; if it passes the test, the task is definitely allocated to that processor and the pair deleted
        from the communicationstack. Each time a task is allocated to this processor the top-down polling of the whole
        communication-stack is repeated and candidate companion tasks tested in the same way, until allocating all
        candidates or until finding that no further tasks can be allocated to this processor. """
        for cpu in cpu_stack:
            while True:
                pair = None
                for c in comm_stack:
                    t1, t2 = c[0], c[1]

                    task_found = None
                    companion = None

                    # So we can easily check against t1 and t2
                    cpu_tasks_ids = [task["id"] for task in cpu["tasks"]]

                    # check if the pair contains a task allocated to the processor
                    if t1 in cpu_tasks_ids:
                        task_found = t1
                        companion = t2
                    else:
                        if t2 in cpu_tasks_ids:
                            task_found = t2
                            companion = t1

                    if task_found is not None:
                        # If the companion tasks is not assigned to another cpu
                        if companion in [t["id"] for t in task_stack]:

                            for task in task_stack:
                                if task["id"] == companion:
                                    companion_task = task

                            # Check if a replica is already allocated in to this cpu
                            if "rep" in companion_task:
                                if companion_task["rep"] in cpu_tasks_ids:
                                    break  # pass

                            # Run allocability test with the new task
                            cpu_task_list_tmp = cpu["tasks"][:]
                            cpu_task_list_tmp.append(companion_task)

                            # if allocable, add to the cpu tasks lists and remove from the task stack
                            if allocation_test(cpu["capacity"], cpu_task_list_tmp) is True:
                                cpu["tasks"].append(companion_task)
                                task_stack = [task for task in task_stack if task is not companion_task]
                                pair = c
                                break  # ends the search for a task
                        else:
                            # if companion is in the same processor, remove the pair from the communication stack
                            if companion in cpu_tasks_ids:
                                pair = c
                                break  # ends the search for a task

                # A communication pair was found and allocated -- remove it from the communication stack
                if pair is not None:
                    comm_stack.remove(pair)
                else:
                    break  # haven't found a comm pair

        return cpu_stack, comm_stack, task_stack

    def step6(cpu_stack, comm_stack, task_stack):
        """ The allocability test is performed on the first free task for allocation to the first, second, . . . , cpu
        until finding one to which it can be allocated. If no allocation is possible, Step 9 follows, else the task is
        deleted from the stack and Step 6 follows. """

        alloc_cnt = 0

        while True:
            alloc_cpu = None
            task_alloc = None

            for task in task_stack:
                for cpu in cpu_stack:
                    # Check if a replica is already allocated in to this cpu
                    if "rep" in task:
                        if task["rep"] in [t["id"] for t in cpu["tasks"]]:
                            continue  # pass

                    # Run allocability test with the new task
                    cpu_task_list_tmp = cpu["tasks"][:]
                    cpu_task_list_tmp.append(task)

                    # If allocable, add to the cpu tasks lists and remove from the task stack
                    if allocation_test(cpu["capacity"], cpu_task_list_tmp) is True:
                        cpu["tasks"].append(task)
                        task_alloc = task
                        alloc_cpu = cpu
                        alloc_cnt += 1
                        break

                if task_alloc:
                    break

            if task_alloc:
                # Remove the allocated task from the task-stack
                task_stack = [task for task in task_stack if task is not task_alloc]

                # As the task could be allocated, Step 7 follows -- which is Step 4 applied to the current cpu.
                while True:
                    pair = None

                    for c in comm_stack:
                        t1, t2 = c[0], c[1]

                        task_found = None
                        companion = None

                        # So we can easily check against t1 and t2
                        cpu_tasks_ids = [task_alloc["id"] for task in alloc_cpu["tasks"]]

                        # check if the pair contains a task allocated to the processor
                        if t1 in cpu_tasks_ids:
                            task_found = t1
                            companion = t2
                        else:
                            if t2 in cpu_tasks_ids:
                                task_found = t2
                                companion = t1

                        if task_found is not None:
                            # If the companion tasks is not assigned to another cpu
                            if companion in [t["id"] for t in task_stack]:

                                for task in task_stack:
                                    if task["id"] == companion:
                                        companion_task = task

                                # Check if a replica is already allocated in to this cpu
                                if "rep" in companion_task:
                                    if companion_task["rep"] in cpu_tasks_ids:
                                        break  # pass

                                # Run allocability test with the new task
                                cpu_task_list_tmp = alloc_cpu["tasks"][:]
                                cpu_task_list_tmp.append(companion_task)

                                # if allocable, add to the cpu tasks lists and remove from the task stack
                                if allocation_test(alloc_cpu["capacity"], cpu_task_list_tmp) is True:
                                    alloc_cpu["tasks"].append(companion_task)
                                    task_stack = [task for task in task_stack if task is not companion_task]
                                    pair = c
                                    break
                            else:
                                # if companion is in the same processor, remove the pair from the communication stack
                                if companion in alloc_cpu["tasks"]:
                                    pair = c
                                    break

                    # A communication pair was found and allocated -- remove it from the communication stack
                    if pair is not None:
                        comm_stack.remove(pair)
                    else:
                        break

            if task_alloc is None:
                break

        return alloc_cnt, cpu_stack, comm_stack, task_stack

    def step8(cpu_stack, comm_stack, task_graph):
        """ The schedulability of each processor containing sending tasks is reverified with deadlines corrected
        according to the precedence constraints """
        delta = 1
        for cpu in cpu_stack:
            for task in cpu["tasks"]:
                new_d = []
                if task["id"] in task_graph.nodes():
                    for successor in task_graph.successors(task["id"]):
                        other_cpu = 0
                        if successor not in [t["id"] for t in cpu["tasks"]]:
                            for cpu2 in cpu_stack:
                                if successor in [t["id"] for t in cpu2["tasks"]]:
                                    task_list = cpu2["tasks"]
                                    other_cpu = 1
                        else:
                            task_list = cpu["tasks"]

                        for succ_task in task_list:
                            if succ_task["id"] == successor:
                                successor_task = succ_task

                        task_list.sort(key=lambda t: (t["t"], t["id"]))  # sort by period (RM) -- lesser id max prio

                        l_s = last_starting_time(successor_task, task_list[:task_list.index(successor_task)])
                        new_d.append(l_s - 1 - (other_cpu * delta))

                    if new_d:
                        task["new_d"] = min(new_d)

        for cpu, cpu_data in cpus.items():
            if not allocation_test(cpu_data["capacity"], cpu_data["tasks"]):
                return False

        return True

    # Store results
    results = []

    # Assign pre-allocated tasks and remove them from the task-stack
    task_stack = []
    for task in rts:
        if "cpu" in task:
            cpu_id = task["cpu"]
            if "rep" in task:
                if task["rep"] in [t["id"] for t in cpus[cpu_id]["tasks"]]:
                    cpu_id = task["ncpu"]
            cpus[cpu_id]["tasks"].append(task)
        else:
            task_stack.append(task)

    # Perfoem allocability test on all pre-allocated tasks.
    for cpu, cpu_data in cpus.items():
        if not allocation_test(cpu_data["capacity"], cpu_data["tasks"]):
            print("Tasks assigned to cpu {0} are not schedulable --- the system is absolutely non-schedulable.".format(cpu))
            sys.exit(1)
    print("Preallocated tasks are schedulable.")

    precedences = get_precedences_list(rts)

    perm_cnt = 0
    tentative_cnt = 0
    valid_cnt = 0

    cpu_keys = list(cpus.keys())
    random.shuffle(cpu_keys)

    for cpu_stack_perm_t in itertools.permutations(cpu_keys):

        # The permutation generated by itertools is a tuple -- make it a list
        cpu_stack = [cpus[x] for x in cpu_stack_perm_t]

        # The task-stack is ordered by decreasing task utilization factors -- do not include preallocated tasks
        task_stack.sort(key=lambda task: task["uf"], reverse=True)

        # The communication-stack is assembled by ordering the communicating pairs by monotonic decreasing times of communication.
        comm_stack = precedences[:]
        comm_stack.sort(key=lambda t: t[2], reverse=True)

        # Perform the Step 4 of the heuristic.
        cpu_stack, comm_stack, task_stack = step4(cpu_stack, comm_stack, task_stack)

        alloc_cnt = 0

        while task_stack:
            # Now only a subset of free (non allocated) tasks remains, and the cpu-stack is reordered by increasing processor
            # utilization factors. This is the Step 5 of the heuristic.
            for cpu in cpu_stack:
                cpu["uf"] = sum([task["uf"] for task in cpu["tasks"]])
            cpu_stack.sort(key=lambda cpu: cpu["uf"])

            # Perform Step 6
            alloc_cnt, cpu_stack, comm_stack, task_stack = step6(cpu_stack, comm_stack, task_stack)

            # No allocations are possible
            if alloc_cnt == 0:
                break

        if alloc_cnt > 0:
            # A tentative solution has been found -- now is tested for precedence constraints
            task_graph = nx.DiGraph()
            task_graph.add_weighted_edges_from(precedences, weight='payload')

            valid_solution = step8(cpu_stack, comm_stack, task_graph)

            # Verify that the solution it is not a duplicate and save it
            if valid_solution:
                valid_cnt += 1

                # create a list of task tuples
                r1 = [tuple([t for t in cpus[k]["tasks"]]) for k in sorted(list(cpus.keys()))]

                if len(results) == 0:
                    results.append(r1)
                else:
                    count = 0
                    for r in results:
                        repeat = 0
                        for x in r1:
                            if x in r:
                                repeat += 1
                        if repeat < len(r):
                            count += 1
                    if count == len(results):
                        results.append(r1)

        perm_cnt += 1

        # Clear assignation
        for cpu, cpu_val in cpus.items():
            cpu_val["tasks"] = []

        # Assign pre-allocated tasks and remove them from the task-stack
        task_stack = []
        for task in rts:
            if "cpu" in task:
                cpu_id = task["cpu"]
                if "rep" in task:
                    if task["rep"] in [t["id"] for t in cpus[cpu_id]["tasks"]]:
                        cpu_id = task["ncpu"]
                cpus[cpu_id]["tasks"].append(task)
            else:
                task_stack.append(task)

    print("{0} permutations tested.".format(perm_cnt))
    print("{0} valid solutions found.".format(valid_cnt))

    # for idx, result in enumerate(results):
    #     print("Result", idx)
    #     for cpu_idx, cpu in enumerate(result):
    #         print("cpu {0}: ".format(cpu_idx), sorted([task["id"] for task in cpu]))
    #         for task in sorted(cpu, key=lambda t: t["id"]):
    #             print(task["id"], ": ", task["d"], task["new_d"] if "new_d" in task else " - ")

    # print tasks allocations
    for idx, result in enumerate(results):
        print("Result ", idx)
        result_t = []
        for cpu_idx, cpu in enumerate(result):
            cpu_mem = sum(t["r"] for t in cpu)
            result_t.append((cpu_idx, ', '.join(str(t["id"]) for t in cpu), "{0} ({1:.2f} %)".format(cpu_mem, (cpu_mem / cpus[cpu_idx]["capacity"])*100)))
        print(tabulate(result_t, ["# cpu", "tasks", "mem"], "psql"))


def get_args():
    """ Command line arguments """
    parser = ArgumentParser(description="Generate CPPs")
    parser.add_argument("xmlfile", help="XML file with RTS", type=str)
    parser.add_argument("xmlid", help="STR in file to process", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.isfile(args.xmlfile):
        print("Can't find {0} XML file.".format(args.xmlfile))
        sys.exit(1)

    rts = get_rts_from_xmlfile(args.xmlid, args.xmlfile)
    #rts, _ = get_rts_from_pot_file("graph1.dot")

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

    # print cpu info
    print(tabulate([(k, v["capacity"]) for k,v in cpus.items()], ["cpu", "memory"], "psql"))

    # run the heuristic method
    heuristic(rts, cpus)


if __name__ == '__main__':
    main()
