def verify_with_empty_slot(tasks):
    import math

    def verify_uf_sum(tasks):
        summation = 0
        for task in tasks:
            summation += (task["c"] /task["t"])
        return summation < 1

    def get_workload(t, tasks):
        workload = 0
        for task in tasks:
            workload += ( task["c"] * math.ceil(t / task["t"]) )
        return workload

    def get_last_empty_slot(task, tasks):
        t = 0
        while True:
            t_tmp = task["c"] + get_workload(t, tasks)
            if t_tmp == t:
                break
            t = t_tmp
        return t

    schedulable = True
    empty_slots = []

    for i, task in enumerate(tasks[1:], 1):
        empty_slot = get_last_empty_slot(task, tasks[:i])
        empty_slots.append(empty_slot)
        schedulable = verify_uf_sum(tasks[:i]) and (task["d"] >= empty_slot)
        if not schedulable:
            break

    return [schedulable, empty_slots]


def last_starting_time(task, tasks):
    """ Calculate the last possible starting time for executing task """
    import math

    def get_workload(t, tasks):
        """ Calculate the workload required for tasks in [0,t] """
        workload = 0
        for task in tasks:
            workload += (task["c"] * math.ceil(t / task["t"]))
        return workload

    def get_empty_slots(d, tasks):
        """ Calculates the amount of empty slots left by the execution of tasks in [0, d] """
        t = 0
        e = 1
        missed = False
        while True:
            while True:
                t_tmp = e + get_workload(t, tasks)
                if t_tmp > d:
                    missed = True
                    break  # inner while
                if t_tmp == t:
                    break  # inner while
                t = t_tmp
            if missed:
                e -= 1
                break  # while
            e += 1
        return e

    d = task["new_d"] if "new_d" in task else task["d"]  # deadline
    c = task["c"]  # period
    n = get_empty_slots(d, tasks)  # slots left empty in [0, d]

    t = 1
    while True:
        t_tmp = n - c + 1 + get_workload(t, tasks)  # << --- removed +1
        if t_tmp == t:
            break
        t = t_tmp

    return t