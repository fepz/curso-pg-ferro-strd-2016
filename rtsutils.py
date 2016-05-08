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