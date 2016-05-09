# Simple simulation of a RM-scheduled mono-processor system

import simpy
import math
import sys


def joseph_wcrt(rts):
    """ Verify schedulability """
    wcrt = [0] * len(rts)
    schedulable = True
    wcrt[0] = rts[0]["c"]
    for i, task in enumerate(rts[1:], 1):
        r = 1
        c, t, d = task["c"], task["t"], task["d"]
        while schedulable:
            w = 0
            for taskp in rts[:i]:
                cp, tp = taskp["c"], taskp["t"]
                w += math.ceil(r / tp) * cp
            w = c + w
            if r == w:
                break
            r = w
            if r > d:
                schedulable = False
        wcrt[i] = r
        if not schedulable: break
    return [schedulable, wcrt]


class Task:
    def __init__(self, env, cpu, name, wcet, period, priority):
        self.env = env
        self.name = name
        self.period = period
        self.wcet = wcet
        self.instance_count = 0
        self.priority = priority
        self.absolute_deadline = period

        # Start "working"
        self.process = env.process(self.work(cpu))

    def work(self, cpu):
        while True:
            release_time = self.env.now
            self.absolute_deadline = release_time + self.period
            remain_c = self.wcet  # the runtime counter is replenished

            print("{0}: {1} ({2}) release".format(release_time, self.name, self.instance_count))

            while remain_c:
                try:
                    with cpu.request(priority=self.priority) as proc:
                        yield proc  # take the cpu
                        start = self.env.now
                        print("{0}: {1} ({2}) start".format(start, self.name, self.instance_count))
                        yield self.env.timeout(remain_c)  # go to sleep with the cpu taken

                        remain_c = 0  # set to 0 to exit while loop.

                        if self.env.now > self.absolute_deadline:
                            print("{0} ({1}) missed a deadline!".format(self.name, self.instance_count))
                            self.env.exit()

                except simpy.Interrupt:
                    remain_c -= self.env.now - start  # execution time left
                    if remain_c > 0:
                        print("{0}: {1} ({2}) preemted".format(self.env.now, self.name, self.instance_count))

            print("{0}: {1} ({2}) ends".format(self.env.now, self.name, self.instance_count))

            self.instance_count += 1  # increments the instance counter

            # block until next period
            yield self.env.timeout((release_time + self.period) - self.env.now)


def main():
    env = simpy.Environment()
    cpu = simpy.PreemptiveResource(env, capacity=1)

    rts = [{"id":1, "c":1, "t":3, "d":3}, {"id":2, "c":1, "t":5, "d":5}, {"id":3, "c":2, "t":6, "d":6}]

    schedtest = joseph_wcrt(rts)
    if schedtest[0] is False:
        print("Not schedulable")
        sys.exit(1)

    for priority, task in enumerate(rts):
        Task(env, cpu, "Task " + str(task["id"]), task["c"], task["t"], priority)

    env.run(until=13)


if __name__ == '__main__':
    main()

