from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class FastSimulation:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.duration, self.n_isects, self.n_streets, self.n_cars, self.bonus_points = \
                (int(tok) for tok in file.readline().split())

            print(f'  duration:     {self.duration}')
            print(f'  n_isects:     {self.n_isects}')
            print(f'  n_streets:    {self.n_streets}')
            print(f'  n_cars:       {self.n_cars}')
            print(f'  bonus points: {self.bonus_points}')

            self.street_name = np.empty(self.n_streets, dtype=object)
            # self.street_isect_beg = np.empty(self.n_streets, dtype=int)
            self.street_isect = np.empty(self.n_streets, dtype=int)
            self.street_duration = np.empty(self.n_streets, dtype=int)

            street_id_by_name = dict()
            for street_id in range(self.n_streets):
                tok = file.readline().split()
                # bed_isect_id = int(tok[0])
                end_isect_id = int(tok[1])
                name = tok[2]
                duration = int(tok[3])
                self.street_isect[street_id] = end_isect_id
                self.street_name[street_id] = name
                self.street_duration[street_id] = duration
                street_id_by_name[name] = street_id

            self.paths = list()
            for car_id in range(self.n_cars):
                street_names = file.readline().split()[1:]
                path_street_ids = [street_id_by_name[street_name] for street_name in street_names]
                self.paths.append(path_street_ids)

        all_path_street_ids = list()  # street ids for all streets in paths
        self.car_path_start_idx = list()  # path start indices for all cars
        for path_street_ids in self.paths:
            self.car_path_start_idx.append(len(all_path_street_ids))
            all_path_street_ids.extend(path_street_ids + [-1])
        self.all_paths_street_id = np.array(all_path_street_ids)

        street_ids_with_cars = set()
        for path_street_ids in self.paths:
            street_ids_with_cars.update(path_street_ids[:-1])
        self.street_has_cars = np.zeros(self.n_streets, dtype=bool)
        self.street_has_cars[np.array(list(street_ids_with_cars))] = True

        self.green_matrix = np.zeros((self.n_streets, self.duration), dtype=bool)

        self.n_arrivals = None
        self.arrival_times = None

    def set_initial_schedule(self):
        self.green_matrix[:] = False
        for isect_id in range(self.n_isects):
            street_ids = np.where((self.street_isect == isect_id) & self.street_has_cars)[0]
            # street_ids = np.where((self.street_isect == isect_id))[0]
            schedule_duration = len(street_ids)
            for i, street_id in enumerate(street_ids):
                self.green_matrix[street_id, i::schedule_duration] = True

    def optimize_schedule(self, isect_id: int, approx: bool):
        street_in_sched = (self.street_isect == isect_id) & self.street_has_cars
        schedule_duration = street_in_sched.sum()

        # --- calculate wait times of cars with the current arrival times for each starting time of the green lights ---
        # wait_times[i,j]
        # is the total wait time for the cars at street i if the lights are green at the time steps j+n*schedule_duration
        # with the current arrivals
        wait_times = np.empty((schedule_duration, schedule_duration))


        if approx:
            arrival_times = self.arrival_times[street_in_sched]
            for i in range(schedule_duration):
                for j in range(schedule_duration):
                    wait_times[i,j] = sum((j-at) % schedule_duration for at in arrival_times[i])
        else:
            n_arrivals = self.n_arrivals[street_in_sched, :]
            for t_offset in range(schedule_duration):
                wait_times_j = np.zeros(schedule_duration, dtype=int)
                n_cars_waiting = np.zeros(schedule_duration, dtype=int)
                for t in range(self.duration):
                    n_cars_waiting += n_arrivals[:, t]
                    if t % schedule_duration == t_offset:
                        n_cars_waiting[n_cars_waiting > 0] -= 1
                    wait_times_j += n_cars_waiting
                wait_times[:, t_offset] = wait_times_j

        _, t_offsets = linear_sum_assignment(wait_times)

        self.green_matrix[street_in_sched, :] = False
        street_ids = np.where(street_in_sched)[0]
        for i, street_id in enumerate(street_ids):
            self.green_matrix[street_id, t_offsets[i]::schedule_duration] = True

    def simulate(self):
        # stats for improving schedule
        self.n_arrivals = np.zeros((self.n_streets, self.duration),
                                   dtype=np.int8)  # number of cars arriving at end of street
        self.arrival_times = [list() for _ in range(self.n_streets)]

        arrivals_by_time = [list() for _ in
                            range(self.duration)]  # arriving at end of street of corresponding path index
        arrivals_by_time[0].extend(self.car_path_start_idx)
        street_queues = [deque() for _ in range(self.n_streets)]
        street_queues_nonempty = np.zeros(self.n_streets, dtype=bool)

        arrived_cars = 0
        score = 0
        for t in tqdm(range(self.duration)):
            # enqueue arrivals at end of street
            for path_idx in sorted(arrivals_by_time[t]):
                if self.all_paths_street_id[path_idx + 1] == -1:
                    # arrived at end of path
                    score += self.bonus_points + (self.duration - t)
                    arrived_cars += 1
                else:
                    street_id = self.all_paths_street_id[path_idx]
                    next_path_idx = path_idx + 1
                    street_queues[street_id].append(next_path_idx)
                    street_queues_nonempty[street_id] = True

                    self.n_arrivals[street_id, t] += 1
                    self.arrival_times[street_id].append(t)

            # pop from queues where cars can drive
            can_drive = street_queues_nonempty & self.green_matrix[:, t]
            for street_id in np.where(can_drive)[0]:
                next_path_idx = street_queues[street_id].popleft()
                if len(street_queues[street_id]) == 0:
                    street_queues_nonempty[street_id] = False

                next_arrival_time = t + self.street_duration[self.all_paths_street_id[next_path_idx]]
                if next_arrival_time < self.duration:
                    arrivals_by_time[next_arrival_time].append(next_path_idx)

        self.arrival_times = np.array(self.arrival_times, dtype=object)

        return score, arrived_cars
