from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


class IntersectionStats:
    def __init__(self):
        self.arrival_times = list()
        self.wait_times = list()


class Intersection:
    def __init__(self, identifier: int):
        self.id = identifier
        self.queues = dict()  # car queues for incoming streets by street id
        self.stats = dict()  # statistics for incoming streets by street id
        self.t_last_drive = -1  # time when last car drove through intersection
        self.schedule = None
        self.green_street_ids = None

    def reset(self):
        self.queues = {k: deque() for k in self.queues.keys()}
        self.stats = {k: IntersectionStats() for k in self.stats.keys()}
        self.t_last_drive = -1

    def add_street(self, street_id: int):
        self.queues[street_id] = deque()
        self.stats[street_id] = IntersectionStats()

    def car_can_drive(self, t: int, street_id: int, car: 'Car', ignore_green=False):
        """Whether the given car waiting at the given street is allowed to drive at time t."""
        return self.t_last_drive < t and self.queues[street_id][0] == car and (
                ignore_green or self.green_street_ids[t] == street_id)

    def push_car(self, t: int, street_id: int, car: 'Car'):
        """Add a car to the queue."""
        self.queues[street_id].append(car)
        self.stats[street_id].arrival_times.append(t)

    def pop_car(self, t: int, street_id: int):
        """Pop a car from the queue."""
        car = self.queues[street_id].popleft()
        self.t_last_drive = t
        self.stats[street_id].wait_times.append(t - car.get_active_arrival_time())

    def set_schedule(self, schedule: List[Tuple[int, int]], sim_duration: int):
        """Set schedule.
        Args:
            schedule: list of tuple (street_id, duration)
            sim_duration: total duration of simulation
        """
        self.schedule = schedule
        self.green_street_ids = np.empty(sim_duration, dtype=int)
        schedule_duration = sum(duration for _, duration in schedule)
        t = 0
        for street_id, duration in schedule:
            for t_i in range(duration):
                self.green_street_ids[t + t_i::schedule_duration] = street_id
                t += duration


@dataclass
class Street:
    name: str
    id: int
    start: Intersection
    dest: Intersection
    duration: int


class Car:
    def __init__(self, identifier: int, path: List[Street]):
        self.id = identifier
        self.path = path
        self.reset()

    def reset(self):
        self.idx_active_street = 0  # index of street the car is currently on
        self.intersection = None  # intersection the car is currently waiting on (None while car is on raod)
        self.arrival_times = np.zeros(len(self.path))  # time the car arrived at each end intersection
        self.wait_times = np.zeros(len(self.path))  # time the car waited at each end intersection
        self.arrived = False
        self.score = 0

    def get_active_street(self):
        """Get street the car is currently on."""
        return self.path[self.idx_active_street]

    def get_active_arrival_time(self):
        return self.arrival_times[self.idx_active_street]

    def is_on_last_street(self):
        """Return whether car is on its last street."""
        return self.idx_active_street == len(self.path) - 1

    def drive_into_next_street(self, t):
        """Drive into the next street."""
        self.wait_times[self.idx_active_street] = t - self.arrival_times[self.idx_active_street]
        self.idx_active_street += 1
        self.arrival_times[self.idx_active_street] = t + self.get_active_street().duration
        self.intersection = None


class Simulation:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.duration, n_isects, n_streets, n_cars, self.bonus_points = (int(tok) for tok in
                                                                             file.readline().split())
            print(f'  duration:     {self.duration}')
            print(f'  n_isects:     {n_isects}')
            print(f'  n_streets:    {n_streets}')
            print(f'  n_cars:       {n_cars}')
            print(f'  bonus points: {self.bonus_points}')

            self.intersections = list()
            for isect_id in range(n_isects):
                self.intersections.append(
                    Intersection(isect_id)
                )

            streets_by_name = dict()
            self.streets = list()
            for street_id in range(n_streets):
                tok = file.readline().split()
                ibeg = int(tok[0])
                iend = int(tok[1])
                name = tok[2]
                dur = int(tok[3])
                street = Street(name=name,
                                id=street_id,
                                start=self.intersections[ibeg],
                                dest=self.intersections[iend],
                                duration=dur)
                self.streets.append(street)
                streets_by_name[name] = street
                self.intersections[iend].add_street(street_id)

            self.cars = list()
            for car_id in range(n_cars):
                street_names = file.readline().split()[1:]
                self.cars.append(Car(car_id, [streets_by_name[street_name] for street_name in street_names]))

    def write_schedule(self, filename):
        isects_with_nonempty_schedule = [isect for isect in self.intersections if len(isect.schedule) > 0]
        with open(filename, 'w') as file:
            file.write(f'{len(isects_with_nonempty_schedule)}\n')
            for isect in isects_with_nonempty_schedule:
                file.write(f'{isect.id}\n')
                file.write(f'{len(isect.schedule)}\n')
                for street_id, duration in isect.schedule:
                    file.write(f'{self.streets[street_id].name} {duration}\n')

    def simulate(self, ignore_green=False):
        for isect in self.intersections:
            isect.reset()
        for car in self.cars:
            car.reset()

        score = 0
        for t in tqdm(range(self.duration)):
            for car in self.cars:
                if car.arrived:
                    continue

                street = car.get_active_street()
                if car.get_active_arrival_time() == t:
                    if car.is_on_last_street():
                        car_score = self.bonus_points + (self.duration - t)
                        score += car_score
                        car.score = car_score
                        car.arrived = True
                        continue
                    else:
                        street.dest.push_car(t, street.id, car)
                        car.intersection = street.dest

                isect = car.intersection
                if isect is not None and isect.car_can_drive(t, street.id, car, ignore_green):
                    isect.pop_car(t, street.id)
                    car.drive_into_next_street(t)

        return score, sum(car.arrived for car in self.cars)
