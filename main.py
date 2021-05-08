from datetime import datetime

from sim import Simulation


def main():
    filename = './data/hashcode.in'
    print('Reading simulation...')

    sim = Simulation(filename)

    street_ids_with_cars = set()
    for car in sim.cars:
        street_ids_with_cars.update([s.id for s in car.path[:-1]])

    for isect in sim.intersections:
        isect.set_schedule([(s, 1) for s in isect.queues.keys() if s in street_ids_with_cars], sim.duration)

    time_beg = datetime.now()

    score, arrivals = sim.simulate()

    time_end = datetime.now()
    print(time_end - time_beg)

    print(f'score: {score}')

    sim.write_schedule('./data/submission.csv')

    # pd.Series([len(isect.queues) for isect in sim.intersections]).hist()
    # plt.show()


if __name__ == '__main__':
    main()
