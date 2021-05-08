from datetime import datetime

from tqdm import tqdm

from sim_fast import FastSimulation


def main():
    filename = './data/hashcode.in'
    print('Reading simulation...')

    sim = FastSimulation(filename)

    sim.set_initial_schedule()

    time_beg = datetime.now()

    score, arrivals = sim.simulate()

    time_end = datetime.now()
    print(time_end - time_beg)

    print(f'score: {score}')

    for i in range(10):
        for isect in tqdm(range(sim.n_isects)):
            sim.optimize_schedule(isect, approx=True)

        score, arrivals = sim.simulate()
        print(f'\nscore after optimization iteration {i+1}: {score}\n')


if __name__ == '__main__':
    main()
