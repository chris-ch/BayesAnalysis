import logging
import os
from typing import Callable
import probaspace
import bayes


def smarties94(event: probaspace.Event) -> float:
    probs = {probaspace.Event('R'): 10, probaspace.Event('G'): 10, probaspace.Event('B'): 20}
    return float(probs[event]) / float(sum(probs.values()))


def smarties96(event: probaspace.Event) -> float:
    probs = {probaspace.Event('R'): 15, probaspace.Event('G'): 10, probaspace.Event('B'): 15}
    return float(probs[event]) / float(sum(probs.values()))


def dice(count_side: int) -> Callable[[probaspace.Event], float]:
    def likelihood(event: probaspace.Event):
        if int(event.name) > count_side:
            return 0.
        else:
            return 1. / float(count_side)

    return likelihood


def main():
    estimator = bayes.BayesAnalysis()
    #
    # estimator.create_hypothesis(Hypothesis("Smarties 94", smarties94), 0.5)
    # estimator.create_hypothesis(Hypothesis("Smarties 96", smarties96), 0.5)
    # estimator.add_event('R')
    # estimator.add_event('G')
    sides = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')
    d4 = sides.create_random_variable("Dice 4", likelihood=dice(4))
    d6 = sides.create_random_variable("Dice 6", likelihood=dice(6))
    d8 = sides.create_random_variable("Dice 8", likelihood=dice(8))
    d12 = sides.create_random_variable("Dice 12", likelihood=dice(12))
    d20 = sides.create_random_variable("Dice 20", likelihood=dice(20))

    estimator.define_uninformed(d4, d6, d8, d12, d20)

    estimator.add_event(probaspace.Event('3'))
    estimator.add_event(probaspace.Event('4'))
    estimator.add_event(probaspace.Event('8'))
    estimator.add_event(probaspace.Event('3'))
    estimator.add_event(probaspace.Event('1'))
    logging.info(estimator)

    universe = probaspace.Universe([probaspace.Event(index) for index in ['1', '2', '3', '4', '5', '6']])
    sa = probaspace.make_power_set(universe)
    logging.info(str(sa.items))
    proba = probaspace.Probability(sa)

    for item in sa.items:
        logging.info('{}: {}'.format(item, proba.value(item)))

    def die(item: probaspace.Event) -> int:
        return int(str(item))

    rv = probaspace.RandomVariable('rv', item_mapper=die, universe=universe)
    dist1 = probaspace.CumulativeDistributionFunction(rv, sa)
    for val in (float(i) / 5. for i in range(0, 35)):
        logging.info('cdf({}) = {}'.format(val, dist1.value(val)))

    logging.info('----------- combined CDF')
    dist2 = probaspace.CombineSumCDF(rv, rv, sa)
    for val in (float(i)/ 5. for i in range(0, 65)):
        logging.info('cdf({}) = {}'.format(val, dist2.value(val)))

    logging.info('----------- probability mass')
    pm_d6 = probaspace.ProbabilityMassUniform(rv)
    logging.info('data: {}'.format(pm_d6.buckets))
    logging.info('simul: {}'.format(str(pm_d6.runs(10000))))

    logging.info('----------- probability mass sum')
    pm_sum_3d6 = probaspace.ProbabilityMassMixed(pm_d6, pm_d6, pm_d6, mix_func=sum)
    logging.info('data: {}'.format(pm_sum_3d6))
    logging.info('simul: {}'.format(str(pm_sum_3d6.runs(10000))))

    logging.info('----------- probability mass max')
    pm_max_3d6_5 = probaspace.ProbabilityMassMixed(pm_sum_3d6, pm_sum_3d6, pm_sum_3d6, pm_sum_3d6, pm_sum_3d6, mix_func=max)
    logging.info('----------- created')
    logging.info('data: {}'.format(pm_max_3d6_5))
    logging.info('----------- computed')
    logging.info('simul: {}'.format(str(pm_max_3d6_5.runs(10000))))
    #pmm_simul = probaspace.simul()
    #pmm = probaspace.ProbabilityMassMixed(pms, pms, pms, pms, pms, mix_func=max)
    #logging.info('data: {}'.format(pmm))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler('{}.log'.format(os.path.basename(__file__).split('.')[0]), mode='w')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    main()
