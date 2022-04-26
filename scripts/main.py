import logging
import math
import os
from typing import Callable
import probaspace
import bayes
from pprint import pprint


def smarties94(event: probaspace.Event) -> float:
    probs = {probaspace.Event('R'): 10, probaspace.Event('G'): 10, probaspace.Event('B'): 20}
    return float(probs[event]) / float(sum(probs.values()))


def smarties96(event: probaspace.Event) -> float:
    probs = {probaspace.Event('R'): 15, probaspace.Event('G'): 10, probaspace.Event('B'): 15}
    return float(probs[event]) / float(sum(probs.values()))


def main():
    universe = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')
    sa = probaspace.make_sigma_algebra_full(universe)
    logging.info(str(sa.items))
    proba = probaspace.Probability(sa)

    for item in sa.items:
        logging.info('{}: {}'.format(item, proba.evaluate(item)))

    def die(item: probaspace.Event) -> int:
        return int(str(item))

    rv = probaspace.RandomVariable('rv', event_mapper=die, universe=universe)
    dist1 = probaspace.SimpleCumulativeDistributionFunction(rv, sa)
    for val in (float(i) / 5. for i in range(0, 35)):
        logging.info('cdf({}) = {}'.format(val, dist1.evaluate(val)))

    logging.info('----------- combined CDF')
    dist2 = probaspace.CombinedCumulativeDistributionFunction(rv, rv, sa)
    for val in (float(i)/ 5. for i in range(0, 65)):
        logging.info('cdf({}) = {}'.format(val, dist2.evaluate(val)))

    logging.info('----------- PDF 1')
    logging.info('{}'.format(dist1.make_probability_density(0., 6.5, 0.2)))
    logging.info('----------- PDF 2')
    logging.info('{}'.format(dist2.make_probability_density(0., 12.5, 0.2)))

    #logging.info('----------- probability mass')
    #pm_d6 = probaspace.CumulativeDistributionFunction(rv, sa)
    #logging.info('data: {}'.format(pm_d6))
    #logging.info('simul: {}'.format(str(pm_d6.runs(10000))))

    #logging.info('----------- probability mass sum')
    #pm_sum_3d6 = probaspace.ProbabilityMassMixed(pm_d6, pm_d6, pm_d6, mix_func=sum)
    #logging.info('data: {}'.format(pm_sum_3d6))
    #logging.info('simul: {}'.format(str(pm_sum_3d6.runs(10000))))

    logging.info('----------- probability mass max')
    #pm_max_3d6_5 = probaspace.ProbabilityMassMixed(pm_sum_3d6, pm_sum_3d6, pm_sum_3d6, pm_sum_3d6, pm_sum_3d6, mix_func=max)
    logging.info('----------- created')
    #logging.info('data: {}'.format(pm_max_3d6_5))
    logging.info('----------- computed')
    #logging.info('simul: {}'.format(str(pm_max_3d6_5.runs(10000))))
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
