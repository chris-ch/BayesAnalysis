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
    sides_6 = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')
    ps = probaspace.make_space_full(sides_6)
    logging.info(str(ps.sigma_algebra.items))
    for item in ps.sigma_algebra.items:
        logging.info('{}: {}'.format(item, ps.probability.evaluate(item)))

    logging.info('----------- combined CDF sum 2x d6')
    dist_2x_die6 = probaspace.MixedDistributionFunction(rv_die6, rv_die6, mix_func=sum, sigma_algebra=sa)
    for val in (float(i)/ 5. for i in range(0, 65)):
        logging.info('cdf({}) = {}'.format(val, dist_2x_die6.evaluate(val)))

    logging.info('dist_2x_die6: {}'.format(probaspace.make_probability_density(dist_2x_die6, 0., 12.5, 65)))

    logging.info('----------- combined CDF max 4x sum 2x d6')
    
    dist_max_4x_2x_die6 = probaspace.MixedDistributionFunction(dist_2x_die6, dist_2x_die6, dist_2x_die6, dist_2x_die6, mix_func=max)
    logging.info('dist_2x_die6: {}'.format(probaspace.make_probability_density(dist_max_4x_2x_die6, 0., 12.5, 65)))

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
