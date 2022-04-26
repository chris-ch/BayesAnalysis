import logging
import os
import unittest
from typing import Callable

import probaspace
import bayes


def dice(count_side: int) -> Callable[[probaspace.Event], float]:
    def likelihood(event: probaspace.Event):
        if int(event.name) > count_side:
            return 0.
        else:
            return 1. / float(count_side)

    return likelihood


class ProbaspaceTest(unittest.TestCase):

    def test_dice(self):
        estimator = bayes.BayesAnalysis()
        #
        # estimator.create_hypothesis(Hypothesis("Smarties 94", smarties94), 0.5)
        # estimator.create_hypothesis(Hypothesis("Smarties 96", smarties96), 0.5)
        # estimator.add_event('R')
        # estimator.add_event('G')
        sides = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')
        d4 = sides.create_random_variable_single("Dice 4", likelihood=dice(4))
        d6 = sides.create_random_variable_single("Dice 6", likelihood=dice(6))
        d8 = sides.create_random_variable_single("Dice 8", likelihood=dice(8))
        d12 = sides.create_random_variable_single("Dice 12", likelihood=dice(12))
        d20 = sides.create_random_variable_single("Dice 20", likelihood=dice(20))

        estimator.define_uninformed(sides, d4, d6, d8, d12, d20)

        estimator.add_event(probaspace.Event('3'))
        estimator.add_event(probaspace.Event('4'))
        estimator.add_event(probaspace.Event('8'))
        estimator.add_event(probaspace.Event('3'))
        estimator.add_event(probaspace.Event('1'))
        self.assertAlmostEqual(estimator.evaluate(d12), 0.11532016915)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler('{}.log'.format(os.path.basename(__file__).split('.')[0]), mode='w')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    unittest.main()
