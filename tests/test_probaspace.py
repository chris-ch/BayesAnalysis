import logging
import os
import unittest
from typing import Callable

import probaspace
import bayes


class ProbaSpaceTest(unittest.TestCase):

    def test_dice(self):
        estimator = bayes.BayesAnalysis()

        def dice(count_side: int) -> Callable[[probaspace.Event], float]:
            def likelihood(event: probaspace.Event):
                if int(event.name) > count_side:
                    return 0.
                else:
                    return 1. / float(count_side)

            return likelihood
        sides = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')
        ps = probaspace.make_space_full(sides)
        d4 = probaspace.make_random_variable_single("Dice 4", event_mapper=dice(4), space=ps)
        d6 = probaspace.make_random_variable_single("Dice 6", event_mapper=dice(6), space=ps)
        d8 = probaspace.make_random_variable_single("Dice 8", event_mapper=dice(8), space=ps)
        d12 = probaspace.make_random_variable_single("Dice 12", event_mapper=dice(12), space=ps)
        d20 = probaspace.make_random_variable_single("Dice 20", event_mapper=dice(20), space=ps)
        estimator.define_uninformed(sides, d4, d6, d8, d12, d20)
        estimator.add_event(probaspace.Event('3'))
        estimator.add_event(probaspace.Event('4'))
        estimator.add_event(probaspace.Event('8'))
        estimator.add_event(probaspace.Event('3'))
        estimator.add_event(probaspace.Event('1'))
        self.assertAlmostEqual(estimator.evaluate(d12), 0.11532016915)

    def test_simple_proba(self):
        sides_6 = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')

        def die(item: probaspace.Event) -> int:
            return int(str(item))

        ps = probaspace.make_space_full(sides_6)
        rv_die6 = probaspace.RandomVariable('rv', event_mapper=die, space=ps)

        dist_die6 = probaspace.SimpleDistributionFunction(rv_die6)
        self.assertAlmostEqual(dist_die6.evaluate(3.).value, 0.5)
        self.assertAlmostEqual(dist_die6.evaluate(-1.).value, 0.)
        self.assertAlmostEqual(dist_die6.evaluate(0.).value, 0.)
        self.assertAlmostEqual(dist_die6.evaluate(1.).value, 0.1666666667)
        self.assertAlmostEqual(dist_die6.evaluate(2.).value, 0.3333333333)
        self.assertAlmostEqual(dist_die6.evaluate(4.).value, 0.6666666667)
        self.assertAlmostEqual(dist_die6.evaluate(5.).value, 0.8333333333)
        self.assertAlmostEqual(dist_die6.evaluate(6.).value, 1.0)
        self.assertAlmostEqual(dist_die6.evaluate(7.).value, 1.0)

        pdf = probaspace.make_probability_density(dist_die6, 0., 6., 30)
        self.assertAlmostEqual(pdf[1.0].value, 0.1666666667)
        self.assertAlmostEqual(pdf[2.0].value, 0.1666666667)
        self.assertAlmostEqual(pdf[3.0].value, 0.1666666667)
        self.assertAlmostEqual(pdf[4.0].value, 0.1666666667)
        self.assertAlmostEqual(pdf[5.0].value, 0.1666666667)
        self.assertAlmostEqual(pdf[6.0].value, 0.1666666667)

    def test_joint_proba(self):
        sides_6 = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')

        def die(item: probaspace.Event) -> int:
            return int(str(item))

        sa = probaspace.SigmaAlgebra(sides_6, probaspace.powerset(sides_6.events))
        ps = probaspace.ProbabilitySpace(sa, probaspace.JointProbability(sa))
        rv_die6 = probaspace.RandomVariable('rv', event_mapper=die, space=ps)

        joint = probaspace.JointDistributionFunction(rv_die6, rv_die6)
        self.assertAlmostEqual(joint.evaluate(4., 4.).value, 0.44444444)
        self.assertAlmostEqual(joint.evaluate(2., 1.).value, 0.05555556)
        self.assertAlmostEqual(joint.evaluate(1., 1.).value, 0.02777778)
        self.assertAlmostEqual(joint.evaluate(1., 3.).value, 0.08333333)
        self.assertAlmostEqual(joint.evaluate(5., 2.).value, 0.27777778)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler('{}.log'.format(os.path.basename(__file__).split('.')[0]), mode='w')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    unittest.main()
