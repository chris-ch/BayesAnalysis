import logging
import os
import unittest
from collections import defaultdict
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

    def test_biased_proba(self):
        coin = probaspace.Universe.from_labels('H', 'T')

        def coin_value(item: probaspace.Event) -> int:
            if item.name == 'H':
                return 0
            else:
                return 1

        ps1 = probaspace.make_space_full(coin)
        rv_coin1 = probaspace.RandomVariable('rv1', event_mapper=coin_value, space=ps1)
        dist_coin1 = probaspace.SimpleDistributionFunction(rv_coin1)
        self.assertAlmostEqual(dist_coin1.evaluate(-0.1).value, 0.0)
        self.assertAlmostEqual(dist_coin1.evaluate(0.1).value, 0.5)
        self.assertAlmostEqual(dist_coin1.evaluate(0.5).value, 0.5)
        self.assertAlmostEqual(dist_coin1.evaluate(0.9).value, 0.5)
        self.assertAlmostEqual(dist_coin1.evaluate(1.).value, 1.0)
        self.assertAlmostEqual(dist_coin1.evaluate(1.1).value, 1.0)

        sa = probaspace.SigmaAlgebra(coin, probaspace.powerset(coin.events))

        def probabilities(sai: probaspace.SigmaAlgebraItem):
            output = 0.
            if sai.set() == {probaspace.Event('H')}:
                output = 0.2

            elif sai.set() == {probaspace.Event('T')}:
                output = 0.8

            elif sai.set() == {probaspace.Event('T'), probaspace.Event('H')}:
                output = 1.0
            
            return probaspace.UnitRangeValue(output)
            
        ps2 = probaspace.ProbabilitySpace(sa, probaspace.Probability(sa, probabilities))
        rv_coin2 = probaspace.RandomVariable('rv2', event_mapper=coin_value, space=ps2)
        dist_coin2 = probaspace.SimpleDistributionFunction(rv_coin2)
        self.assertAlmostEqual(dist_coin2.evaluate(-0.1).value, 0.0)
        self.assertAlmostEqual(dist_coin2.evaluate(0.1).value, 0.2)
        self.assertAlmostEqual(dist_coin2.evaluate(0.5).value, 0.2)
        self.assertAlmostEqual(dist_coin2.evaluate(0.9).value, 0.2)
        self.assertAlmostEqual(dist_coin2.evaluate(1.).value, 1.0)
        self.assertAlmostEqual(dist_coin2.evaluate(1.1).value, 1.0)

    def test_joint_proba(self):
        sides_6 = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')

        def die(item: probaspace.Event) -> int:
            return int(str(item))

        sa = probaspace.SigmaAlgebra(sides_6, probaspace.powerset(sides_6.events))
        ps = probaspace.ProbabilitySpace(sa, probaspace.UnbiasedProbability(sa))
        rv_die6 = probaspace.RandomVariable('rv', event_mapper=die, space=ps)

        joint = probaspace.JointDistributionFunction(rv_die6, rv_die6)
        self.assertAlmostEqual(joint.evaluate(4., 4.).value, 0.44444444)
        self.assertAlmostEqual(joint.evaluate(2., 1.).value, 0.05555556)
        self.assertAlmostEqual(joint.evaluate(1., 1.).value, 0.02777778)
        self.assertAlmostEqual(joint.evaluate(1., 3.).value, 0.08333333)
        self.assertAlmostEqual(joint.evaluate(5., 2.).value, 0.27777778)

        self.assertAlmostEqual(rv_die6.expectancy(), 3.5)
        self.assertAlmostEqual(rv_die6.variance(), 2.91666667)

    def test_double_dice(self):

        def die(item: probaspace.Event) -> int:
            return int(str(item))

        sides_6 = probaspace.Universe.from_labels('1', '2', '3', '4', '5', '6')
        ps = probaspace.make_space_full(sides_6)
        rv_die6 = probaspace.RandomVariable('rv', event_mapper=die, space=ps)
        logging.info('----------- combined CDF sum 2x d6')

        rv_2x_die6 = probaspace.MixedRandomVariable(rv_die6, rv_die6, mix_func=sum)
        dist_2x_die6 = probaspace.SimpleDistributionFunction(rv_2x_die6)

        self.assertAlmostEqual(dist_2x_die6.evaluate(1).value, 0.)
        self.assertAlmostEqual(dist_2x_die6.evaluate(2).value, 0.02777776)
        self.assertAlmostEqual(dist_2x_die6.evaluate(3).value, 0.08333333)
        self.assertAlmostEqual(dist_2x_die6.evaluate(4).value, 0.16666666)
        self.assertAlmostEqual(dist_2x_die6.evaluate(5).value, 0.27777778)
        self.assertAlmostEqual(dist_2x_die6.evaluate(6).value, 0.41666667)
        self.assertAlmostEqual(dist_2x_die6.evaluate(7).value, 0.58333334)
        self.assertAlmostEqual(dist_2x_die6.evaluate(8).value, 0.72222222)
        self.assertAlmostEqual(dist_2x_die6.evaluate(9).value, 0.83333334)
        self.assertAlmostEqual(dist_2x_die6.evaluate(10).value, 0.91666666)
        self.assertAlmostEqual(dist_2x_die6.evaluate(11).value, 0.97222222)
        self.assertAlmostEqual(dist_2x_die6.evaluate(12).value, 1.0)
        self.assertAlmostEqual(dist_2x_die6.evaluate(13).value, 1.0)

        pdf = probaspace.make_probability_density(dist_2x_die6, 1., 13., 100)
        self.assertAlmostEqual(pdf[2.04].value, 0.0277777777)
        self.assertAlmostEqual(pdf[5.04].value, 0.111111111111)
        self.assertAlmostEqual(pdf[6.0].value, 0.1388888889)
        self.assertAlmostEqual(pdf[7.08].value, 0.166666666667)
        self.assertAlmostEqual(pdf[8.04].value, 0.1388888889)
        self.assertAlmostEqual(pdf[10.08].value, 0.0833333333)

        rv_3x_die6 = probaspace.MixedRandomVariable(rv_die6, rv_die6, rv_die6, mix_func=sum)
        dist_3x_die6 = probaspace.SimpleDistributionFunction(rv_3x_die6)

        pdf2 = probaspace.make_probability_density(dist_3x_die6, 1., 19., 100)
        self.assertAlmostEqual(pdf2[3.06].value, 0.004629629629629629)
        self.assertAlmostEqual(pdf2[5.04].value, 0.027777777777777776)
        self.assertAlmostEqual(pdf2[8.1].value, 0.09722222222)
        self.assertAlmostEqual(pdf2[11.16].value, 0.125)
        self.assertAlmostEqual(pdf2[14.04].value, 0.06944444444444453)
        self.assertAlmostEqual(pdf2[15.12].value, 0.04629629629629628)
        self.assertAlmostEqual(pdf2[18.0].value, 0.00462962962962965)

        rv_max_2x_2x_die6 = probaspace.MixedRandomVariable(rv_2x_die6, rv_2x_die6, mix_func=max)
        dist_max_2x_2x_die6 = probaspace.SimpleDistributionFunction(rv_max_2x_2x_die6)
        pdf3 = round_keys(probaspace.make_probability_density(dist_max_2x_2x_die6, 1., 19., 100))
        self.assertAlmostEqual(pdf3[2], 0.0007716049382716049)
        self.assertAlmostEqual(pdf3[6], 0.09645061728395062)
        self.assertAlmostEqual(pdf3[8], 0.18132716049382713)
        self.assertAlmostEqual(pdf3[10], 0.14583333333333337)
        self.assertAlmostEqual(pdf3[12], 0.054783950617283916)


def round_keys(pdf):
    rounded = defaultdict(float)
    for key in pdf.keys():
        rounded[int(key)] = pdf[key].value

    return rounded


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler('{}.log'.format(os.path.basename(__file__).split('.')[0]), mode='w')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    unittest.main()
