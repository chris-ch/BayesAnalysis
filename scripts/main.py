import itertools
import logging
import os
from collections import defaultdict
from itertools import chain, combinations
from typing import Callable, Iterable, Any, List


class Event(object):

    def __init__(self, event_name: str):
        self._event_name = event_name

    def __eq__(self, other):
        return self._event_name == other.name

    def __hash__(self):
        return self._event_name.__hash__()

    @property
    def name(self):
        return self._event_name


class Hypothesis(object):

    def __init__(self, description, probs: Callable[[Event], float]):
        self._description = description
        self._probs = probs

    def likelihood(self, event: Event) -> float:
        return self._probs(event)

    def __repr__(self):
        return self._description


class BayesAnalysis(object):

    def __init__(self):
        self._probabilities = defaultdict(float)

    def create_hypothesis(self, hypothesis: Hypothesis, prior: float) -> None:
        self._probabilities[hypothesis] = prior

    def add_event(self, event: Event) -> None:
        for hypothesis in self.hypotheses:
            self._probabilities[hypothesis] = self._probabilities[hypothesis] * hypothesis.likelihood(event)

        self._normalize()

    def _normalize(self) -> None:
        nomalization_factor = sum((self._probabilities[h] for h in self.hypotheses))
        for hypothesis in self.hypotheses:
            self._probabilities[hypothesis] /= nomalization_factor

    @property
    def hypotheses(self) -> Iterable[Hypothesis]:
        return self._probabilities.keys()

    def __repr__(self) -> str:
        output = os.linesep
        for h in self.hypotheses:
            output += repr(h) + ': ' + "%.2f %%" % (100. * self._probabilities[h]) + os.linesep

        return output


def smarties94(event: Event) -> float:
    probs = {Event('R'): 10, Event('G'): 10, Event('B'): 20}
    return float(probs[event]) / float(sum(probs.values()))


def smarties96(event: Event) -> float:
    probs = {Event('R'): 15, Event('G'): 10, Event('B'): 15}
    return float(probs[event]) / float(sum(probs.values()))


def dice(count_side: int) -> Callable[[Event], float]:
    def likelihood(event: Event):
        if int(event.name) > count_side:
            return 0.
        else:
            return 1. / float(count_side)

    return likelihood


def main():
    estimator = BayesAnalysis()
    #
    # estimator.create_hypothesis(Hypothesis("Smarties 94", smarties94), 0.5)
    # estimator.create_hypothesis(Hypothesis("Smarties 96", smarties96), 0.5)
    # estimator.add_event('R')
    # estimator.add_event('G')
    estimator.create_hypothesis(Hypothesis("Dice 4", dice(4)), 0.2)
    estimator.create_hypothesis(Hypothesis("Dice 6", dice(6)), 0.2)
    estimator.create_hypothesis(Hypothesis("Dice 8", dice(8)), 0.2)
    estimator.create_hypothesis(Hypothesis("Dice 12", dice(12)), 0.2)
    estimator.create_hypothesis(Hypothesis("Dice 20", dice(20)), 0.2)
    estimator.add_event(Event('3'))
    estimator.add_event(Event('4'))
    estimator.add_event(Event('8'))
    logging.info(estimator)

    universe = Universe([UniverseItem(index) for index in ['1', '2', '3', '4', '5', '6']])
    sa = make_power_set(universe)
    logging.info(str(sa.items))
    proba = Probability(sa)

    for item in sa.items:
        logging.info('{}: {}'.format(item, proba.value(item)))

    def die(item: UniverseItem) -> int:
        return int(str(item))

    rv = RandomVariable(item_mapper=die, universe=universe)
    dist1 = CumulativeDistributionFunction(rv, sa)
    for val in (float(i) / 5. for i in range(0, 35)):
        logging.info('cdf({}) = {}'.format(val, dist1.value(val)))

    logging.info('----------- combined CDF')
    dist2 = CombineSumCDF(rv, rv, sa)
    for val in (float(i)/ 5. for i in range(0, 65)):
        logging.info('cdf({}) = {}'.format(val, dist2.value(val)))

    logging.info('----------- probability mass')
    pm = ProbabilityMass(rv)
    logging.info('data: {}'.format(pm.buckets))

    logging.info('----------- probability mass sum 0')
    pms = ProbabilityMassSum0(rv, rv, rv)
    logging.info('data: {}'.format(pms))

    logging.info('----------- probability mass sum')
    pms = ProbabilityMassSum(pm, pm, pm)
    logging.info('data: {}'.format(pms))

    logging.info('----------- probability mass max')
    pmm = ProbabilityMassMax(pms, pms, pms, pms)
    logging.info('data: {}'.format(pmm))


def powerset(iterable: Iterable[Any]) -> Iterable[Iterable[Any]]:
    """
    Example:
    > powerset([1,2,3])
    () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class UniverseItem(object):
    def __init__(self, label: str):
        self._label = label

    def __repr__(self) -> str:
        return self._label


class Universe(object):
    def __init__(self, items: Iterable[UniverseItem]):
        self._items = list(items)

    @property
    def items(self) -> List[UniverseItem]:
        return self._items

    def size(self) -> int:
        return len(self._items)


class SigmaAlgebraItem(object):

    def __init__(self, items: Iterable[UniverseItem]):
        self._items = list(items)

    @property
    def items(self) -> List[UniverseItem]:
        return self._items

    def __repr__(self):
        return repr(self.items)


class SigmaAlgebra(object):

    def __init__(self, universe: Universe, items: Iterable[Iterable[UniverseItem]]):
        self._universe = universe
        self._items = [SigmaAlgebraItem(universe_items) for universe_items in items]

    @property
    def items(self) -> List[SigmaAlgebraItem]:
        return self._items

    @property
    def universe(self) -> Universe:
        return self._universe

    def __repr__(self):
        return '<sa>' + repr(list(self.items))


def make_power_set(universe: Universe) -> SigmaAlgebra:
    return SigmaAlgebra(universe, powerset(universe.items))


class Probability(object):

    def __init__(self, sigma_algebra: SigmaAlgebra):
        self._sigma_algebra = sigma_algebra

    def value(self, sigma_algebra_item: SigmaAlgebraItem) -> float:
        return float(len(sigma_algebra_item.items) / float(self._sigma_algebra.universe.size()))


class RandomVariable(object):
    def __init__(self, item_mapper: Callable[[UniverseItem], float], universe: Universe):
        self._item_mapper = item_mapper
        self._universe = universe

    def value(self, universe_item: UniverseItem) -> float:
        return self._item_mapper(universe_item)

    @property
    def universe(self) -> Universe:
        return self._universe


class CumulativeDistributionFunction(object):
    def __init__(self, random_variable: RandomVariable, sigma_algebra: SigmaAlgebra):
        self._random_variable = random_variable
        self._sigma_algebra = sigma_algebra

    def value(self, value: float) -> float:
        items = list()
        for item in self._sigma_algebra.universe.items:
            if self._random_variable.value(item) <= value:
                items.append(item)

        return Probability(self._sigma_algebra).value(SigmaAlgebraItem(items))


class CombineSumCDF(object):
    def __init__(self, rv1: RandomVariable, rv2: RandomVariable, sigma_algebra: SigmaAlgebra):
        self._rv1 = rv1
        self._rv2 = rv2
        self._sigma_algebra = sigma_algebra

    def value(self, value: float) -> float:
        items = list()
        for item in self._sigma_algebra.universe.items:
            if self._rv1.value(item) + self._rv2.value(item) <= value:
                items.append(item)

        return Probability(self._sigma_algebra).value(SigmaAlgebraItem(items))


class ProbabilityMass(object):
    def __init__(self, rv: RandomVariable):
        self._rv = rv
        self._buckets = defaultdict(int)
        for item in rv.universe.items:
            self._buckets[rv.value(item)] += 1

    @property
    def buckets(self):
        return self._buckets


class ProbabilityMassSum0(object):

    def __init__(self, *random_variables: RandomVariable):
        self._random_variables = random_variables
        self._buckets = defaultdict(int)
        for items in itertools.product(*(rv.universe.items for rv in random_variables)):
            self._buckets[sum(rv.value(item) for rv, item in zip(random_variables, items))] += 1

    @property
    def buckets(self):
        return self._buckets

    @property
    def normalized_buckets(self):
        total_occurences = float(sum(self._buckets.values()))
        return {bucket: float(occurence) / total_occurences for bucket, occurence in self._buckets.items()}

    def __repr__(self):
        return str({bucket: '{:.2f} %'.format(100. * occurence) for bucket, occurence in self.normalized_buckets.items()})


class ProbabilityMassSum(object):

    def __init__(self, *pmfs: ProbabilityMass):
        self._pmfs = pmfs
        self._buckets = defaultdict(int)
        for items in itertools.product(*(pmf.buckets.items() for pmf in pmfs)):
            # TODO weighting ?
            self._buckets[sum([bucket[0] for bucket in items])] += 1

    @property
    def buckets(self):
        return self._buckets

    @property
    def normalized_buckets(self):
        total_occurences = float(sum(self._buckets.values()))
        return {bucket: float(occurence) / total_occurences for bucket, occurence in self._buckets.items()}

    def __repr__(self):
        return str({bucket: '{:.2f} %'.format(100. * occurence) for bucket, occurence in self.normalized_buckets.items()})


class ProbabilityMassMax(object):

    def __init__(self, *pmfs: ProbabilityMassSum):
        self._pmfs = pmfs
        self._buckets = defaultdict(int)
        for items in itertools.product(*(pmf.buckets.items() for pmf in pmfs)):
            target_bucket, occurence = sorted(items, key=lambda item: item[0], reverse=True)[0]
            self._buckets[target_bucket] += occurence

    @property
    def buckets(self):
        return self._buckets

    @property
    def normalized_buckets(self):
        total_occurences = float(sum(self._buckets.values()))
        return {bucket: float(occurence) / total_occurences for bucket, occurence in self._buckets.items()}

    def __repr__(self):
        return str({bucket: '{:.2f} %'.format(100. * occurence) for bucket, occurence in self.normalized_buckets.items()})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler('{}.log'.format(os.path.basename(__file__).split('.')[0]), mode='w')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    main()
