import collections
import itertools
from collections import defaultdict
from numbers import Number
import random
from typing import Callable, Iterable, Any, Dict


def powerset(iterable: Iterable[Any]) -> Iterable[Iterable[Any]]:
    """
    Example:
    > powerset([1,2,3])
    () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def cardinality(iterable: Iterable) -> int:
    d = collections.deque(enumerate(iterable, 1), maxlen=1)
    return d[0][0] if d else 0


class Event(object):
    def __init__(self, label: str):
        self._label = label

    def __repr__(self) -> str:
        return self._label

    @property
    def name(self):
        return self._label


class Universe(object):

    def __init__(self, items: Iterable[Event]):
        self._items = list(items)

    @property
    def items(self) -> Iterable[Event]:
        return self._items

    def size(self) -> int:
        return len(self._items)

    @classmethod
    def from_labels(cls, *params: str) -> 'Universe':
        return Universe((Event(label) for label in params))

    def create_random_variable(self, label, likelihood: Callable[[Event], float]) -> 'RandomVariable':
        return RandomVariable(label, likelihood, self)


class SigmaAlgebraItem(object):

    def __init__(self, items: Iterable[Event]):
        self._items = list(items)

    @property
    def items(self) -> Iterable[Event]:
        return self._items

    def __repr__(self):
        return repr(self.items)


class SigmaAlgebra(object):

    def __init__(self, universe: Universe, items: Iterable[Iterable[Event]]):
        self._universe = universe
        self._items = [SigmaAlgebraItem(universe_items) for universe_items in items]

    @property
    def items(self) -> Iterable[SigmaAlgebraItem]:
        return self._items

    @property
    def universe(self) -> Universe:
        return self._universe

    def __repr__(self):
        return '<sa>' + repr(list(self.items))


class Probability(object):

    def __init__(self, sigma_algebra: SigmaAlgebra):
        self._sigma_algebra = sigma_algebra

    def value(self, sigma_algebra_item: SigmaAlgebraItem) -> float:
        return float(cardinality(sigma_algebra_item.items) / float(self._sigma_algebra.universe.size()))


class RandomVariable(object):

    def __init__(self, description: str, item_mapper: Callable[[Event], float], universe: Universe):
        self._description = description
        self._item_mapper = item_mapper
        self._universe = universe

    def value(self, universe_item: Event) -> float:
        return self._item_mapper(universe_item)

    @property
    def universe(self) -> Universe:
        return self._universe

    @property
    def description(self) -> str:
        return self._description

    def __repr__(self):
        return self.description


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

    def __init__(self):
        self._buckets = defaultdict(int)

    @property
    def buckets(self) -> Dict[Number, int]:
        return self._buckets

    @property
    def normalized_buckets(self) -> Dict[Number, float]:
        total_occurences = float(sum(self._buckets.values()))
        return {bucket: float(occurence) / total_occurences for bucket, occurence in self.buckets.items()}

    def __repr__(self) -> str:
        return str({bucket: '{:.2f} %'.format(100. * occurence) for bucket, occurence in self.normalized_buckets.items()})

    def _run(self) -> Number:
        total = sum(self.buckets.values())
        target = random.randint(1, total)
        current = 0
        target_key = -1
        items = self.buckets.items()
        for key, value in sorted(items, key=lambda item: item[0]):
            current += value
            if target <= current:
                target_key = key
                break

        return target_key

    def runs(self, count) -> Dict[int, float]:
        buckets = defaultdict(int)
        for i in range(count):
            buckets[self._run()] += 1

        return {k: '{:.2f} %'.format(100. * float(buckets[k]) / sum(buckets.values())) for k in sorted(buckets.keys())}


class ProbabilityMassUniform(ProbabilityMass):

    def __init__(self, rv: RandomVariable):
        super().__init__()
        for item in rv.universe.items:
            self.buckets[rv.value(item)] = 1


class ProbabilityMassMixed(ProbabilityMass):

    def __init__(self, *pmfs: ProbabilityMass, mix_func: Callable[[Iterable[Number]], Number]):
        super().__init__()
        self._pmfs = pmfs
        self._mix_func = mix_func
        for items in itertools.product(*(pmf.buckets.items() for pmf in pmfs)):
            weight = 1
            for bucket in items:
                weight *= bucket[1]
            self.buckets[mix_func((bucket[0] for bucket in items))] += weight

    def _run(self) -> Number:
        # overrides
        return self._mix_func((pmf._run() for pmf in self._pmfs))


def make_power_set(universe: Universe) -> SigmaAlgebra:
    return SigmaAlgebra(universe, powerset(universe.items))

