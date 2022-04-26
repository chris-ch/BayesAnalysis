import collections
import itertools
import logging
import math
from collections import defaultdict
from numbers import Number
import random
from typing import Callable, Iterable, Any, Dict, List


class UnitSegmentValue(object):
    def __init__(self, value: float):
        if 0. <= value <= 1.:
            self._value = value
        else:
            raise ValueError('{} not in unit segment'.format(value))

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return repr(self.value)

    def __add__(self, other: 'UnitSegmentValue'):
        return UnitSegmentValue(self.value + other.value)


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

    def __init__(self, events: Iterable[Event]):
        self._events = sorted(events, key=lambda event: event.name)

    @property
    def events(self) -> List[Event]:
        return self._events

    def size(self) -> int:
        return len(self._events)

    @classmethod
    def from_labels(cls, *params: str) -> 'Universe':
        return Universe((Event(label) for label in params))

    @classmethod
    def from_range(cls, start, stop):
        return Universe((Event(str(pos)) for pos in range(start, stop)))

    def create_random_variable_single(self, label, likelihood: Callable[[Event], float]) -> 'RandomVariable':
        return RandomVariable(label, likelihood, self)

    def create_random_variables(self, likelihood: Callable[[Event], float]) -> Iterable['RandomVariable']:
        return (RandomVariable(str(event), event_mapper=likelihood, universe=self) for event in self.events)

    def fetch(self, event_name: str):
        return [event for event in self.events if event.name == event_name][0]


class SigmaAlgebraItem(object):

    def __init__(self, events: Iterable[Event]):
        self._events = list(events)

    @property
    def events(self) -> Iterable[Event]:
        return self._events

    def __repr__(self):
        return repr(self.events)


class SigmaAlgebra(object):

    def __init__(self, universe: Universe, events_set: Iterable[Iterable[Event]]):
        self._universe = universe
        self._items = [SigmaAlgebraItem(events) for events in events_set]

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

    def evaluate(self, sigma_algebra_item: SigmaAlgebraItem) -> UnitSegmentValue:
        return UnitSegmentValue(float(cardinality(sigma_algebra_item.events) / float(self._sigma_algebra.universe.size())))


class RandomVariable(object):

    def __init__(self, description: str, event_mapper: Callable[[Event], float], universe: Universe):
        self._description = description
        self._event_mapper = event_mapper
        self._universe = universe

    def evaluate(self, event: Event) -> float:
        return self._event_mapper(event)

    @property
    def universe(self) -> Universe:
        return self._universe

    @property
    def description(self) -> str:
        return self._description

    def __repr__(self):
        return self.description


class CumulativeDistributionFunction(object):

    def make_probability_density(self, start: float, stop: float, step: float) -> Dict[float, UnitSegmentValue]:
        density = dict()
        count = (stop - start) / step
        for index in range(int(count)):
            value_prev = step * index
            value_next = step * (index + 1)
            diff = self.evaluate(value_next).value - self.evaluate(value_prev).value
            if diff != 0:
                density[value_next] = UnitSegmentValue(diff)

        return density

    def evaluate(self, value: float) -> UnitSegmentValue:
        pass


class SimpleCumulativeDistributionFunction(CumulativeDistributionFunction):

    def __init__(self, random_variable: RandomVariable, sigma_algebra: SigmaAlgebra):
        self._random_variable = random_variable
        self._sigma_algebra = sigma_algebra

    def evaluate(self, value: float) -> UnitSegmentValue:
        events = list()
        for event in self._sigma_algebra.universe.events:
            if self._random_variable.evaluate(event) <= value:
                events.append(event)

        return Probability(self._sigma_algebra).evaluate(SigmaAlgebraItem(events))

    def __repr__(self) -> str:
        return str({event: '{:.2f} %'.format(100. * self.evaluate(float(event.name)).value) for event in self._sigma_algebra.universe.events})


class CombinedCumulativeDistributionFunction(CumulativeDistributionFunction):

    def __init__(self, rv1: RandomVariable, rv2: RandomVariable, sigma_algebra: SigmaAlgebra):
        self._rv1 = rv1
        self._rv2 = rv2
        self._sigma_algebra = sigma_algebra

    def evaluate(self, value: float) -> UnitSegmentValue:
        occurences = 0
        for event1, event2 in itertools.product(self._sigma_algebra.universe.events, self._sigma_algebra.universe.events):
            if self._rv1.evaluate(event1) + self._rv2.evaluate(event2) <= value:
                occurences += 1

        return UnitSegmentValue(float(occurences) / (len(self._sigma_algebra.universe.events) * len(self._sigma_algebra.universe.events)))

    def make_probability_density(self, start: float, stop: float, step: float) -> Dict[float, UnitSegmentValue]:
        density = dict()
        count = (stop - start) / step
        for index in range(int(count)):
            value_prev = step * index
            value_next = step * (index + 1)
            diff = self.evaluate(value_next).value - self.evaluate(value_prev).value
            if diff != 0:
                density[value_next] = UnitSegmentValue(diff)

        return density

    def _run(self) -> Event:
        target = random.random()
        target_event = None
        for event in self._sigma_algebra.universe.events:
            target_event = event
            if self.evaluate(float(event.name)).value >= target:
                break

        return target_event

    def runs(self, count) -> Dict[int, float]:
        buckets = defaultdict(int)
        for i in range(count):
            buckets[self._run()] += 1

        return {k: '{:.2f} %'.format(100. * float(buckets[k]) / sum(buckets.values())) for k in sorted(buckets.keys(), key=lambda b: b.name)}


def make_sigma_algebra_full(universe: Universe) -> SigmaAlgebra:
    return SigmaAlgebra(universe, powerset(universe.events))
