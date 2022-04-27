import collections
import itertools
import logging
import math
import operator
from collections import defaultdict
from numbers import Number
import random
from typing import Callable, Iterable, Any, Dict, List, Tuple, Union


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

    def __eq__(self, other):
        return isinstance(other, Event) and self.name == other.name

    def __hash__(self):
        return self.name.__hash__()

    @property
    def name(self) -> str:
        return self._label


class Universe(object):

    def __init__(self):
        self._events = list()

    @property
    def events(self) -> List[Event]:
        return self._events

    def size(self) -> int:
        return len(self._events)

    @classmethod
    def from_labels(cls, *labels: str) -> 'Universe':
        return cls.from_iterable(labels)

    @classmethod
    def from_range(cls, start, stop) -> 'Universe':
        return cls.from_iterable((str(pos) for pos in range(start, stop)))

    @classmethod
    def from_iterable(cls, event_names: Iterable[str]) -> 'Universe':
        universe = Universe()
        for event_name in event_names:
            universe.event(event_name)

        return universe

    def create_random_variable_single(self, label, likelihood: Callable[[Event], float]) -> 'RandomVariable':
        return RandomVariable(label, likelihood, self)

    def create_random_variables(self, likelihood: Callable[[Event], float]) -> Iterable['RandomVariable']:
        return (RandomVariable(str(event), event_mapper=likelihood, universe=self) for event in self.events)

    def fetch(self, event_name: str):
        events = [event for event in self.events if event.name == event_name]
        if len(events) == 0:
            return None
        elif len(events) == 1:
            return events[0]
        else:
            return ValueError('event {} found multiple times'.format(event_name))

    def event(self, event_name: str):
        event = self.fetch(event_name)
        if event is None:
            event = Event(event_name)
            self._events.append(event)

        return event


class SigmaAlgebraItem(object):

    def __init__(self, events: Iterable[Event]):
        self._events = list(events)

    @property
    def events(self) -> Iterable[Event]:
        return self._events

    def __repr__(self):
        return repr(self.events)


class SigmaAlgebra(object):
    """
    """

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
    """
    Function P : 𝓕 → [0, 1] with the properties that P(Ω) = 1 and P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅
    """

    def __init__(self, sigma_algebra: SigmaAlgebra):
        self._sigma_algebra = sigma_algebra

    def evaluate(self, sigma_algebra_item: SigmaAlgebraItem) -> UnitSegmentValue:
        return UnitSegmentValue(float(cardinality(sigma_algebra_item.events) / float(self._sigma_algebra.universe.size())))


class RandomVariable(object):
    """
    Function X : Ω → ℝ with the property that { ω ∈ Ω : X(ω) ≤ x } ∈ 𝓕, ∀x ℝ
    """

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


class DistributionFunction(object):
    """
    Function Fₓ : ℝ → [0, 1] defined as Fₓ(x) = P( { ω ∈ Ω : X(ω) ≤ x } )
    """
    def evaluate(self, value: float) -> UnitSegmentValue:
        return UnitSegmentValue(0.)


def is_tuple_less_than(t1: Tuple[float], t2: Tuple[float]) -> bool:
    return sum((True for v1, v2 in zip(t1, t2) if v1 > v2)) == 0


class JointDistributionFunction(DistributionFunction):

    def __init__(self, *rvs: RandomVariable, sigma_algebra: SigmaAlgebra):
        self._rvs = rvs
        self._sigma_algebra = sigma_algebra

    def evaluate(self, *random_values: float) -> UnitSegmentValue:
        values = tuple(random_values)
        if len(values) != len(self._rvs):
            raise ValueError('{} has wrong size, expected {}'.format(values, len(self._rvs)))

        vectors = (vector for vector in itertools.product(*(self._sigma_algebra.universe.events for _ in self._rvs)) if is_tuple_less_than(tuple(float(event.name) for event in vector), values))
        return UnitSegmentValue(cardinality(vectors) / len(self._sigma_algebra.universe.events) ** len(self._rvs))

    def __repr__(self) -> str:
        return str({events: '{:.2f} %'.format(100. * self.evaluate(*(float(event.name) for event in events)).value) for events in itertools.product(*(self._sigma_algebra.universe.events for _ in self._rvs))})


class SimpleDistributionFunction(JointDistributionFunction):

    def __init__(self, random_variable: RandomVariable, sigma_algebra: SigmaAlgebra):
        super().__init__(random_variable, sigma_algebra=sigma_algebra)

    def evaluate(self, value: float) -> UnitSegmentValue:
        events = (event for event in self._sigma_algebra.universe.events if self._rvs[0].evaluate(event) <= value)
        return Probability(self._sigma_algebra).evaluate(SigmaAlgebraItem(events))

    def __repr__(self) -> str:
        return str({event: '{:.2f} %'.format(100. * self.evaluate(float(event.name)).value) for event in self._sigma_algebra.universe.events})


class MixedDistributionFunction(DistributionFunction):

    def __init__(self, *rvs: RandomVariable, mix_func: Callable[[Iterable[float]], float], sigma_algebra: SigmaAlgebra):
        self._rvs = rvs
        self._sigma_algebra = sigma_algebra
        self._mix_func = mix_func

    def evaluate(self, value: float) -> UnitSegmentValue:
        occurrences = 0
        total = 0
        for events in itertools.product(*(self._sigma_algebra.universe.events for _ in self._rvs)):
            rv_event_pairs = zip(self._rvs, events)
            total += 1
            if self._mix_func((rv.evaluate(event) for rv, event in rv_event_pairs)) <= value:
                occurrences += 1

        return UnitSegmentValue(float(occurrences) / total)


def make_sigma_algebra_full(universe: Universe) -> SigmaAlgebra:
    return SigmaAlgebra(universe, powerset(universe.events))


def make_probability_density(distribution: Union[SimpleDistributionFunction, MixedDistributionFunction], start: float, stop: float, count: int) -> Dict[float, UnitSegmentValue]:
    density = dict()
    step = (stop - start) / count
    for index in range(int(count)):
        value_prev = step * index
        value_next = step * (index + 1)
        diff = distribution.evaluate(value_next).value - distribution.evaluate(value_prev).value
        if diff != 0:
            density[value_next] = UnitSegmentValue(diff)

    return density
