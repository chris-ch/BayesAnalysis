import collections
import itertools
from typing import Callable, Iterable, Any, Dict, List, Tuple, Union, Set


class UnitRangeValue(object):
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

    def __add__(self, other: 'UnitRangeValue'):
        return UnitRangeValue(self.value + other.value)


def powerset(iterable: Iterable[Any]) -> Iterable[Iterable[Any]]:
    """
    Example:
    > powerset([1,2,3])
    () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def cardinality(iterable: Iterable[Any]) -> int:
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


class CompoundEvent(object):

    def __init__(self, events: Iterable[Event]):
        self._sub_events = events

    @property
    def name(self) -> str:
        return '<{}>'.format(self._sub_events)

    @property
    def sub_events(self):
        return self._sub_events

    def __repr__(self) -> str:
        return self.name


class Universe(object):

    def __init__(self):
        self._events = list()

    @property
    def events(self) -> List[Union[Event, CompoundEvent]]:
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

    def fetch(self, event_name: str) -> Union[Event, CompoundEvent, None]:
        try:
            return next(event for event in self.events if event.name == event_name)
        except StopIteration:
            return None

    def event(self, event_name: str) -> Event:
        event = self.fetch(event_name)
        if event is None:
            event = Event(event_name)
            self._events.append(event)

        return event

    def add(self, event: Union[Event, CompoundEvent]) -> None:
        self._events.append(event)

class SigmaAlgebraItem(object):

    def __init__(self, events: Iterable[Event]):
        self._events = list(events)

    @property
    def events(self) -> List[Event]:
        return self._events

    def set(self) -> Set[Event]:
        return set(self.events)
        
    def __repr__(self):
        return '<Events:{}>'.format(','.join((event.name for event in self.events)))


class SigmaAlgebra(object):
    """
    """

    def __init__(self, universe: Universe, events_set: Iterable[Iterable[Event]]):
        self._universe = universe
        self._events_set = events_set

    @property
    def items(self) -> Iterable[SigmaAlgebraItem]:
        return (SigmaAlgebraItem(events) for events in self._events_set)

    @property
    def universe(self) -> Universe:
        return self._universe

    def __repr__(self):
        return '<sa>' + repr(list(self.items))


class Probability(object):
    """
    Function P : 𝓕 → [0, 1] with the properties that P(Ω) = 1 and P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅
    """

    def __init__(self, sigma_algebra: SigmaAlgebra, mapping: Callable[[SigmaAlgebraItem], UnitRangeValue]):
        self._sigma_algebra = sigma_algebra
        self._mapping = mapping

    def evaluate(self, sigma_algebra_item: SigmaAlgebraItem) -> UnitRangeValue:
        return self._mapping(sigma_algebra_item)


class UnbiasedProbability(Probability):
    """
    Function P : 𝓕 → [0, 1] with the properties that P(Ω) = 1 and P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅
    """

    def __init__(self, sigma_algebra: SigmaAlgebra):
        super().__init__(sigma_algebra, lambda sai: UnitRangeValue(float(len(sai.events)) / sigma_algebra.universe.size()))


class ProbabilitySpace(object):

    def __init__(self, sigma_algebra: SigmaAlgebra, probability: Probability):
        self._sigma_algebra = sigma_algebra
        self._probability = probability

    @property
    def universe(self):
        return self._sigma_algebra.universe

    @property
    def sigma_algebra(self):
        return self._sigma_algebra

    @property
    def probability(self):
        return self._probability


class RandomVariable(object):
    """
    Function X : Ω → ℝ with the property that { ω ∈ Ω : X(ω) ≤ x } ∈ 𝓕, ∀x ℝ
    """

    def __init__(self, description: str, event_mapper: Callable[[Event], float], space: ProbabilitySpace):
        self._description = description
        self._event_mapper = event_mapper
        self._space = space

    def evaluate(self, event: Event) -> float:
        return self._event_mapper(event)

    @property
    def universe(self) -> Universe:
        return self._space.universe

    @property
    def space(self) -> ProbabilitySpace:
        return self._space

    @property
    def description(self) -> str:
        return self._description

    def __repr__(self):
        return self.description

    def expectancy(self) -> float:
        result = 0.
        for event in self.universe.events:
            result += self.evaluate(event) * self.space.probability.evaluate(SigmaAlgebraItem({event})).value

        return result

    def variance(self) -> float:
        result = 0.
        mu = self.expectancy()
        for event in self.universe.events:
            result += ((self.evaluate(event) - mu) ** 2) * self.space.probability.evaluate(SigmaAlgebraItem({event})).value

        return result


class MixedRandomVariable(object):

    def __init__(self, *random_vector: Union[RandomVariable, 'MixedRandomVariable'], mix_func: Callable[[Iterable[float]], float]):
        universe = Universe()
        for events in itertools.product(*(rv.universe.events for rv in random_vector)):
            universe.add(CompoundEvent(events))

        # universe: { (a, a), (a, b), (b, a), (b, b) }
        # { {}, {(a, a)}, {(a, b)}, {(b, a)}, {(b, b)}, {(a, a), (a, b)}, {(a, a), (b, a)}, {(a, a), (b, b)}, ... }
        # from { {}, {a}, {b},{a, b} }
        self._space = make_space_full(universe)
        self._rvs = random_vector
        self._mix_func = mix_func

    def evaluate(self, event: CompoundEvent) -> float:
        return self._mix_func((rv.evaluate(evt) for evt, rv in zip(event.sub_events, self._rvs)))

    @property
    def universe(self) -> Universe:
        return self._space.universe

    @property
    def space(self) -> ProbabilitySpace:
        return self._space


def is_tuple_less_than(t1: Tuple[float], t2: Tuple[float]) -> bool:
    return sum((True for v1, v2 in zip(t1, t2) if v1 > v2)) == 0


class JointDistributionFunction(object):

    def __init__(self, *random_vector: RandomVariable):
        self._random_vector = random_vector

    def evaluate(self, *vector: float) -> UnitRangeValue:
        values = tuple(vector)
        if len(values) != len(self._random_vector):
            raise ValueError('{} has wrong size, expected {}'.format(values, len(self._random_vector)))

        product_universes = itertools.product(*(rv.universe.events for rv in self._random_vector))
        vectors = list(vector for vector in product_universes if is_tuple_less_than(tuple(rv.evaluate(event) for event, rv in zip(vector, self._random_vector)), values))
        return UnitRangeValue(float(len(vectors)) / cardinality(itertools.product(*(rv.universe.events for rv in self._random_vector))))

    def __repr__(self) -> str:
        return str({events: '{:.2f} %'.format(100. * self.evaluate(*(rv.evaluate(event) for event, rv in zip(events, self._random_vector))).value) for events in itertools.product(*(rv.universe.events for rv in self._random_vector))})


class SimpleDistributionFunction(object):

    def __init__(self, random_variable: Union[RandomVariable, MixedRandomVariable]):
        self._rv = random_variable

    def evaluate(self, value: float) -> UnitRangeValue:
        events = SigmaAlgebraItem(event for event in self._rv.universe.events if self._rv.evaluate(event) <= value)
        return self._rv.space.probability.evaluate(events)


class MixedDistributionFunction(object):

    def __init__(self, *random_vector: RandomVariable, mix_func: Callable[[Iterable[float]], float]):
        self.random_vector = random_vector
        self._mix_func = mix_func

    def evaluate(self, value: float) -> UnitRangeValue:
        occurrences = 0
        total = 0
        for events in self._events():
            rv_event_pairs = zip(self.random_vector, events)
            total += 1
            if self._mix_func((rv.evaluate(event) for rv, event in rv_event_pairs)) <= value:
                occurrences += 1

        return UnitRangeValue(float(occurrences) / total)

    def _events(self):
        return itertools.product(*(rv.universe.events for rv in self.random_vector))


def make_space_full(universe: Universe) -> ProbabilitySpace:
    sa = SigmaAlgebra(universe, powerset(universe.events))
    return ProbabilitySpace(sa, UnbiasedProbability(sa))


def make_probability_density(distribution: Union[SimpleDistributionFunction, MixedDistributionFunction], start: float, stop: float, count: int) -> Dict[float, UnitRangeValue]:
    density = dict()
    step = (stop - start) / count
    for index in range(int(count)):
        value_prev = step * index
        value_next = step * (index + 1)
        diff = distribution.evaluate(value_next).value - distribution.evaluate(value_prev).value
        if diff != 0:
            density[value_next] = UnitRangeValue(diff)

    return density


def make_random_variable_single(label, event_mapper: Callable[[Event], float], space: ProbabilitySpace) -> RandomVariable:
    return RandomVariable(label, event_mapper, space)


def make_random_variables(event_mapper: Callable[[Event], float], space: ProbabilitySpace) -> Iterable[RandomVariable]:
    return (RandomVariable(str(event), event_mapper=event_mapper, space=space) for event in space.universe.events)

