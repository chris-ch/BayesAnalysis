import os
from collections import defaultdict
from typing import Iterable, Callable

import probaspace


class BayesAnalysis(object):

    def __init__(self):
        self._probabilities = defaultdict(float)
        self._universe = None

    def define_uninformed(self, universe: probaspace.Universe, *hypotheses: probaspace.RandomVariable) -> None:
        self._universe = universe
        prior = 1. / float(len(hypotheses))
        for h in hypotheses:
            self._probabilities[h] = prior

    def define_from_range(self, start: int, stop: int, likelihood: Callable[[probaspace.Event], float]):
        universe = probaspace.Universe.from_range(start, stop)
        hypotheses = universe.create_random_variables(likelihood=likelihood)
        self.define_uninformed(universe, *hypotheses)

    def add_event(self, event: probaspace.Event) -> None:
        for hypothesis in self.hypotheses:
            self._probabilities[hypothesis] = self._probabilities[hypothesis] * hypothesis.evaluate(event)

        self._normalize()

    def add_events(self, *params: str):
        for param in params:
            self.add_event(self._universe.fetch(param))

    def _normalize(self) -> None:
        nomalization_factor = sum((self._probabilities[h] for h in self.hypotheses))
        for hypothesis in self.hypotheses:
            self._probabilities[hypothesis] /= nomalization_factor

    def evaluate(self, hypothesis: probaspace.RandomVariable) -> float:
        return self._probabilities[hypothesis]

    @property
    def hypotheses(self) -> Iterable[probaspace.RandomVariable]:
        return self._probabilities.keys()

    def __repr__(self) -> str:
        output = os.linesep
        for h in self.hypotheses:
            output += repr(h) + ': ' + "%.2f %%" % (100. * self._probabilities[h]) + os.linesep

        return output


