import os
from collections import defaultdict
from typing import Iterable

import probaspace


class BayesAnalysis(object):

    def __init__(self):
        self._probabilities = defaultdict(float)

    def define_uninformed(self, *hypotheses: probaspace.RandomVariable) -> None:
        prior = 1. / float(len(hypotheses))
        for h in hypotheses:
            self._probabilities[h] = prior

    def add_event(self, event: probaspace.Event) -> None:
        for hypothesis in self.hypotheses:
            self._probabilities[hypothesis] = self._probabilities[hypothesis] * hypothesis.value(event)

        self._normalize()

    def _normalize(self) -> None:
        nomalization_factor = sum((self._probabilities[h] for h in self.hypotheses))
        for hypothesis in self.hypotheses:
            self._probabilities[hypothesis] /= nomalization_factor

    @property
    def hypotheses(self) -> Iterable[probaspace.RandomVariable]:
        return self._probabilities.keys()

    def __repr__(self) -> str:
        output = os.linesep
        for h in self.hypotheses:
            output += repr(h) + ': ' + "%.2f %%" % (100. * self._probabilities[h]) + os.linesep

        return output

