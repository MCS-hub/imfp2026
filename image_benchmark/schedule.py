"""Whole-sweep scheduling for fixed-count and wall-time-limited experiments."""
import numpy as np

PHASES = ('adaptation', 'burnin', 'retained_sampling')


class SweepSchedule:
    def __init__(self, counts, budget_sec=None, fractions=(1/6, 1/6, 2/3)):
        self.requested = dict(zip(PHASES, counts))
        self.counts = dict.fromkeys(PHASES, 0)
        self.budget_sec = budget_sec
        self.phase = 0
        self.boundaries = np.cumsum(fractions)*budget_sec if budget_sec is not None else None
        self.durations = []

    def next_phase(self, elapsed):
        # Reserve a full recent sweep; never interrupt a transition mid-proposal.
        estimate = max(self.durations[-5:], default=0.) * 1.10
        while self.phase < 3:
            name = PHASES[self.phase]
            if self.budget_sec is None:
                done = self.counts[name] >= self.requested[name]
            else:
                done = elapsed + estimate >= self.boundaries[self.phase]
            if not done:
                return name
            self.phase += 1
        return None

    def record(self, phase, seconds):
        self.counts[phase] += 1
        self.durations.append(seconds)
