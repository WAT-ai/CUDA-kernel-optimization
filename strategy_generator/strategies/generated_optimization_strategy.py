import numpy as np
import random
from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc

_options = dict(
    popsize=("Population size", 20),
    maxiter=("Maximum number of iterations", 100),
    w=("Inertia weight constant", 0.5),
    c1=("Cognitive constant", 2.0),
    c2=("Social constant", 1.0),
    crossover_rate=("Rate at which crossover is applied", 0.8),
    mutation_rate=("Rate at which mutation is applied", 0.1)
)

class HybridParticle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.random.uniform(-1, 1, len(position))
        self.best_pos = position.copy()
        self.best_score = float('inf')
        self.score = float('inf')

def tune(searchspace: Searchspace, runner, tuning_options):
    
    options = tuning_options.strategy_options
    num_particles, maxiter, w, c1, c2,crossover_rate ,mutation_rate= common.get_options(options,_options)