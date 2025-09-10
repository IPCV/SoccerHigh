from torch import randperm
from torch.nn import Module
from torch.distributions.beta import Beta


class MixUp(Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
    ):
        super().__init__()
        
        self.betadistr = Beta(alpha, beta)
    
    def forward(self, x, y):
        # Get lambda sample from beta distribution
        lamb = self.betadistr.sample()
        # Get random permutation
        index = randperm(x.size(0))
        # Mix data
        mixed_x = lamb * x + (1 - lamb) * x[index, :]
        # Mix labels depending on its structure
        if isinstance(y, list):
            mixed_y = [lamb * y_i + (1 - lamb) * y_i[index] for y_i in y]
        elif isinstance(y, dict):
            mixed_y = {}
            for key, value in y.items():
                mixed_y[key] = lamb * value + (1 - lamb) * value[index]
        else:
            mixed_y = lamb * y + (1 - lamb) * y[index]
        # Return mixed data
        return mixed_x, mixed_y