import torch as th
import torch.nn as nn
import gymnasium
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

VALID_SHAPES = {
    "fine":   (3, 16, 12),
    "coarse": (3,  8,  6),
}

class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for structured probe observations.

    Observation space: Box of shape (n_channels, n, m), where channels are:
      - 0: u_x  (streamwise velocity at each probe)
      - 1: u_y  (transverse velocity at each probe)
      - 2: p    (pressure at each probe)

    Output: flat feature vector of size `features_dim`, initialized to 256 
    """

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 256):

        shape = observation_space.shape
        valid = list(VALID_SHAPES.values())
        if shape not in valid:
            raise ValueError(
                f"CNNFeatureExtractor expects observation shape {valid[0]} (fine) "
                f"or {valid[1]} (coarse), got {shape}. "
                f"Check the probe_density argument passed to AirfoilConfig."
            )
        
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            cnn_flat_dim = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_flat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))




