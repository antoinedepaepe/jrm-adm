import torch
from src.physics.physics import *


def convert_m1_p1_to_attenuation(mu_water_80keV: float = 0.0193,
                                  min_hu: float = -1000,
                                  max_hu: float = 2000):
    
    def inverse_transform(x: torch.Tensor) -> torch.Tensor:
        mu_min = mu_water_80keV * (min_hu / 1000 + 1)
        mu_max = mu_water_80keV * (max_hu / 1000 + 1)
        x = (x + 1) * (mu_max - mu_min) / 2 + mu_min
        return x

    return inverse_transform


class ExtractorRadonRigidWarper(Physics):

    def __init__(self, extractor: Physics,
                       radon: Physics,
                       warper: Physics) -> None:
        super().__init__()
        
        self.extractor = extractor
        self.radon = radon
        self.warper = warper
        self.unnormalizer = convert_m1_p1_to_attenuation()


    def transform(self, x: torch.tensor,
                        theta: torch.Tensor,
                        angles: torch.Tensor = None,
                        unnormalize: bool = False) -> torch.tensor:
        
        if unnormalize:
            x = self.unnormalizer(x)
        x = self.warper.transform(x, theta)
        x = self.radon.transform(x, angles)
        x = self.extractor.transform(x)
        return x


class ExtractorRadonRigidWarperWavelet(Physics):

    def __init__(self, extractor: Physics,
                       radon: Physics,
                       warper: Physics,
                       wavelet: Physics) -> None:
        super().__init__()
        
        self.extractor = extractor
        self.radon = radon
        self.warper = warper
        self.wavelet = wavelet
        self.unnormalizer = convert_m1_p1_to_attenuation()

    def transform(self, x: torch.tensor,
                        theta: torch.Tensor,
                        angles: torch.Tensor = None) -> torch.tensor:
        
        x = self.wavelet.transposed_transform(x)
        x = self.unnormalizer(x)
        x = self.warper.transform(x, theta)
        x = self.radon.transform(x, angles)
        x = self.extractor.transform(x)
        return x

