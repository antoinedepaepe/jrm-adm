from abc import ABC, abstractmethod
import torch

import torch
# import torch_radon as tr
from src.physics.utils.wavelet import DWT_3D, IDWT_3D
import torch.nn.functional as F



class Physics(ABC):
    """
    Abstract base class for defining common transformation operations.
    Subclasses must implement transform and transposed_transform methods.
    """

    @abstractmethod
    def transform(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Applies a transformation to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor to transform.
            *args, **kwargs: Additional parameters for the transformation.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        pass


class Identity(Physics):

    def __init__(self) -> None:
        return
        
    def transform(self, x: torch.tensor):
        return x

    def transposed_transform(self, x: torch.tensor) -> torch.tensor:
        return x


class ConeBeam(Physics):

    def __init__(self, cone_beam) -> None:
        
        self.cone_beam = cone_beam

    def transform(self, x: torch.tensor,
                  angles: torch.Tensor = None) -> torch.tensor:
        return self.cone_beam.forward(x, angles)

    def transposed_transform(self, x: torch.tensor,
                                   angles: torch.Tensor = None) -> torch.tensor:
        return self.cone_beam.backward(x, angles)

    def fbp(self, y: torch.tensor, angles: torch.Tensor = None) -> torch.tensor:
        filtered_sinogram = self.cone_beam.filter_sinogram(y)
        fbp = self.cone_beam.backward(filtered_sinogram, angles)
        return fbp

class Extractor(Physics):
    
    def __init__(self):
        super().__init__()

    def transform(self, x: torch.tensor) -> torch.Tensor:
        
        d = torch.arange(x.shape[0], device=x.device)
        x = x[d, 0, d, :, :]
        x = x.unsqueeze(0).unsqueeze(0)

        return x


class Wavelet(Physics):

    def __init__(self, low_freq_factor: float = 3.0) -> None:
        self.low_freq_factor = low_freq_factor
        self.dwt = DWT_3D()
        self.idwt = IDWT_3D()

    def transform(self, x: torch.tensor):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x)
        x_wav = torch.cat([LLL / self.low_freq_factor, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        return x_wav

    def transposed_transform(self, x: torch.tensor):
        # NOTE: writen this way, it is not the exact adjoint,
        # but the inverse of the transform

        B, _, H, W, D = x.size()
        return self.idwt(x[:, 0, :, :, :].view(B, 1, H, W, D) * self.low_freq_factor,
                         x[:, 1, :, :, :].view(B, 1, H, W, D),
                         x[:, 2, :, :, :].view(B, 1, H, W, D),
                         x[:, 3, :, :, :].view(B, 1, H, W, D),
                         x[:, 4, :, :, :].view(B, 1, H, W, D),
                         x[:, 5, :, :, :].view(B, 1, H, W, D),
                         x[:, 6, :, :, :].view(B, 1, H, W, D),
                         x[:, 7, :, :, :].view(B, 1, H, W, D))

class RigidWarper(Physics):

    def __init__(self) -> None:
        pass

    def transform(self, x: torch.tensor,
                  theta: torch.tensor):
        
        dims = x.shape[2:]
        scale = torch.tensor([dims[2] - 1, dims[1] - 1, dims[0] - 1], dtype=theta.dtype, device=theta.device)
        theta = theta.clone()  # clone the tensor to avoid modifying a view

        theta[:, :, 3] = 2 * theta[:, :, 3] / scale
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        warped = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='zeros')
        return warped
