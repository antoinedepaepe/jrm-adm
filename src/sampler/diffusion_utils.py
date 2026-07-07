import torch
from abc import ABC, abstractmethod

from src.physics.physics import Wavelet

def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class OneStepSampler(ABC):
    def __init__(self,
                 beta_start: float,
                 beta_end: float,
                 T: int,
                 device: str = 'cuda'):
        super().__init__()

        self.beta_start = beta_start
        self.beta_end = beta_start
        self.T = T
        self.device = device

    @abstractmethod
    def sample_one_step(xt: torch.Tensor, 
                        x0: torch.Tensor,
                        *args, **kwargs) -> torch.Tensor:
        pass


class OneStepDDIMSampler(OneStepSampler):
    def __init__(self, beta_start: float,
                       beta_end: float,
                       T: int,
                       steps: int,
                       eta: float = 0.0,
                       device = 'cuda'):
        super().__init__(beta_start=beta_start,
                         beta_end=beta_end, 
                         T=T, 
                         device=device)

        self.steps = steps
        self.eta = eta
        beta_t = torch.linspace(beta_start, beta_end, T, dtype=torch.float32).to(device)
        alpha_t = 1.0 - beta_t
        self.alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        self.wavelet = Wavelet()

    @torch.no_grad()
    def sample_one_step(self, xt: torch.Tensor, 
                              x0: torch.Tensor,
                              x0_no_step: torch.Tensor,
                              time_step: int,
                              prev_time_step: int) -> torch.Tensor:
        
        x_t = xt
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)
        
        epsilon_theta_t = (x_t - torch.sqrt(alpha_t) * x0_no_step) / torch.sqrt(1 - alpha_t) 
        
        sigma_t = self.eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = self.wavelet.transform(torch.randn_like(self.wavelet.transposed_transform(x_t) ))
        
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev) * x0 +
                
                torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * epsilon_theta_t +
                
                sigma_t * epsilon_t
        )

        return x_t_minus_one
