import torch
import torch.nn as nn
from tqdm import tqdm

from src.sampler.diffusion_utils import OneStepSampler, OneStepDDIMSampler
import numpy as np
from src.physics.physics import Physics
import matplotlib.pyplot as plt
from src.utils.utils_plot import save_slices
from src.utils.dataloaders import convert_m1_p1_to_attenuation, get_head_ct_dataloaders

to_attenuation = convert_m1_p1_to_attenuation()

class AdaptativeDiffusionSampler: 
    
    def __init__(self,  model: nn.Module,
                        operator: Physics,
                        data_fidelity: callable,
                        x_solver: callable,
                        motion_solver: callable,
                        one_step_sampler: OneStepSampler,             
                        device: str = 'cuda') -> None:
        
        super().__init__()

        self.model = model
        self.operator = operator
        self.data_fidelity = data_fidelity
        self.one_step_sampler = one_step_sampler
        self.device = device

        self.x_solver = x_solver
        self.motion_solver = motion_solver

    def sample(self, xt: torch.Tensor,
                     thetas: torch.Tensor,
                     b: torch.Tensor,
                     gamma: float,
                     root_plot: str = None) -> torch.Tensor:
        
        if isinstance(self.one_step_sampler, OneStepDDIMSampler):
            delta = self.one_step_sampler.T // self.one_step_sampler.steps
            timesteps = np.asarray(list(range(0, self.one_step_sampler.T, delta)))
            timesteps = timesteps + 1
            timesteps_minus_delta = np.concatenate([[0], timesteps[:-1]])

        middle = int(thetas.shape[0] // 2)
        x0 = torch.zeros_like(xt)
        idx = 0

        with tqdm(reversed(range(0, self.one_step_sampler.steps)), colour="#6565b5", total=self.one_step_sampler.steps) as sampling_steps:
            for i in sampling_steps:

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                if idx < 10: #5
                    iteration = 15 # 25
                    iter_motion = 10 #10
                    gamma_tilde = 0
                else:
                    iteration = 10 # was 5
                    iter_motion = 5
                    gamma_tilde = gamma

                if idx == 1:
                    self.x_solver.lr = 0.001
                    self.motion_solver.lr = [0.01, 0.01]

                if idx >= 50:
                    self.x_solver.lr = 0.0005
                    self.motion_solver.lr = [0.001, 0.001]

                # Turn off motion compensation
                x0_model = self.model_inference(xt, timesteps[i])
                x0_no_step = x0_model.clone()
            
                x0 = self.prox_step(x0=x0,
                                    thetas=thetas,
                                    b=b,
                                    x0_model=x0_model,
                                    gamma=gamma_tilde,
                                    iteration=iteration)
                
                if root_plot is not None:
                    self.plot(x0, thetas[middle:middle+1, ...], root=root_plot, prefix=f'x0_prox_{idx}')
                    self.plot(x0_model, thetas[middle:middle+1, ...], root=root_plot, prefix=f'x0_pred_{idx}')

                # Motion estimation step
                thetas, motion_pattern_full = self.blind_step(x0=x0, b=b, iteration=iter_motion)
               
                x0 = self.clip(x0=x0)
                xt = self.one_step_sampler.sample_one_step(xt, x0, x0_no_step, timesteps[i], timesteps_minus_delta[i])

                idx += 1
            
            x0 = self.clip(x0=x0_model)
            x0 = self.operator.transposed_transform(x0)

            return x0, thetas, self.motion_solver.motion_pattern_generator.get_control_points().detach(), motion_pattern_full

    @torch.no_grad()
    def model_inference(self, xt: torch.Tensor,
                              t: int) -> torch.Tensor:
        t_ = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        x0 = self.model(xt, t_).detach()
        x0 = self.clip(x0=x0)
        return x0

    def clip(self, x0: torch.Tensor) -> torch.Tensor:
        x0 = self.operator.transposed_transform(x0)
        x0 = torch.clip(x0, -1.0, 1.0)
        x0 = self.operator.transform(x0).detach()
        return x0

    def blind_step(self, x0: torch.Tensor,
                         b: torch.Tensor,
                         iteration: int = 15):
        
        x0 = self.operator.transposed_transform(x0)
        thetas, motion_pattern_full = self.motion_solver.solve(x0, b, iteration)
        
        return thetas.clone().detach(), motion_pattern_full.clone().detach()

    @torch.enable_grad()
    def prox_step(self, x0: torch.Tensor,
                            thetas: torch.Tensor,    
                            b: torch.Tensor,
                            x0_model: torch.Tensor,
                            gamma: float,
                            iteration: int = 15) -> torch.Tensor:
        
        x0 = self.operator.transposed_transform(x0).detach()
        x0_model = self.operator.transposed_transform(x0_model).detach()
        x0 = self.x_solver.solve(x0, thetas, b,  x0_model, gamma, iteration)
        x0 = self.operator.transform(x0).detach()

        return x0

    def load_model(self, path: str) -> None:
        checkpoints = torch.load(path)
        self.model.load_state_dict(checkpoints['model_state_dict'])


    def plot(self, x: torch.Tensor, thetas: torch.Tensor, root:str , prefix:str) -> torch.Tensor:
        to_plot = self.data_fidelity.physics.warper.transform(to_attenuation(self.operator.transposed_transform(x)), thetas)
        save_slices(to_plot, root=root, prefix=prefix)
