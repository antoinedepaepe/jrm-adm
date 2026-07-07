
import torch
from src.physics.physics import Physics
from src.utils.creator_utils import motion_pattern_to_theta
from typing import Tuple

class RMSpropMotionEstimator:

    def __init__(self, physics: Physics,
                       motion_pattern_generator: torch.nn.Module,
                       lr: Tuple[float, float],
                       yi: torch.Tensor,
                       angles: torch.Tensor,
                       angle_batch_size: int = 20) -> None:
        
        self.physics = physics
        self.lr = lr
        self.motion_pattern_generator = motion_pattern_generator

        self.optimizer = torch.optim.RMSprop(([
                                            {"params": list(self.motion_pattern_generator.tx_splines.parameters()) + 
                                                        list(self.motion_pattern_generator.ty_splines.parameters()) +
                                                        list(self.motion_pattern_generator.tz_splines.parameters()), "lr": self.lr[0]},
                                            {"params": list(self.motion_pattern_generator.rx_splines.parameters()) + 
                                                        list(self.motion_pattern_generator.ry_splines.parameters()) +
                                                        list(self.motion_pattern_generator.rz_splines.parameters()), "lr": self.lr[1]}
                                        ]))
        self.yi = yi
        self.angle_batch_size = angle_batch_size
        self.angles = angles

    def solve(self, x: torch.Tensor,
                    y: torch.Tensor,
                    iteration: int = 20): 
    
        
        n_timesteps= self.angles.shape[0]
        n_chunks = n_timesteps // self.angle_batch_size
        timesteps = torch.linspace(0, 1, n_timesteps).unsqueeze(0).to('cuda')
        x_chunk = x.expand(self.angle_batch_size, -1, -1, -1, -1)

        for it in range(iteration):
            
            self.optimizer.zero_grad()
            
            motion_pattern = self.motion_pattern_generator(None, sub_sample=True)          
            thetas = motion_pattern_to_theta(motion_pattern)
            
            loss_total = 0.0

            for idx_chunk in range(n_chunks):

                z_min = idx_chunk * self.angle_batch_size
                z_max = (idx_chunk + 1) * self.angle_batch_size
                sub_angles = self.angles[z_min:z_max]
                y_pred_chunk = self.physics.transform(x_chunk, thetas[z_min:z_max, ...], sub_angles, True)

                loss_chunk = torch.mean( ( (y_pred_chunk - y[:, :, z_min:z_max, ...])**2 ) * self.yi[:, :, z_min:z_max, ...] )# + tau * torch.nn.functional.mse_loss( motion_pattern, self.pre_estimated_motion)
                loss_total += loss_chunk.item()

                loss_chunk.backward(retain_graph= (idx_chunk != (n_chunks - 1)))
        
                print(f"Iteration {it+1}/{iteration} - Loss motion: {loss_total}")

            self.optimizer.step()

        with torch.no_grad():

            motion_pattern = self.motion_pattern_generator(None, sub_sample=True) #self.motion_pattern_generator(timesteps, sub_sample=False) #
            motion_pattern_tilde = self.motion_pattern_generator(torch.linspace(0, 1, 120).unsqueeze(0).cuda(), sub_sample=False).clone().detach()

            # torch.save(motion_pattern_tilde.clone().detach(), f'./outputs/motion_patterns/motion_pattern_pred.pth')

            theta = motion_pattern_to_theta(motion_pattern)
            theta = theta.detach()
            self.cps =  self.motion_pattern_generator.get_control_points().clone().detach()
            
        self._reset_model_and_optimizer()

        return theta, motion_pattern_tilde


    def _reset_model_and_optimizer(self):
       
        for param in self.motion_pattern_generator.parameters():
            if param.requires_grad:
                param.data.zero_()
        self.motion_pattern_generator.set_control_points(self.cps)
        self.optimizer = torch.optim.RMSprop(([
                                                {"params": list(self.motion_pattern_generator.tx_splines.parameters()) + 
                                                            list(self.motion_pattern_generator.ty_splines.parameters()) +
                                                            list(self.motion_pattern_generator.tz_splines.parameters()), "lr": self.lr[0]},
                                                {"params": list(self.motion_pattern_generator.rx_splines.parameters()) + 
                                                            list(self.motion_pattern_generator.ry_splines.parameters()) +
                                                            list(self.motion_pattern_generator.rz_splines.parameters()), "lr": self.lr[1]}
                                            ]))