import torch
from src.physics.physics import Physics
import torch


class JRMDataFidelity:

    def __init__(self, physics: Physics,
                        yi: torch.Tensor,
                        angles: torch.Tensor,
                        angle_batch_size: int,):

        self.physics = physics
        self.angles = angles
        self.angle_batch_size = angle_batch_size
        self.yi = yi


    def loss(self, x: torch.Tensor,
                   thetas: torch.Tensor,
                   x_star: torch.Tensor,
                   tau: float,
                   b: torch.Tensor) -> torch.Tensor:
 
        n_timesteps= self.angles.shape[0]
        n_chunks = n_timesteps // self.angle_batch_size
        timesteps = torch.linspace(0, 1, n_timesteps).unsqueeze(0).to('cuda')
        x_chunk = x.expand(self.angle_batch_size, -1, -1, -1, -1)
            
        loss_total = 0.0

        for idx_chunk in range(n_chunks):

            z_min = idx_chunk * self.angle_batch_size
            z_max = (idx_chunk + 1) * self.angle_batch_size

            sub_angles = self.angles[z_min:z_max]
            b_pred_chunk = self.physics.transform(x_chunk, thetas[z_min:z_max, ...], sub_angles, True)
            loss_chunk = torch.mean(((b_pred_chunk - b[:, :, z_min:z_max, ...]) **2) * self.yi[:, :, z_min:z_max, ...])+ tau * torch.mean((x_star-x)**2 ) 
            loss_total += loss_chunk.item()

            loss_chunk.backward(retain_graph= (idx_chunk != (n_chunks - 1)))
    
        print(f'Loss solver : {loss_total}')
            
        return loss_total

     