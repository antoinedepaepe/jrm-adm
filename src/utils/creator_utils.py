from src.models.w3dm import UNetModel

import torch
import torch.nn as nn
from typing import Tuple

from src.physics.physics import ConeBeam

import torch_radon as tr


####### Model creation and operator creation utils ########

def create_model(image_size=192,
                  num_channels=64,
                  num_res_blocks=2,
                  channel_mult=(1, 2, 2, 4, 4),
                  attention_resolutions=None,
                  num_heads=1,
                  num_head_channels=-1,
                  num_heads_upsample=-1,
                  use_scale_shift_norm=False,
                  dropout=0,
                  resblock_updown=True,
                  use_fp16=False,
                  use_checkpoint=False,
                  use_new_attention_order=False,
                  num_groups=32,
                  dims=3,
                  in_channels=8,
                  out_channels=8,
                  bottleneck_attention=False,
                  resample_2d=False,
                  additive_skips=True,
                  num_classes=None) -> nn.Module:

    if attention_resolutions is None:
        attention_resolutions = []

    model = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            dims=dims,
            num_groups=num_groups,
            bottleneck_attention=bottleneck_attention,
            additive_skips=additive_skips,
            resample_2d=resample_2d,
            )
    return model


####### Physics creation utils ########

def create_volume(center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  size: Tuple[int, int, int] = (160, 192, 192),
                  voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  ) -> tr.volumes.Volume3D:
    
    volume = tr.volumes.Volume3D(center=center, voxel_size=voxel_size)
    volume.set_size(*size)
    return volume

def create_cone_beam_projector(angles: torch.Tensor = None,
                         volume: tr.Volume3D = None,
                         det_count_u: int = 192 * 2,
                         det_count_v: int = 60,
                         det_spacing_u: float = 1.0,
                         det_spacing_v: float = 1.0,
                         src_dist: float = 650,
                         det_dist: float = 300,
                         pitch: float = 0.0,
                         base_z: float = 0.0,
                         ) -> ConeBeam:
    
    if angles is None:
        raise ValueError("angles parameter must be provided.")
    if volume is None:
        raise ValueError("volume parameter must be provided.")
    
    cone_beam = tr.ConeBeam(
        det_count_u=det_count_u,
        angles=angles,
        src_dist=src_dist,
        det_dist=det_dist,
        det_count_v=det_count_v,
        det_spacing_u=det_spacing_u,
        det_spacing_v=det_spacing_v,
        pitch=pitch,
        base_z=base_z,
        volume=volume,
    )

    cone_beam = ConeBeam(cone_beam=cone_beam)
    return cone_beam




####### Motion creation utils ########


import torch
import matplotlib.pyplot as plt
from src.physics.physics import Physics

import torch
from src.models.rigid_motion_theta import RigidMotionTheta


def random_control_points(n_control_points: int,
                                   amplitude : float):
    rand = torch.rand(1, n_control_points)
    out =  amplitude * (2 * rand - 1)
    return out

def create_random_motion_control_points(n_control_points: int,
                                            amplitude_degree: float,
                                            amplitude_translation: float,
                                            device: str):

    tx = random_control_points(n_control_points=n_control_points, amplitude=amplitude_translation)
    ty = random_control_points(n_control_points=n_control_points, amplitude=amplitude_translation)
    tz = random_control_points(n_control_points=n_control_points, amplitude=amplitude_translation)
    rx = random_control_points(n_control_points=n_control_points, amplitude=amplitude_degree)
    ry = random_control_points(n_control_points=n_control_points, amplitude=amplitude_degree)
    rz = random_control_points(n_control_points=n_control_points, amplitude=amplitude_degree)

    control_points = torch.cat([tx, ty, tz, rx, ry, rz]).to(device)
    
    return control_points

@torch.no_grad()
def create_rigid_motion_pattern(timesteps: int,
                                  n_control_points: int,
                                  amplitude_degree: float,
                                  amplitude_translation: float,
                                  device: str):
    

    control_points = create_random_motion_control_points(n_control_points,
                                                    amplitude_degree,
                                                    amplitude_translation,
                                                    device)
    timesteps = torch.linspace(0, 1, timesteps).unsqueeze(0).to(device)

    rigid_motion_theta = RigidMotionTheta(n_control_points=n_control_points)
    rigid_motion_theta.eval()
    rigid_motion_theta.set_control_points(control_points)
    motion_patterns = rigid_motion_theta(timesteps)

    return motion_patterns


@torch.no_grad()
def create_motion_measurements(physics: Physics,
                                x: torch.Tensor,
                                thetas: torch.Tensor,
                                angles: torch.Tensor,
                                angle_batch_size: int) -> torch.Tensor:
    
    with torch.no_grad():

        n_timesteps= angles.shape[0]
        n_chunks = n_timesteps // angle_batch_size
        x_chunk = x.expand(angle_batch_size, -1, -1, -1, -1)

        y_chunks = []

        for idx_chunk in range(n_chunks):

            z_min = idx_chunk * angle_batch_size
            z_max = (idx_chunk + 1) * angle_batch_size
            sub_angles = angles[z_min:z_max]
            y_chunk = physics.transform(x_chunk, thetas[z_min:z_max, ...], sub_angles)
            y_chunks.append(y_chunk.detach().cpu())

        y_chunks = torch.cat(y_chunks, dim = 2)    
        y_chunks = y_chunks.to(x.device)

    return y_chunks


def extract_ts(obj: torch.Tensor,
               n_extracts: int) -> torch.Tensor:
    n = obj.shape[0]
    steps = n // n_extracts
    indexes = torch.torch.arange(0, n, steps).to(obj.device)
    return obj[indexes]


def extract_sino_ts(obj: torch.Tensor,
                    n_extracts: int) -> torch.Tensor:
    n = obj.shape[2]
    steps = n // n_extracts
    indexes = torch.torch.arange(0, n, steps).to(obj.device)
    return obj[:, :, indexes, :, :]


def create_initial_thetas(num_angles: int = 360):
    theta_identity = torch.tensor([[[1, 0, 0, 0], 
                                 [0, 1, 0, 0], 
                                 [0, 0, 1, 0]]], dtype=torch.float32)
    theta_identity = theta_identity.repeat(num_angles, 1, 1)

    return theta_identity

def m1_p1_norm(x: torch.Tensor) -> torch.Tensor:
    x = 2 * (x - x.min())/(x.max() - x.min()) - 1
    return x


def motion_pattern_to_theta(x: torch.Tensor):
    psi, theta, phi = torch.deg2rad(x[..., 3]), torch.deg2rad(x[..., 4]), torch.deg2rad(x[..., 5])
    output = torch.stack((torch.cos(psi) * torch.cos(theta),
                            torch.sin(
                                phi) * torch.sin(psi) * torch.cos(theta) - torch.cos(phi) * torch.sin(theta),
                            torch.cos(
                                phi) * torch.sin(psi) * torch.cos(theta) + torch.sin(phi) * torch.sin(theta),
                            x[..., 0],
                            torch.cos(psi) * torch.sin(theta),
                            torch.sin(phi) * torch.sin(psi) * torch.sin(theta) +
                            torch.cos(phi) * torch.cos(theta),
                            torch.cos(phi) * torch.sin(psi) * torch.sin(theta) -
                            torch.sin(phi) * torch.cos(theta),
                             x[..., 1],
                            - torch.sin(psi),
                            torch.sin(phi) * torch.cos(psi),
                            torch.cos(phi) * torch.cos(psi),
                            x[..., 2])).T #.flatten()
    output = output.view(x.shape[0], 3, 4)
    return output

