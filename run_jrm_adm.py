import numpy as np
import random
from pathlib import Path

import torch

from src.sampler.adaptative_diffusion_sampler import AdaptativeDiffusionSampler
from src.sampler.diffusion_utils import OneStepDDIMSampler
from src.optim.data_fidelity_grad import JRMDataFidelity
from src.models.rigid_motion_theta import RigidMotionTheta
from src.physics.composed_physics import ExtractorRadonRigidWarper
from src.physics.physics import Extractor, RigidWarper, Wavelet
from src.optim.rmsprop_motion_estimation import RMSpropMotionEstimator
from src.optim.rmsprop_x_estimation import RMSpropXEstimator
from src.utils.creator_utils import (
    create_cone_beam_projector,
    create_initial_thetas,
    create_model,
    create_volume,
    extract_sino_ts,
    extract_ts,
)
from src.utils.dataloaders import convert_m1_p1_to_attenuation
from src.utils.utils_plot import save_slices
from src.utils.yaml_config import load_yaml

config_path = 'config/adm_jrm.yaml'
cfg = load_yaml(config_path)
common_cfg = cfg["common"]
acquisition_cfg = cfg["aquisition_params"]
run_cfg = cfg["adm_jrm"]
diffusion_cfg = cfg["diffusion"]

seed = common_cfg["seed"]
device = torch.device(common_cfg["device"])

model_path = run_cfg["model_path"]

volume_size = acquisition_cfg["volume_size"]
n_angles = acquisition_cfg["n_angles"]
n_base_angles = acquisition_cfg["n_base_angles"]
angle_batch_size = acquisition_cfg["angle_batch_size"]

det_count_u = acquisition_cfg["det_count_u"]
det_count_v = acquisition_cfg["det_count_v"]
det_spacing_u = acquisition_cfg["det_spacing_u"]
det_spacing_v = acquisition_cfg["det_spacing_v"]
src_dist = acquisition_cfg["src_dist"]
det_dist = acquisition_cfg["det_dist"]

n_control_points = acquisition_cfg["n_control_points"]
motion_lr = run_cfg["motion_lr"]
x_lr = run_cfg["x_lr"]
gamma = run_cfg["gamma"]

beta_start = diffusion_cfg["beta_start"]
beta_end = diffusion_cfg["beta_end"]
diffusion_timesteps = diffusion_cfg["diffusion_timesteps"]
steps = diffusion_cfg["steps"]
eta = diffusion_cfg["eta"]
sampler_device = run_cfg.get("sampler_device", device)

idx = run_cfg.get("idx", 0)
input_path_template = run_cfg["input_path"]
output_path = Path(run_cfg["output_path"]) / f"adm_jrm_{n_angles}_view.pt"
plot_output_dir = run_cfg["plot_output_dir"]


def set_seed(seed=seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(plot_output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    model = create_model()
    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    angles = torch.linspace(0, 2 * torch.pi, n_base_angles)[
        torch.arange(0, n_base_angles, n_base_angles // n_angles)
    ].to(device)

    volume = create_volume(size=volume_size)
    conebeam = create_cone_beam_projector(angles=angles,
                                          volume=volume,
                                          det_count_u=det_count_u,
                                          det_count_v=det_count_v,
                                          det_spacing_u=det_spacing_u,
                                          det_spacing_v=det_spacing_v,
                                          src_dist=src_dist,
                                          det_dist=det_dist)

    to_attenuation = convert_m1_p1_to_attenuation()
    extractor = Extractor()
    warper = RigidWarper()
    wavelet = Wavelet()
    physics = ExtractorRadonRigidWarper(extractor=extractor,
                                        radon=conebeam,
                                        warper=warper)

    path = input_path_template.format(idx=idx)
    data = torch.load(path)

    x = data['x_true']
    thetas_gt = data['thetas_gt']
    thetas_gt_extract = extract_ts(thetas_gt, n_angles)

    # plot GT
    middle = int(thetas_gt_extract.shape[0] / 2)
    x_true_to_plot = physics.warper.transform(
        x,
        thetas_gt_extract[middle:middle+1, ...],
    )
    save_slices(x_true_to_plot, plot_output_dir, prefix='x_true')


    b_tilde = extract_sino_ts(data['b'], n_extracts=n_angles)
    yi = extract_sino_ts(data['yi'], n_extracts=n_angles)
    thetas = create_initial_thetas(num_angles=n_angles).to(device)

    print(f"b_tilde: {b_tilde.shape}")
    print(f"yi: {yi.shape}")
    print(f"thetas: {thetas.shape}")

    data_fidelity = JRMDataFidelity(physics=physics,
                                                yi=yi,
                                                angles=angles,
                                                angle_batch_size=angle_batch_size)

    motion_pattern_generator = RigidMotionTheta(n_control_points=n_control_points,
                                                n_angles=n_angles,
                                                n_base_angles=n_base_angles)
    motion_pattern_generator.train()

    motion_solver = RMSpropMotionEstimator(physics=physics,
                                            motion_pattern_generator=motion_pattern_generator,
                                            angles=angles,
                                            lr=motion_lr,
                                            yi=yi,
                                            angle_batch_size=angle_batch_size)

    x_solver = RMSpropXEstimator(data_fidelity=data_fidelity,
                                 lr=x_lr)

    one_step_sampler = OneStepDDIMSampler(beta_start=beta_start,
                                          beta_end=beta_end,
                                          T=diffusion_timesteps,
                                          steps=steps,
                                          eta=eta,
                                          device=sampler_device)

    sampler = AdaptativeDiffusionSampler(model=model,
                                         operator=wavelet,
                                         data_fidelity=data_fidelity,
                                         x_solver=x_solver,
                                         motion_solver=motion_solver,
                                         one_step_sampler=one_step_sampler)


    xt = wavelet.transform(torch.randn_like(x))
    x_est, thetas_est, cps_est, motion_pattern_full_est = sampler.sample(
        xt=xt,
        thetas=thetas,
        b=b_tilde,
        gamma=gamma,
        root_plot=plot_output_dir,
    )

    x_est = to_attenuation(x_est)

    torch.save({
                'x_est': x_est,
                'thetas_est': thetas_est,
                'cps_est': cps_est,
                'motion_pattern_full_est': motion_pattern_full_est,
            }, output_path)

    #NOTE: if you want to plot the motion pattern with the GT motion pattern
    # you will need to realign the translation/rotation curves to make them comparable.
    # This is due to the fact that the reconstructed volume x can be in any position, as
    # long as it aligns with the projection when warped and projected.

if __name__ == "__main__":
    main()
