import random
from pathlib import Path

import numpy as np
import torch

from src.physics.composed_physics import ExtractorRadonRigidWarper
from src.physics.physics import Extractor, RigidWarper
from src.utils.creator_utils import (
    create_cone_beam_projector,
    create_motion_measurements,
    create_rigid_motion_pattern,
    create_volume,
    motion_pattern_to_theta,
)
from src.utils.dataloaders import add_noise_sino, get_head_ct_dataloaders
from src.utils.yaml_config import load_yaml


config_path = 'config/adm_jrm.yaml'
cfg = load_yaml(config_path)
common_cfg = cfg["common"]
acquisition_cfg = cfg["aquisition_params"]
data_cfg = cfg["generate_data"]

seed = common_cfg["seed"]
device = torch.device(common_cfg["device"])
batch_size = common_cfg["batch_size"]

angle_batch_size = acquisition_cfg["angle_batch_size"]
volume_size = acquisition_cfg["volume_size"]
n_base_angles = acquisition_cfg["n_base_angles"]

det_count_u = acquisition_cfg["det_count_u"]
det_count_v = acquisition_cfg["det_count_v"]
det_spacing_u = acquisition_cfg["det_spacing_u"]
det_spacing_v = acquisition_cfg["det_spacing_v"]
src_dist = acquisition_cfg["src_dist"]
det_dist = acquisition_cfg["det_dist"]

n_control_points = acquisition_cfg["n_control_points"]
motion_amplitude_degree = acquisition_cfg["motion_amplitude_degree"]
motion_amplitude_translation = acquisition_cfg["motion_amplitude_translation"]
noise_intensity = acquisition_cfg["noise_intensity"]

output_dir = Path(data_cfg["output_path"])
output_path = output_dir / "gts_and_measurements.pt"


def set_seed(seed=seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    dl_test = get_head_ct_dataloaders(root_dir=common_cfg["root_test"],
                                     batch_size=batch_size)

    angles = torch.linspace(0, 2 * torch.pi, n_base_angles).to(device)
    volume = create_volume(size=volume_size)
    conebeam = create_cone_beam_projector(angles=angles,
                                          volume=volume,
                                          det_count_u=det_count_u,
                                          det_count_v=det_count_v,
                                          det_spacing_u=det_spacing_u,
                                          det_spacing_v=det_spacing_v,
                                          src_dist=src_dist,
                                          det_dist=det_dist)

    extractor = Extractor()
    warper = RigidWarper()
    physics = ExtractorRadonRigidWarper(extractor=extractor,
                                        radon=conebeam,
                                        warper=warper)

    x = next(iter(dl_test))
    x = x.to(device)

    motion_patterns_gt = create_rigid_motion_pattern(timesteps=n_base_angles,
                                                     n_control_points=n_control_points,
                                                     amplitude_degree=motion_amplitude_degree,
                                                     amplitude_translation=motion_amplitude_translation,
                                                     device=device)

    thetas_gt = motion_pattern_to_theta(motion_patterns_gt)
    sino_motion_affected = create_motion_measurements(physics=physics,
                                                      x=x,
                                                      thetas=thetas_gt,
                                                      angles=angles,
                                                      angle_batch_size=angle_batch_size)

    b_tilde, yi = add_noise_sino(sino_motion_affected, I=noise_intensity)

    torch.save({
                'x_true': x,
                'true_b': sino_motion_affected,
                'b': b_tilde,
                'yi': yi,
                'thetas_gt': thetas_gt,
                'motion_pattern_est': motion_patterns_gt.detach(),
            }, output_path)

    print('saved')


if __name__ == "__main__":
    main()
