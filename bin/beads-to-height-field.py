import alluvion as al
import alluvol
import numpy as np
from util import Unit, read_alu


def compute_height_field(recon_dir, episode_t):
    rl = np.load(f'{recon_dir}/kernel_radius_recon.npy').item()
    recon_raster_radius = rl * 0.36
    recon_voxel_size = recon_raster_radius / np.sqrt(3.0)
    container_width_real = 0.24

    truth_dir = '/media/kennychufk/mldata/alluvion-data/val-loop2/rltruth-073526d1-1222.15.37.04'
    sample_layer_x = np.copy(read_alu(f'{truth_dir}/sample-x.alu'))
    sample_layer_x[:, 1] = 10.0
    recon_x = read_alu(f'{recon_dir}/x-{episode_t}.alu')
    recon_ls = alluvol.create_liquid_level_set(recon_x, recon_raster_radius,
                                               recon_voxel_size)
    recon_ls.setGridClassAsLevelSet()
    recon_hf = alluvol.LevelSetRayIntersector(recon_ls).probe_heights(
        sample_layer_x, default_value=-container_width_real * 0.5)

    np.save(f"{recon_dir}/recon-hf-{episode_t}.npy", recon_hf)


for i in range(50, 1999, 50):
    compute_height_field(
        '/media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-piv-0.011/val-0416_103739-2v7m4muc-2600-685aa05c',
        i)
