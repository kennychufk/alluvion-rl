import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import ndimage
from scipy import interpolate

matplotlib.use('Agg')


def calculate_masked_mse(v1, v2, mask):
    v_diff = v1 - v2
    masked_se = np.sum(v_diff * v_diff, axis=1) * mask
    num_masked = np.sum(mask)
    return np.sum(masked_se) / num_masked if num_masked > 0 else 0


def analyze_recon(recon_dir, filter_size=250):
    piv_dir = '/media/kennychufk/vol1bk0/' + recon_dir[6:recon_dir.
                                                       find('wr518z6d') - 1]
    h_start_pos = recon_dir.find('9000') + 5
    real_h = recon_dir[h_start_pos:recon_dir.find('-', h_start_pos)]
    # print(real_h)
    mask_hyphen_start_pos = recon_dir.find('-at')
    mask_postfix = ''
    if (mask_hyphen_start_pos > 0):
        mask_start_pos = mask_hyphen_start_pos + 3
        mask_size = recon_dir[mask_start_pos:recon_dir.
                              find('-', mask_start_pos)]
        mask_postfix = f'-at{mask_size}'
    # print(mask)
    f_start_pos = recon_dir.find('-f') + 2
    f_size = recon_dir[f_start_pos:f_start_pos + 2]

    vel_piv = np.load(f'{recon_dir}/piv/truth_v_real.npy')[..., [2, 1]]
    vel_recon = np.load(f'{recon_dir}/piv/sim_v_real.npy')[..., [2, 1]]
    sim_errors = np.load(f'{recon_dir}/piv/sim_errors.npy')

    mask = np.load(f'{piv_dir}/mat_results/mask{mask_postfix}.npy').astype(
        bool)
    piv_freq = 500.0

    num_frames = len(vel_piv)
    num_samples = vel_piv.shape[1]
    # print(num_frames)
    # print(num_samples)
    vel_piv = vel_piv.reshape(num_frames, num_samples, 2)
    np.nan_to_num(vel_piv, copy=False)
    zero_vector = np.zeros_like(vel_piv[0])
    vel_recon = vel_recon.reshape(num_frames, num_samples, 2)
    mask = mask.reshape(num_frames, num_samples)

    mse_sum = 0
    v2_sum = 0
    mse_list = np.zeros(num_frames)
    v2_sim_list = np.zeros(num_frames)
    v2_list = np.zeros(num_frames)

    for frame_id in range(num_frames):
        mse = calculate_masked_mse(vel_recon[frame_id], vel_piv[frame_id],
                                   mask[frame_id])
        piv_v2 = calculate_masked_mse(zero_vector, vel_piv[frame_id],
                                      mask[frame_id])
        sim_v2 = calculate_masked_mse(zero_vector, vel_recon[frame_id],
                                      mask[frame_id])

        mse_sum += mse
        v2_sum += piv_v2
        mse_list[frame_id] = mse
        v2_list[frame_id] = piv_v2
        v2_sim_list[frame_id] = sim_v2
    score = -mse_sum / v2_sum
    print(real_h, f_size, mask_postfix, score,
          np.load(f'{recon_dir}/piv/score.npy'))

    mse_filtered = ndimage.uniform_filter(mse_list,
                                          size=filter_size,
                                          mode='mirror')
    v2_filtered = ndimage.uniform_filter(v2_list,
                                         size=filter_size,
                                         mode='mirror')
    score_filtered = -mse_filtered / v2_filtered
    v2_sim_filtered = ndimage.uniform_filter(v2_sim_list,
                                             size=filter_size,
                                             mode='mirror')

    return score, score_filtered, v2_filtered, v2_sim_filtered


def render_field_6232963a(recon_dir, frame_id, score_filtered):
    fps = 30
    t = frame_id / fps
    truth_dir = recon_dir[6:recon_dir.find('-wr518z6d')]
    frame_dir = f"/media/kennychufk/vol1bk0/{truth_dir}/hstack-scaled-frames"
    my_dpi = 64
    img_filename = f"{frame_dir}/frame-{frame_id+1}.png"

    fig = plt.figure(figsize=[1920 // my_dpi, 960 // my_dpi],
                     dpi=my_dpi,
                     frameon=False)
    img_ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(img_ax)
    img = plt.imread(img_filename)

    img_ax.imshow(img, cmap='gray')
    img_ax.set_axis_off()

    ax_box = fig.add_axes([0.305, 0.585, 0.355, 0.379])
    ax_box.xaxis.set_visible(False)
    ax_box.yaxis.set_visible(False)
    #     ax_box.set_zorder(1000)
    ax_box.patch.set_alpha(0.5)
    ax_box.patch.set_color('white')

    ax_score = fig.add_axes([0.35, 0.65, 0.3, 0.3])
    cmap = plt.get_cmap("tab10")
    piv_freq = 500
    ts = np.arange(len(score_filtered)) / piv_freq
    ax_score.plot(ts, score_filtered)
    ax_score.set_xlabel(r'$t$ (s)')
    ax_score.set_ylim(-1, 0)
    ax_score.set_ylabel(r"Time-averaged score")
    ax_score.set_xlim(0, 14)

    ax_score.axvspan(0.833, 4.85, color=cmap(1), alpha=0.5)
    ax_score.axvspan(5.083, 9.1, color=cmap(3), alpha=0.5)
    ax_score.annotate("Diagonal 1",
                      xy=(2.7, -0.1),
                      xycoords="data",
                      va="center",
                      ha="center",
                      bbox=dict(boxstyle="square,pad=0.3",
                                fc="w",
                                ec="black",
                                lw=1.5,
                                alpha=0.5))

    ax_score.annotate("Diagonal 2",
                      xy=(7.4, -0.1),
                      xycoords="data",
                      va="center",
                      ha="center",
                      bbox=dict(boxstyle="square,pad=0.3",
                                fc="w",
                                ec="black",
                                lw=1.5,
                                alpha=0.5))
    ax_score.xaxis.set_ticks(np.arange(0, 20.5, 1))
    ax_score.yaxis.set_ticks(np.arange(-1, 0.05, 0.1))

    score_interp = interpolate.interp1d(ts, score_filtered)

    if t < 20:
        ax_score.axvline(x=t, color=cmap(2))
        inst_score = score_interp(t)
        if inst_score > -1:
            ax_score.plot((0, t), (inst_score, inst_score), color=cmap(2))

    ax_score.patch.set_alpha(0.8)

    fig.savefig(f'{frame_dir}/framemod-{frame_id+1}.png', dpi=my_dpi)
    #     [left, bottom, width, height]

    plt.close('all')


selected_hc_dir = "recon-20210416_103739-wr518z6d-9000-0.011-at0.0424264-f18-6232963a/"
plt.rcParams.update({'font.size': 22})

score, score_filtered, v2_filtered, v2_sim_filtered = analyze_recon(
    selected_hc_dir, filter_size=200)

for frame_id in range(405, 602):
    render_field_6232963a(selected_hc_dir, frame_id, score_filtered)
