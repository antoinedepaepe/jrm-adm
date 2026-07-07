import torch
import matplotlib.pyplot as plt


def plot_motion_pattern(motion_pattern: torch.Tensor):
    
    # Check shapes
    motion_pattern = motion_pattern.T.cpu().detach()

    print("GT Shape:", motion_pattern.shape)

    # Plot time series
    plt.figure(figsize=(12, 6))
    for i in range(0, motion_pattern.shape[0]):  # Iterate over 6 motion patterns
        plt.plot(motion_pattern[i, :], label=f'GT {i+1}', linestyle='dashed')

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Motion Patterns")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tensor_volume(x: torch.Tensor, save_path=None, vmin=None, vmax=None):

    x = x.detach().cpu().squeeze().numpy()
    D, H, W = x.shape  # Example dimensions
    # Extract middle slices along each dimension
    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2

    slice_d = x[mid_d, :, :]  # Middle slice along D-axis
    slice_h = x[:, mid_h, :] # Middle slice along H-axis
    slice_w = x[:, :, mid_w]# Middle slice along W-axis

    # Plot slices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    imshow_kwargs = {"cmap": "gray", "vmin": vmin, "vmax": vmax}

    axes[0].imshow(slice_d, **imshow_kwargs)
    axes[0].set_title(f"Slice along D-axis (D={mid_d})")

    axes[1].imshow(slice_h, **imshow_kwargs)
    axes[1].set_title(f"Slice along H-axis (H={mid_h})")

    axes[2].imshow(slice_w, **imshow_kwargs)
    axes[2].set_title(f"Slice along W-axis (W={mid_w})")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def save_slices(x, root, prefix, s=100):
    x = x.detach().cpu()[0, 0]
    plt.imsave(f'{root}/{prefix}_1.png', x[s, :, :], cmap='gray')
    plt.imsave(f'{root}/{prefix}_2.png', x[:, s, :], cmap='gray')
    plt.imsave(f'{root}/{prefix}_3.png', x[:, :, s], cmap='gray')