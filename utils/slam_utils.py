import torch
import os
import matplotlib.pyplot as plt

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False, rm_dynamic=False, mask=None, save_img=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, rm_dynamic=rm_dynamic, mask=mask, save_img=save_img)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=False, mask=None, save_img=False, save_img_path=None):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    # folder_path = config["Results"]["save_dir"] 
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid>0:
        rgb_pixel_mask = viewpoint.motion_mask.view(*mask_shape) * rgb_pixel_mask
        
    if mask is not None:
        rgb_pixel_mask = mask.view(*mask_shape) * rgb_pixel_mask
    
    
    if save_img:
        rgb_np = (image*rgb_pixel_mask).permute(1, 2, 0).cpu().detach().numpy()  # Change shape to (480, 640, 3)
        rgb_pixel_mask_np = rgb_pixel_mask.permute(1, 2, 0).cpu().detach().numpy()  # Change shape to (480, 640, 3)
        gt_image = viewpoint.original_image.cuda()
        _, h, w = gt_image.shape
        rgb_gt_np = (gt_image*rgb_pixel_mask).permute(1, 2, 0).cpu().detach().numpy()  # Change shape to (480, 640, 3)
        rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min())  # Normalize
        rgb_gt_np = (rgb_gt_np - rgb_gt_np.min()) / (rgb_gt_np.max() - rgb_gt_np.min())  # Normalize
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(rgb_np)
        axs[0].set_title("Rendered Static RGB Image")
        axs[1].imshow(rgb_gt_np, cmap="gray")
        axs[1].set_title("Ground truth RGB Image")
        axs[2].imshow(rgb_pixel_mask_np, cmap="gray")
        axs[2].set_title("Mask")
        
        # print(folder_path)
        for ax in axs:
            ax.axis("off")
        plt.savefig(f"{save_img_path}/tracking_{viewpoint.uid}_rgb_loss.png")  # Saves the figure to a file
        plt.close(fig)  # Close the figure to prevent memory leaks


    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False, rm_dynamic=False, mask=None, save_img=False
):
    folder_path = config["Results"]["save_dir"]
    subfolder_path = "tracking" 
    output_dir = os.path.join(folder_path, subfolder_path)

    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    depth_pixel_mask *= (gt_depth < 1000.).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, 
                                   image, 
                                   depth, 
                                   opacity, 
                                   viewpoint, 
                                   rm_dynamic=rm_dynamic, 
                                   mask=mask, 
                                   save_img=save_img, 
                                   save_img_path=output_dir)
    depth_mask = depth_pixel_mask * opacity_mask
    
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid>0:
        depth_mask = viewpoint.motion_mask.view(*depth.shape) * depth_mask
        
    if mask is not None:
        depth_mask = mask.view(*depth.shape) * depth_mask
        
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    
    # print("depth", depth.shape, depth_pixel_mask.shape)
    #VISUALIZATION
    if save_img:
        
        # plt.imshow(depth[0].detach().cpu().numpy(), cmap='plasma')
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Predicted Depth")
        plt.imshow((depth[0]*depth_mask[0]).detach().cpu().numpy(), cmap='plasma')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("GT Depth")
        plt.imshow((gt_depth[0]*depth_mask[0]).detach().cpu().numpy(), cmap='plasma')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("L1 Error (masked)")
        plt.imshow(l1_depth[0].detach().cpu().numpy(), cmap='hot')
        plt.colorbar()

        plt.tight_layout()

        plt.savefig(f"{output_dir}/tracking_{viewpoint.uid}_depth_loss.png")  # Saves the figure to a file
        # Close the figure to prevent memory leaks
        plt.close()

    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_network_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=False, mask=None, dynamic=False):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (opacity>0.95).view(*mask_shape)
    #rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid>0:
        rgb_pixel_mask = viewpoint.motion_mask.view(*mask_shape) * rgb_pixel_mask
    if mask is not None and rm_dynamic:
        rgb_pixel_mask = mask.view(*mask_shape) * rgb_pixel_mask
    l1 = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    #######
    if dynamic:
        if mask is not None:
            l1[(~viewpoint.motion_mask.view(*mask_shape)).repeat(3, 1, 1) | (~mask.view(*mask_shape)).repeat(3, 1, 1)] *= 3
        else:
            l1[~viewpoint.motion_mask.view(*mask_shape).repeat(3, 1, 1)] *= 3
    #######
    return l1.mean()

def pearson_loss(depth, viewpoint):
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=depth.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape) 
    
    rendered_depth = (depth*depth_pixel_mask).view(-1)
    true_depth = (gt_depth*depth_pixel_mask).view(-1)

    mean_rendered = torch.mean(rendered_depth)
    mean_true = torch.mean(true_depth)

    numerator = torch.sum((rendered_depth - mean_rendered) * (true_depth - mean_true))
    denominator = torch.sqrt(torch.sum((rendered_depth - mean_rendered) ** 2) * torch.sum((true_depth - mean_true) ** 2))

    correlation = numerator / denominator

    loss = 1 - correlation

    return loss


def get_loss_network(config, image, depth, viewpoint, opacity, initialization=False, rm_dynamic=False, mask=None, alpha=None, dynamic=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if alpha is None:
        alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.9
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)
    gt_image = viewpoint.original_image.cuda()
    l1_rgb = get_loss_network_rgb(config, image, depth, opacity, viewpoint, rm_dynamic=rm_dynamic, mask=mask, dynamic=dynamic)
    depth_mask = depth_pixel_mask * opacity_mask
    
    if viewpoint.motion_mask is not None and rm_dynamic and viewpoint.uid>0:
        depth_mask = viewpoint.motion_mask.view(*depth.shape) * depth_mask
    if mask is not None and rm_dynamic:
        depth_mask = mask.view(*depth.shape) * depth_mask
        
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    #####
    if dynamic:
        if mask is not None:
            l1_depth[(~viewpoint.motion_mask.view(*depth.shape)) | (~mask.view(*depth.shape))] *= 3
        else:
            l1_depth[~viewpoint.motion_mask.view(*depth.shape)] *= 3
    #######
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()
    
    

def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False, alpha=None, rm_dynamic=False, mask=None, dynamic=False, split=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint, alpha=alpha, rm_dynamic=rm_dynamic, mask=mask, dynamic=dynamic, split=split)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, alpha=None, rm_dynamic=False, mask=None, dynamic=False, split=False):
    if alpha is None:
        alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    depth_pixel_mask *= (gt_depth < 10000.).view(*depth.shape)
    
    if viewpoint.motion_mask is not None and rm_dynamic:  # and viewpoint.uid>0:
        rgb_pixel_mask = viewpoint.motion_mask.view(*depth.shape) * rgb_pixel_mask
        depth_pixel_mask = viewpoint.motion_mask.view(*depth.shape) * depth_pixel_mask
    if mask is not None and rm_dynamic:
        rgb_pixel_mask = mask.view(*depth.shape) * rgb_pixel_mask
        depth_pixel_mask = mask.view(*depth.shape) * depth_pixel_mask

    if split:
        motion_mask = viewpoint.motion_mask.view(*depth.shape)
        l1_static = alpha * torch.abs(motion_mask * rgb_pixel_mask*image-motion_mask * rgb_pixel_mask*gt_image).mean()
        l1_static += (1-alpha) * torch.abs(motion_mask * depth_pixel_mask*depth-motion_mask * depth_pixel_mask*gt_depth).mean()
        
        l1_dynamic = alpha * torch.abs((~motion_mask) * rgb_pixel_mask*image-(~motion_mask) * rgb_pixel_mask*gt_image).mean()
        l1_dynamic += (1-alpha) * torch.abs((~motion_mask) * depth_pixel_mask*depth-(~motion_mask) * depth_pixel_mask*gt_depth).mean()
        if dynamic:
            return l1_static, 2*l1_dynamic
        else:
            return l1_static, l1_dynamic
        
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    # print("image", image.shape, "rgb_pixel_mask", rgb_pixel_mask.shape, "gt_image", gt_image.shape)
    # print("depth", depth.shape, "depth_pixel_mask", depth_pixel_mask.shape, "gt_depth", gt_depth.shape)
    
    if False:
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        image_np = image_np.clip(0.0, 1.0)
        # Scale to [0, 255] and convert to uint8
        image_np = (image_np * 255.0).astype("uint8")
        
        # ---------- Save Depth ----------
        depth_np = depth.detach().cpu().squeeze().numpy()
        depth_viz = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

        # Convert pixel mask to numpy
        mask_np = rgb_pixel_mask.detach().cpu().squeeze().numpy()  # shape: [H, W]
        
        # Setup figure
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(image_np)
        axes[0].set_title("Rendered RGB")
        axes[0].axis("off")

        axes[1].imshow(depth_viz, cmap='plasma')
        axes[1].set_title("Rendered Depth")
        axes[1].axis("off")
        
        # RGB pixel mask (binary mask)
        axes[2].imshow(mask_np, cmap='gray')
        axes[2].set_title("RGB Pixel Mask")
        axes[2].axis("off")

        plt.tight_layout()
        viewpoint_id = viewpoint.uid
        # Save final image
        output_dir = os.path.join(config["Results"]["save_dir"], "mapping")
        os.makedirs(output_dir, exist_ok=True)

        viewpoint_id = viewpoint.uid
        save_path = os.path.join(output_dir, f"mapping_{viewpoint_id}_loss.png")
        plt.savefig(save_path)
        plt.close()
    ######
    if dynamic:
        if mask is not None:
            #l1_depth[(~viewpoint.motion_mask.view(*depth.shape)) | (~mask.view(*depth.shape))] *= 5
            #l1_rgb[((~viewpoint.motion_mask.view(*depth.shape)) | (~mask.view(*depth.shape))).repeat(3, 1, 1)] *= 5
            l1_depth[mask.view(*depth.shape).bool().detach()|~viewpoint.motion_mask.view(*depth.shape).detach()] *= 2
            l1_rgb[mask.view(*depth.shape).repeat(3, 1, 1).bool().detach()|~viewpoint.motion_mask.view(*depth.shape).repeat(3, 1, 1).detach()] *= 2
            #l1_rgb[((~viewpoint.motion_mask.view(*depth.shape)) | (mask.view(*depth.shape))).repeat(3, 1, 1)] *= 3
        else:
            l1_depth[~viewpoint.motion_mask.view(*depth.shape)] *= 2
            l1_rgb[~viewpoint.motion_mask.view(*depth.shape).repeat(3, 1, 1)] *= 2
    ######
    
    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
