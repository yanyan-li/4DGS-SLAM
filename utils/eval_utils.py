import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torchvision
#from torchvision.utils import save_image

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def vis_render_process(render_pkg, gt_color, gt_depth, cur_frame_idx, save_dir, out_dir="fin_render"):
    #print("cur_frame_idx", cur_frame_idx)
    with torch.no_grad():
        fig, ax = plt.subplots(2, 2, figsize=(54, 34))
        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        viz_depth = render_pkg['depth'][0, :, :].unsqueeze(0).detach().cpu()
        gt_im = torch.clip(gt_color.permute(1, 2, 0).detach().cpu(), 0, 1)
        gt_depth = gt_depth
        ax[0, 0].grid(False)
        ax[0, 0].imshow(gt_im)
        ax[0, 0].set_title("GT RGB", fontsize=30)
        ax[0, 1].grid(False)
        ax[0, 1].imshow(gt_depth, cmap='jet', vmin=0, vmax=6)
        ax[0, 1].set_title("GT Depth", fontsize=30)
        ax[1, 0].grid(False)
        ax[1, 0].imshow(viz_im)
        ax[1, 0].set_title("render color", fontsize=30)
        ax[1, 1].grid(False)
        ax[1, 1].imshow(viz_depth[0], cmap='jet', vmin=0, vmax=6)
        ax[1, 1].set_title("render depth", fontsize=30)
        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        fig.suptitle(f"Frame: {cur_frame_idx}", y=0.95, fontsize=50)
        #plt.savefig(os.path.join(process_dir, f"{cur_frame_idx}.png"))
        plt.close()
        
        h, w, _ = viz_im.shape
        #print("cur_frame_idx", cur_frame_idx)
        #fig, ax = plt.subplots(figsize=(4, 4))  # 你可以调整图像的尺寸
        #if out_dir=="novel_render":
        #    fig, ax = plt.subplots(figsize=(w*2/100, h*2/100), dpi=100) 
        #else:
        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100) 
        cax = ax.imshow(viz_im)
        ax.axis('off')
        # 去除空白区域
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # 保存图像
        save_path = os.path.join(process_dir, f"{cur_frame_idx}_color.png")
        #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.savefig(save_path)
        plt.close()
        
        
        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100) 
        cax = ax.imshow(viz_depth[0], cmap='jet', vmin=0, vmax=6)
        # 显示深度图像，应用 'jet' 颜色映射
        #cax = ax.imshow((viz_depth[0] - gt_depth), cmap='jet', vmin=0, vmax=6)
        ax.axis('off')
        # 保存图像
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}_depth.png")
        plt.savefig(save_path)
        plt.close()
        
        return
        
        fig, ax = plt.subplots(figsize=(8, 8))  # 你可以调整图像的尺寸
        
        norm = mcolors.LogNorm(vmin=0.01, vmax=6) 
        mask = (gt_depth == 0)
    
        # 计算绝对差异
        abs_diff = np.abs(viz_depth[0] - gt_depth)
    
        # 应用掩码，将 depth 为 0 的部分的差异也设置为 0
        abs_diff[mask] = 0
        cax = ax.imshow(viz_depth[0], cmap='jet', vmin=0, vmax=6)
        # 显示深度图像，应用 'jet' 颜色映射
        #cax = ax.imshow((viz_depth[0] - gt_depth), cmap='jet', vmin=0, vmax=6)
        ax.axis('off')
        # 保存图像
        save_path = os.path.join(process_dir, f"{cur_frame_idx}_depth.png")
        #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        
def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close()
    return ape_stat


def write_pose(pose, save_dir):
    plot_dir = os.path.join(save_dir, "pose.txt")
    with open(plot_dir, "a") as f:
        for row in pose:
            c2w = ' '.join(str(x) for x in row)
            f.write(str(c2w)+"\n")
        f.close()


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error
    

def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  =  np.stack(gt_traj_pts).T
    est_traj_pts =  np.stack(est_traj_pts).T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error
            

def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []
    
    trj_est_ate, trj_gt_ate = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose
        
    if final == True:
        plot_dir = os.path.join(save_dir, "pose.txt")
        with open(plot_dir, "wb") as f:
            f.truncate()
            f.close()
        for frame_id in range(0, len(frames)):
            kf = frames[frame_id]
            pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
            write_pose(pose_est, save_dir)
            pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))
            trj_est_ate.append(pose_est)
            trj_gt_ate.append(pose_gt)
        ate_rmse = evaluate_ate(trj_gt_ate, trj_est_ate)
        print("\nTacking ATE is :",ate_rmse,"\n")
            
    #for kf_id in kf_ids:
    for kf_id in range(0, len(frames)):
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)
        
    if final == True:
        with open(
            os.path.join(plot_dir, "ATE_{}.json".format(iterations)),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ate_rmse, f, indent=4)
    
    try:
        ate = evaluate_evo(
            poses_gt=trj_gt_np,
            poses_est=trj_est_np,
            plot_dir=plot_dir,
            label=label_evo,
            monocular=monocular,
        )
    except Exception as e:
        print("running real dataset")
        ate = 0

    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    save_interval=None
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    depth_array = []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    #print(kf_indices)
    for idx in range(0, end_idx):
        #if idx in kf_indices:  # 是否不计算关键帧的指标
        #    continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, gt_depth, _, motion_mask = dataset[idx]

        #rendering = render(frame, gaussians, pipe, background)["render"]
        if gaussians.deform_init:
            time_input = gaussians.deform.deform.expand_time(frame.fid)
            d_values = gaussians.deform.step(gaussians.get_dygs_xyz.detach(), time_input, 
                                             iteration=0, feature=None, 
                                             motion_mask=gaussians.motion_mask, 
                                             camera_center=frame.camera_center, 
                                             time_interval=gaussians.time_interval)
            dxyz = d_values['d_xyz']
            d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
            #print("eval using deform network")
        else:
            dxyz, d_rot, d_scale = 0, 0, 0
        
        
        render_pkg = render(
            frame, gaussians, pipe, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot
        )

        if iteration == "before_opt":
            if (idx + 1) % save_interval == 0 or idx == 0:
                vis_render_process(render_pkg, gt_image, gt_depth, idx, save_dir, out_dir="before_opt")
        elif iteration == "after_opt":
            if (idx + 1) % save_interval == 0 or idx == 0:
                vis_render_process(render_pkg, gt_image, gt_depth, idx, save_dir, out_dir="after_opt")
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0
        
        valid_depth_mask = gt_depth > 0
        if not gaussians.deform_init:
            #print("eval remove motion region")
            mask = mask * motion_mask.view(*valid_depth_mask.shape) * torch.from_numpy(valid_depth_mask).to(device=motion_mask.device)
            valid_depth_mask = valid_depth_mask * motion_mask.view(*valid_depth_mask.shape).cpu().numpy()
        else:
            mask = mask * torch.from_numpy(valid_depth_mask).to(device=motion_mask.device) 
        
        l1_depth = np.abs((gt_depth[None] - render_pkg['depth'].cpu().detach().numpy()) * valid_depth_mask)
        #if iteration == "before_opt":
        #    print(l1_depth)
        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))
        depth_score = l1_depth.sum() / (valid_depth_mask.sum()+1e-7)
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())
        depth_array.append(depth_score)
        
        if iteration == "after_opt":
            with torch.no_grad():
                render_pkg_novel = render(
                    frame, gaussians, pipe, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, novel=1
                )
                vis_render_process(render_pkg_novel, gt_image, gt_depth, idx, save_dir, out_dir="novel_render")
                
    if iteration == "after_opt":
        for idx in range(0, end_idx):
            #if idx in kf_indices:  # 是否不计算关键帧的指标
            #    continue
            frame = frames[end_idx-1]
            if idx > end_idx/2:
                render_pkg_novel = render(
                    frame, gaussians, pipe, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, novel= -4/frame.uid
                )
            else:
                render_pkg_novel = render(
                    frame, gaussians, pipe, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, novel= 3/frame.uid
                )
            vis_render_process(render_pkg_novel, gt_image, gt_depth, idx, save_dir, out_dir="fin_novel_render")
            
    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["l1_depth"] = float(np.mean(depth_array))
    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, l1_depth: {output["l1_depth"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    #min_xyz = gaussians._xyz.min(dim=0)[0]
    #max_xyz = gaussians._xyz.max(dim=0)[0]
    #print("min_xyz: ", min_xyz, "max_xyz: ", max_xyz)
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
