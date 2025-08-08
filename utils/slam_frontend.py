import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render, get_dynamic_mask, render_flow
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth, get_loss_network, pearson_loss
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
import os
import matplotlib.pyplot as plt
import random


def vis_render_process(gaussians, pipeline_params, background, viewpoint, cur_frame_idx, save_dir, out_dir="track", mask=None):
    with torch.no_grad():
        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, mask=mask
        )
        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        #viz_depth = render_pkg['depth'][0, :, :].unsqueeze(0).detach().cpu()
        
        fig, ax = plt.subplots(figsize=(8, 8))  # the size of the figure
        cax = ax.imshow(viz_im)
        ax.axis('off')
        # save the figure
        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        return
        

        
        # print("vis the process")
        # fig, ax = plt.subplots(2, 3, figsize=(54, 34))
        # gt_image = viewpoint.original_image.cuda()
        # _, h ,w = gt_image.shape
        # mask_shape = (1, h, w)
        # rgb_mask = (gt_image.sum(dim=0) > 0.01).view(*mask_shape)
        # rgb_mask = rgb_mask * viewpoint.grad_mask
        # rgb_mask = rgb_mask.permute(1, 2, 0).detach().cpu()
        # viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        # viz_depth = render_pkg['depth'][0, :, :].unsqueeze(0).detach().cpu()
        # gt_im = torch.clip(viewpoint.original_image.permute(1, 2, 0).detach().cpu(), 0, 1)
        # gt_depth = viewpoint.depth
        # depth_mask = (torch.from_numpy(viewpoint.depth).to(dtype=torch.float32, device=viewpoint.device)
        #               [None] > 0.01).view(*render_pkg["depth"].shape)
        # opacity_mask = (render_pkg["opacity"] > 0.95).view(*render_pkg["depth"].shape)
        # depth_mask = depth_mask * opacity_mask
        # ax[0, 0].grid(False)
        # ax[0, 0].imshow(gt_im)
        # ax[0, 0].set_title("GT RGB", fontsize=30)
        # ax[0, 1].grid(False)
        # ax[0, 1].imshow(gt_depth, cmap='jet', vmin=0, vmax=6)
        # ax[0, 1].set_title("GT Depth", fontsize=30)
        # ax[0, 2].grid(False)
        # ax[0, 2].imshow(rgb_mask, cmap="gray")
        # ax[0, 2].set_title("rgb mask", fontsize=30)
        # ax[1, 0].grid(False)
        # ax[1, 0].imshow(viz_im)
        # ax[1, 0].set_title("render color", fontsize=30)
        # ax[1, 1].grid(False)
        # ax[1, 1].imshow(viz_depth[0], cmap='jet', vmin=0, vmax=6)
        # ax[1, 1].set_title("render depth", fontsize=30)
        # ax[1, 2].grid(False)
        # ax[1, 2].imshow(depth_mask.squeeze().detach().cpu(), cmap="gray")
        # ax[1, 2].set_title("depth mask", fontsize=30)
        # os.makedirs(save_dir, exist_ok=True)
        # process_dir = os.path.join(save_dir, out_dir)
        # os.makedirs(process_dir, exist_ok=True)
        # fig.suptitle(f"Frame: {cur_frame_idx}", y=0.95, fontsize=50)
        # plt.savefig(os.path.join(process_dir, f"tmp{cur_frame_idx}.png"))
        # plt.close()
        # print("vis complete")

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.dynamic_model = config["model_params"]["dynamic_model"]
        self.dynamic_objects = 0

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        #if not init and self.dynamic_model or self.config["Dataset"]["type"] == "CoFusion": # and self.config["Dataset"]["type"] != "CoFusion":
        #    initial_depth = initial_depth.detach().clone()  # change 0 region according to opacity rendering
        #    initial_depth[0][~viewpoint.motion_mask.cpu().numpy()] = 0
        #if not init and self.dynamic_model:
        if self.dynamic_model:
            initial_depth = initial_depth.detach().clone()  # change 0 region according to opacity rendering
            initial_depth[0][~viewpoint.motion_mask.cpu().numpy()] = 0
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    # def update_net(self, cur_frame_idx, viewpoint, windows, keyframe_list, last_keyframe_idx):
    #     print("update network")
    #     opt_params = []
    #     opt_params.append(
    #         {
    #             "params": [viewpoint.cam_rot_delta],
    #             "lr": self.config["Training"]["lr"]["cam_rot_delta"]*0.5,
    #             "name": "rot_{}".format(viewpoint.uid),
    #         }
    #     )
    #     opt_params.append(
    #         {
    #             "params": [viewpoint.cam_trans_delta],
    #             "lr": self.config["Training"]["lr"]["cam_trans_delta"]*0.5,
    #             "name": "trans_{}".format(viewpoint.uid),
    #         }
    #     )
    #     opt_params.append(
    #         {
    #             "params": [viewpoint.exposure_a],
    #             "lr": 0.01,
    #             "name": "exposure_a_{}".format(viewpoint.uid),
    #         }
    #     )
    #     opt_params.append(
    #         {
    #             "params": [viewpoint.exposure_b],
    #             "lr": 0.01,
    #             "name": "exposure_b_{}".format(viewpoint.uid),
    #         }
    #     )
    #     pose_optimizer = torch.optim.Adam(opt_params)
    #     random_viewpoint_stack = []
    #     #for cam_idx in keyframe_list:
    #     #    if cam_idx in windows:
    #     #        continue
    #     #    random_viewpoint_stack.append(cam_idx)
    #     for _ in range(70):
    #         scaling = 0
    #         loss = 0
    #         if len(self.cameras) > 0 and False:
    #             #rand_frame = random.sample(random_viewpoint_stack, 1)
    #             viewpoint_cam = self.cameras[0]
    #
    #             #time_input = self.gaussians.deform.deform.expand_time(viewpoint_cam.fid)
    #             #N = time_input.shape[0]
    #             #ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(20*cur_frame_idx+int(cur_frame_idx/5)*150)
    #             #d_values = self.gaussians.deform.step(self.gaussians.get_xyz.detach(), time_input+ast_noise,
    #             #                                      iteration=0, feature=None,
    #             #                                      motion_mask=self.gaussians.motion_mask,
    #             #                                      camera_center=viewpoint_cam.camera_center,
    #             #                                      time_interval=self.gaussians.time_interval)
    #             #dxyz = d_values['d_xyz'].detach()
    #             #d_rot, d_scale = d_values['d_rotation'].detach(), d_values['d_scaling'].detach()
    #             #scaling += d_scale
    #             render_pkg = render(
    #                 viewpoint_cam, self.gaussians, self.pipeline_params, self.background, dynamic=False, #dx=dxyz, ds=d_scale, dr=d_rot
    #             )
    #             image, depth, opacity = (
    #                 render_pkg["render"],
    #                 render_pkg["depth"],
    #                 render_pkg["opacity"],
    #             )
    #             #loss += 0.1*pearson_loss(depth, viewpoint_cam)
    #             image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
    #             gt_image = viewpoint_cam.original_image.cuda()
    #             gt_depth = torch.from_numpy(viewpoint_cam.depth).to(
    #                 dtype=torch.float32, device=image.device
    #             )[None]
    #             depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    #             l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    #             Ll1 = l1_loss(image, gt_image)
    #             loss += (1.0 - 0.2) * (Ll1) + 0.8 * (1.0 - ssim(image, gt_image))
    #             loss += 0.1*l1_depth.mean()
    #
    #             #loss += 1e-5 * self.gaussians.deform.deform.elastic_loss(t=viewpoint_cam.fid, delta_t=10*self.gaussians.time_interval) #, cur_time=viewpoint.time)  # viewpoint_cam
    #             #loss += 1e-5 * self.gaussians.deform.deform.acc_loss(t=viewpoint_cam.fid, delta_t=10*self.gaussians.time_interval) #, cur_time=viewpoint.time)
    #             #loss += 1e-4 * self.gaussians.deform.deform.arap_loss(t=viewpoint_cam.fid, delta_t=10*self.gaussians.time_interval, t_samp_num=8) #, cur_time=viewpoint.time)
    #
    #         time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
    #         N = time_input.shape[0]
    #         ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(20*cur_frame_idx+int(cur_frame_idx/5)*150)
    #         d_values = self.gaussians.deform.step(self.gaussians.get_xyz.detach(), time_input+ast_noise,
    #                                               iteration=0, feature=None,
    #                                               motion_mask=self.gaussians.motion_mask,
    #                                               camera_center=viewpoint.camera_center,
    #                                               time_interval=self.gaussians.time_interval)
    #         dxyz = d_values['d_xyz']
    #         d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
    #         scaling += d_scale
    #         render_pkg = render(
    #                 viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot
    #             )
    #         image, depth, opacity = (
    #             render_pkg["render"],
    #             render_pkg["depth"],
    #             render_pkg["opacity"],
    #         )
    #         #loss = get_loss_mapping(self.config, image, depth, viewpoint, opacity)
    #         #with torch.no_grad():
    #             #mask = viewpoint.reproject_mask(self.dataset, self.cameras[last_keyframe_idx])
    #             #render = render_flow()
    #
    #         loss += get_loss_network(self.config, image, depth, viewpoint, opacity, mask=mask, dynamic=True)
    #
    #         loss += 1e-4 * self.gaussians.deform.deform.arap_loss(t=viewpoint.fid, delta_t=2*self.gaussians.time_interval)#, cur_time=viewpoint.time)
    #         loss += 1e-4 * self.gaussians.deform.deform.elastic_loss(t=viewpoint.fid, delta_t=2*self.gaussians.time_interval)
    #         scaling += self.gaussians.get_scaling
    #         isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
    #         loss += 10 * isotropic_loss.mean()
    #
    #
    #
    #         #loss += 0.01*pearson_loss(depth, viewpoint)
    #         #loss += self.gaussians.deform.reg_loss
    #
    #         loss.backward()
    #
    #         with torch.no_grad():
    #             #self.gaussians.network_optimizer.step()
    #             #self.gaussians.network_optimizer.zero_grad(set_to_none=True)
    #             self.gaussians.optimizer.step()
    #             self.gaussians.optimizer.zero_grad(set_to_none=True)
    #             self.gaussians.deform.optimizer.step()
    #             self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
    #             #pose_optimizer.step()
    #             #pose_optimizer.zero_grad(set_to_none=True)
    #             #update_pose(viewpoint)
            

    def tracking(self, cur_frame_idx, viewpoint, last_keyframe_idx):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        
        
        viewpoint.update_RT(prev.R, prev.T)
        
        #viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        #with torch.no_grad():
        #    render_pkg = render(
        #            viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False
        #    )
        #    image, depth, opacity = (
        #            render_pkg["render"],
        #            render_pkg["depth"],
        #            render_pkg["opacity"],
        #    )
        #    self.median_depth = get_median_depth(depth, opacity)
        #return render_pkg
        
        
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )
        
        pose_optimizer = torch.optim.Adam(opt_params, capturable=True)
        # with torch.no_grad():
        #    dynamic_mask = get_dynamic_mask(viewpoint, self.gaussians, self.pipeline_params)
        # render_pkg = render(
        #    viewpoint, self.gaussians, self.pipeline_params, self.background, mask=dynamic_mask
        # )  
        # vis_render_process(self.gaussians, self.pipeline_params, self.background, viewpoint, 
        #                   cur_frame_idx, self.save_dir, out_dir="track", mask=dynamic_mask)  #self.gaussians._xyz
            
        #print("dynamic_mask: ", dynamic_mask.sum(), "gaussian_number: ",self.gaussians.get_xyz.shape[0] )
        
        # add dx, ds, dr
        with torch.no_grad():
            if False:
                time_input = self.gaussians.deform.deform.expand_time(torch.tensor(last_keyframe_idx).to(device=viewpoint.fid.device)) #if viewpoint.uid / 2 != 0 else self.gaussians.deform.deform.expand_time(viewpoint.fid-2)
                N = time_input.shape[0]
                #ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count)
                d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, #+ast_noise, 
                                                  iteration=0, feature=None, 
                                                  motion_mask=self.gaussians.motion_mask, 
                                                  camera_center=viewpoint.camera_center, 
                                                  time_interval=self.gaussians.time_interval)
                dxyz = d_values['d_xyz'].detach()
                d_rot, d_scale = d_values['d_rotation'].detach(), d_values['d_scaling'].detach()
            else:
                dxyz, d_rot, d_scale = None, None, None
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, mask=(self.gaussians.dygs==False)
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            
            # remove the dynamic object at frame 0
            with torch.no_grad():
                if self.dynamic_model and self.gaussians.deform_init and False:
                    mask = viewpoint.reproject_mask(self.dataset, self.cameras[last_keyframe_idx])
                else:
                    mask = None
            # print(image.shape, depth.shape, opacity.shape)

            save_tracking_loss = (tracking_itr % 10 == 0)

            loss_tracking = get_loss_tracking(
                self.config, 
                image, 
                depth, 
                opacity, 
                viewpoint, 
                rm_dynamic=True, 
                mask=mask, #not self.dynamic_model
                save_img = save_tracking_loss
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                pose_optimizer.zero_grad()
                self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        
        with torch.no_grad():
            render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot,
            )
            
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))
            
            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap, add_new_gaussian=True, dynamic_render=False):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, add_new_gaussian, dynamic_render]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 1 == 0:
            torch.cuda.empty_cache()
    
    # def dynamic_overlap(self, viewpoint, last_viewpoint):
    #     with torch.no_grad():
    #         if self.dynamic_model:
    #             time_input = self.gaussians.deform.deform.expand_time(torch.tensor(last_viewpoint.uid).to(device=viewpoint.fid.device))
    #             N = time_input.shape[0]
    #             d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input,
    #                                               iteration=0, feature=None,
    #                                               motion_mask=self.gaussians.motion_mask,
    #                                               camera_center=viewpoint.camera_center,
    #                                               time_interval=self.gaussians.time_interval)
    #             dxyz = d_values['d_xyz'].detach()
    #             d_rot, d_scale = d_values['d_rotation'].detach(), d_values['d_scaling'].detach()
    #         render_pkg = render(
    #             last_viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, mask=(self.gaussians.dygs==True))
    #         curr = render_pkg["opacity"] > 0.
    #         #union = torch.logical_or(curr, ~last_viewpoint.motion_mask).count_nonzero()
    #         intersection = torch.logical_and(curr, ~last_viewpoint.motion_mask).count_nonzero()
    #         ratio = intersection/curr.count_nonzero()
    #         print(ratio)
            
    def run(self):
        # init
        cur_frame_idx = 0
        last_keyframe_idx = 0
        # projection_matrix for viewpoints
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        # 
        dynamic_render=False
        keyframe_list = [0]
        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                if self.requested_keyframe > 0 and (cur_frame_idx - last_keyframe_idx) >= self.kf_interval:
                    time.sleep(0.01)
                    continue
                    
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )
                
                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint, last_keyframe_idx)
                #print(cur_frame_idx, "Tracking Complete ")
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                if self.dynamic_model:
                    self.gaussians.deform.deform.reg_loss = 0.  # Prevent deepcopy errors
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    print("skip")
                    continue

                last_keyframe_idx = self.current_window[0]
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf

                #if not create_kf and cur_frame_idx % 2 == 0 and self.dynamic_model and self.gaussians.deform_init:  
                #    self.update_net(cur_frame_idx, viewpoint, self.current_window, keyframe_list, last_keyframe_idx)
                    
                #if (cur_frame_idx - last_keyframe_idx) >= 5 and create_kf == False and point_ratio>0.95:
                #    add_new_gaussian = True
                #else:
                #    add_new_gaussian = True
                #if ~(~torch.all(viewpoint.motion_mask)):
                #    dynamic_render=True
                #else:
                #    dynamic_render=False
                #create_kf = ((cur_frame_idx - last_keyframe_idx) >= 5 and self.gaussians.deform_init and ~torch.all(viewpoint.motion_mask)) or create_kf  # + mask all True
                
                create_kf = ((cur_frame_idx - last_keyframe_idx) >= 5) or create_kf or cur_frame_idx == self.dystart
                
                #self.dynamic_overlap(viewpoint, self.cameras[last_keyframe_idx])
                
                
                if self.dataset.dynamic_objects > self.dynamic_objects and cur_frame_idx>0:
                    create_kf = True
                    new_object = True
                else:
                    new_object = False
                    
                if create_kf:
                    keyframe_list.append(cur_frame_idx)
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map, True, dynamic_render
                    )
                    
                    temp_log = ("create keyframe:", cur_frame_idx, 
                                "add new gaussian:", True, 
                                "point_ratio:", point_ratio, 
                                "dynamic_render:", dynamic_render)
                    
                    Log(tag="Frontend", *temp_log)
                    self.cameras[cur_frame_idx].clean_key()
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1
                self.dynamic_objects = self.dataset.dynamic_objects
                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
