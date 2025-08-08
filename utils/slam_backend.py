import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render, render_flow
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, get_loss_network, pearson_loss
import os
import matplotlib.pyplot as plt 
import numpy as np
# import imageio.v2 as imageio  # or imageio.v3 if needed

def vis_render_process(gaussians, pipeline_params, background, viewpoint, cur_frame_idx, save_dir, out_dir="map", mask=None, dynamic=False):
    with torch.no_grad():
        if dynamic:
            time_input = gaussians.deform.deform.expand_time(viewpoint.fid)
            d_values = gaussians.deform.step(gaussians.get_dygs_xyz.detach(), time_input, 
                                             iteration=0, feature=None, 
                                             motion_mask=gaussians.motion_mask, 
                                             camera_center=viewpoint.camera_center, 
                                             time_interval=gaussians.time_interval)
            dxyz = d_values['d_xyz']
            d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
            #print("scale: ", d_scale)
        else:
            dxyz, d_rot, d_scale = 0, 0, 0
        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, mask=mask,
        )
        #render_pkg = render(
        #    viewpoint, gaussians, pipeline_params, background, mask=mask, dynamic=False
        #)
        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        #viz_depth = render_pkg['depth'][0, :, :].unsqueeze(0).detach().cpu()
        
        h, w, _ = viz_im.shape
        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100) 
        cax = ax.imshow(viz_im)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}.png")
        plt.savefig(save_path)
        plt.close()
        return
        
class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.sc_params = None
        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.dynamic_model = config["model_params"]["dynamic_model"]

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]
        
        self.save_dir = self.config["Results"]["save_dir"]
        
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Training"]["single_thread"]
            if "single_thread" in self.config["Training"]
            else False
        )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,
        )
        if frame_idx == self.dystart:
            self.gaussians.extend_from_pcd_seq(
                viewpoint, kf_id=frame_idx, init=True, scale=scale, depthmap=depth_map, add_dygs=True
            )
        #if self.dynamic_model and frame_idx == 0:
        #    depth_map_clone = np.copy(depth_map)
        #    depth_map_clone[viewpoint.motion_mask] = 0 
        #    self.gaussians.extend_from_pcd_seq(
        #        viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map_clone)
                
        #if self.dynamic_model and frame_idx>0 and not self.gaussians.deform_init:
        #    first_init = self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
        #    if first_init:
        #        self.initialize_network(frame_idx, viewpoint)
        # Cofusion
        #if self.gaussians.deform_init and self.config["Dataset"]["type"] == "CoFusion":
        #    self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
            
    def add_next_node(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        depth = np.copy(viewpoint.depth)
        depth[viewpoint.get_mask_outlier().cpu().numpy()] = 0
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth
        )
        if self.dynamic_model and frame_idx>0 and not self.gaussians.deform_init:
            first_init = self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
            if first_init:
                self.initialize_network(frame_idx, viewpoint)
        elif self.gaussians.deform_init and self.config["Dataset"]["type"] == "CoFusion":
            self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
        self.initialize_network(frame_idx, viewpoint, update_gaussians=True)

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
    
    def initialize_network(self, cur_frame_idx, viewpoint, update_gaussians=False):
        if cur_frame_idx == self.dystart:
            inited = self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
            if not inited:
                return
        #self.gaussians.deform.deform.init(opt=self.opt_params, init_pcl=self.gaussians.get_xyz, keep_all=True, force_init=True, reset_bbox=False)
        #self.gaussians.deform.train_setting(self.sc_params)
        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
        for mapping_iteration in range(100):
            d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                  iteration=0, feature=None, 
                                                  motion_mask=self.gaussians.motion_mask, 
                                                  camera_center=viewpoint.camera_center, 
                                                  time_interval=self.gaussians.time_interval)#, detach_node=False)
            dxyz = d_values['d_xyz']
            #d_rot, d_scale = 0., 0.
            d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True,
            )
            #loss_init += self.gaussians.deform.reg_loss
                                                               
            #scaling = self.gaussians.get_scaling
            #isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            #loss_init += 10 * isotropic_loss.mean()
            
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                self.gaussians.deform.optimizer.step()
                self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                if update_gaussians:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        
        vis_render_process(self.gaussians, self.pipeline_params, self.background, viewpoint, 
                           viewpoint.uid, self.save_dir, out_dir="mapping", mask=None, dynamic=True)
        Log("Initialized mlp", tag="Backend")
        
    
    def initialize_map(self, cur_frame_idx, viewpoint):
    
        #self.gaussians.deform.deform.init(opt=self.opt_params, init_pcl=self.gaussians.get_xyz, keep_all=True, force_init=True, reset_bbox=False)
        #self.gaussians.deform.train_setting(self.sc_params)
        for mapping_iteration in range(self.init_itr_num):  # self.init_itr_num
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, 
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True, rm_dynamic=not (self.dystart==cur_frame_idx)
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map", tag="Backend")
        #vis_render_process(self.gaussians, self.pipeline_params, self.background, viewpoint, 
        #                   viewpoint.uid, self.save_dir, out_dir="map", mask=None, dynamic=False)
        return render_pkg
    
    def find_closest_keyframe(self, uid):
        keys = [key for key in self.viewpoints if key < uid]
        if not keys:
            return None
        closest_key = max(keys)
        return closest_key

    def map(self, current_window, prune=False, iters=1, dynamic_network=False, dynamic_render=False, rm_initdy=False):
        if len(current_window) == 0:
            return
        #
        key_opt = []
        if len(current_window) > 3:
            key_opt = self.viewpoints[current_window[0]].keyframe_selection_overlap(self.dataset, self.viewpoints, self.viewpoints[current_window[2]].uid)
        
        key_opt = current_window[:3] + key_opt
        
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in key_opt]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        current_window_set = set(key_opt)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        flow_weights = self.config["Training"]["flow_loss"]
        delta = self.config["Training"]["delta"] if "delta" in self.config["Training"] else 5
        
        #viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        #random_viewpoint_stack = []
        #frames_to_optimize = self.config["Training"]["pose_window"]
        #current_window_set = set(current_window)
        #for cam_idx, viewpoint in self.viewpoints.items():
        #    if cam_idx in current_window_set:
        #        continue
        #    random_viewpoint_stack.append(viewpoint)
            
        for i in range(iters):
            if i>100:
                self.iteration_count += 1
            loss_network = 0
            self.last_sent += 1
            dygs_scaling = 0
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
            
            if i < iters/2:
                dynamic = True
                flow_weights = self.config["Training"]["flow_loss"]
            else:
                dynamic = False
                flow_weights = self.config["Training"]["flow_loss_fine"] if "flow_loss_fine" in self.config["Training"] else self.config["Training"]["flow_loss"]  
                
            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                if dynamic_network and self.gaussians.deform_init:
                    time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                    N = time_input.shape[0]
                    #ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count)
                    d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                      iteration=0, feature=None, 
                                                      motion_mask=self.gaussians.motion_mask, 
                                                      camera_center=viewpoint.camera_center, 
                                                      time_interval=self.gaussians.time_interval)
                    dxyz = d_values['d_xyz']
                    d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                    d_opac, d_color=d_values['d_opacity'], d_values["d_color"]
                elif dynamic_render and self.gaussians.deform_init:
                    with torch.no_grad():
                        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                        N = time_input.shape[0]
                        ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count) 
                        d_values = self.gaussians.deform.step(self.gaussians.get_xyz.detach(), time_input+ast_noise, 
                                                          iteration=0, feature=None, 
                                                          motion_mask=self.gaussians.motion_mask, 
                                                          camera_center=viewpoint.camera_center, 
                                                          time_interval=self.gaussians.time_interval)
                        dxyz = d_values['d_xyz'].detach()
                        d_rot, d_scale = d_values['d_rotation'].detach(), d_values['d_scaling'].detach()
                        if d_values['d_opacity'] is not None: 
                            d_opac=d_values['d_opacity'].detach()
                        else:
                            d_opac =None
                        if d_values["d_color"] is not None: 
                            d_color = d_values["d_color"].detach()
                        else:
                            d_color=None
                else:
                    dxyz = 0
                    d_rot, d_scale, d_opac, d_color = None, 0, None, None
                dygs_scaling += d_scale
                
                render_pkg = render(
                    viewpoint, 
                    self.gaussians, 
                    self.pipeline_params, 
                    self.background, 
                    dynamic=False, 
                    dx=dxyz, 
                    ds=d_scale, 
                    dr=d_rot, 
                    do=d_opac, 
                    dc=d_color
                )
                
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                # print(f"Mapping {viewpoint.uid} with {viewpoint.original_image.shape} image, {viewspace_point_tensor.shape} points, {visibility_filter.sum()} visible points, depth: {depth.mean().item():.3f}")
                # Extract from render_pkg
                image = render_pkg["render"]  # torch.Size([3, H, W]), assumed in [0, 1]
                depth = render_pkg["depth"]   # torch.Size([1, H, W]) or [H, W]
                viewpoint_id = viewpoint.uid  # int or str

                # Setup output directory
                output_dir = os.path.join(self.config["Results"]["save_dir"], "mapping")
                os.makedirs(output_dir, exist_ok=True)
                
                if iters-1 == i and self.save_results:
                    #---------- Save RGB ----------
                    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
                    image_np = image_np.clip(0.0, 1.0)
                    # Scale to [0, 255] and convert to uint8
                    image_np = (image_np * 255.0).astype("uint8")
                    
                    # ---------- Save Depth ----------
                    depth_np = depth.detach().cpu().squeeze().numpy()
                    depth_viz = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                    # Setup figure
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].imshow(image_np)
                    axes[0].set_title("Rendered RGB")
                    axes[0].axis("off")

                    axes[1].imshow(depth_viz, cmap='plasma')
                    axes[1].set_title("Rendered Depth")
                    axes[1].axis("off")

                    plt.tight_layout()

                    viewpoint_id = viewpoint.uid
                    save_path = os.path.join(output_dir, f"mapping_{viewpoint_id}_loss.png")
                    plt.savefig(save_path)
                    plt.close()

                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[0])
                else:
                    mask = None
                    
                if dynamic_network and self.gaussians.deform_init:
                    closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                    if closest_keyframe is not None:# and i<iters*3/4:
                        #mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[closest_keyframe])
                        flow, flow_back, mask_fwd, mask_bwd = viewpoint.generate_flow(viewpoint.original_image.cuda(), self.viewpoints[closest_keyframe].original_image.cuda())
                        time_input = self.gaussians.deform.deform.expand_time(self.viewpoints[closest_keyframe].fid)
                        N = time_input.shape[0]
                        #ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count)
                        d_value2 = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                          iteration=0, feature=None, 
                                                          motion_mask=self.gaussians.motion_mask, 
                                                          camera_center=self.viewpoints[closest_keyframe].camera_center, 
                                                          time_interval=self.gaussians.time_interval)
                        d_xyz2 = d_value2["d_xyz"]
                        ## backward flow
                        render_pkg2 = render_flow(pc=self.gaussians, viewpoint_camera1=viewpoint, viewpoint_camera2=self.viewpoints[closest_keyframe], d_xyz1=dxyz, d_xyz2=d_xyz2, d_rotation1=d_rot, d_scaling1=d_scale, scale_const=None)
                        coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)

                        
                        # using motion_mask
                        dynamic_mask = (~viewpoint.motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()
                        #dynamic_mask |= render_pkg2["render"][2].unsqueeze(0).permute(1, 2, 0).repeat(1, 1, 2).detach().bool()
                        loss_network += flow_weights*l1_loss(flow_back*dynamic_mask, coor1to2_motion*dynamic_mask)
                        
                        ## forward flow
                        render_pkg_back = render_flow(pc=self.gaussians, viewpoint_camera1=self.viewpoints[closest_keyframe], viewpoint_camera2=viewpoint, d_xyz1=d_xyz2, d_xyz2=dxyz, d_rotation1=d_value2["d_rotation"], d_scaling1=d_value2["d_scaling"], scale_const=None)
                        coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
                        # using motion_mask
                        dynamic_mask = (~self.viewpoints[closest_keyframe].motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()

                        # print("coor1to2_motion",coor1to2_motion.shape,dynamic_mask.shape, flow.shape, flow_back.shape)
                        #dynamic_mask |= render_pkg_back["render"][2].unsqueeze(0).permute(1, 2, 0).repeat(1, 1, 2).detach().bool()
                        loss_network += flow_weights*l1_loss(flow*dynamic_mask, coor2to1_motion*dynamic_mask)  
                        if i==iters-1 and self.save_results:
                           viewpoint.save_flow(coor1to2_motion, save_path=os.path.join(output_dir, f"mapping_{viewpoint.uid}_flow1to2.png"))
                           viewpoint.save_flow(coor2to1_motion, save_path=os.path.join(output_dir, f"mapping_{viewpoint.uid}_flow2to1.png"))
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, rm_dynamic=not (dynamic_network or dynamic_render) , dynamic=dynamic
                    )
                        
                else:
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, rm_dynamic=not (dynamic_network or dynamic_render),
                    )
                #if dynamic_network and self.gaussians.deform_init: 
                #    loss_mapping += pearson_loss(depth, viewpoint)
                if dynamic_network and self.gaussians.deform_init:
                    loss_network += 1e-3 * self.gaussians.deform.deform.arap_loss(t=viewpoint.fid, delta_t=delta*self.gaussians.time_interval, t_samp_num=4)
                    loss_network += 1e-3 * self.gaussians.deform.deform.elastic_loss(t=viewpoint.fid, delta_t=5*self.gaussians.time_interval)
                
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                
                if dynamic_network and self.gaussians.deform_init:
                    time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                    N = time_input.shape[0]
                    #ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count)
                    d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                      iteration=0, feature=None, 
                                                      motion_mask=self.gaussians.motion_mask, 
                                                      camera_center=viewpoint.camera_center, 
                                                      time_interval=self.gaussians.time_interval)
                    dxyz = d_values['d_xyz']
                    d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                    d_opac, d_color=d_values['d_opacity'], d_values["d_color"]
                elif dynamic_render and self.gaussians.deform_init:
                    with torch.no_grad():
                        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                        N = time_input.shape[0]
                        ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count) 
                        d_values = self.gaussians.deform.step(self.gaussians.get_xyz.detach(), time_input+ast_noise, 
                                                          iteration=0, feature=None, 
                                                          motion_mask=self.gaussians.motion_mask, 
                                                          camera_center=viewpoint.camera_center, 
                                                          time_interval=self.gaussians.time_interval)
                        dxyz = d_values['d_xyz'].detach()
                        d_rot, d_scale = d_values['d_rotation'].detach(), d_values['d_scaling'].detach()
                        if d_values['d_opacity'] is not None: 
                            d_opac=d_values['d_opacity'].detach()
                        else:
                            d_opac =None
                        if d_values["d_color"] is not None: 
                            d_color = d_values["d_color"].detach()
                        else:
                            d_color=None
                else:
                    dxyz = 0
                    d_rot, d_scale, d_opac, d_color = None, 0, None, None
                dygs_scaling += d_scale
                
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[0])
                else:
                    mask = None
                if dynamic_network and self.gaussians.deform_init:
                    if dynamic or True:
                        closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                        #    with torch.no_grad():
                        #        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[closest_keyframe])
                        if closest_keyframe is not None:# and i<iters*3/4:
                            #mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[closest_keyframe])
                            flow, flow_back, mask_fwd, mask_bwd = viewpoint.generate_flow(viewpoint.original_image.cuda(), self.viewpoints[closest_keyframe].original_image.cuda())
                            time_input = self.gaussians.deform.deform.expand_time(self.viewpoints[closest_keyframe].fid)
                            N = time_input.shape[0]
                            #ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count)
                            d_value2 = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                              iteration=0, feature=None, 
                                                              motion_mask=self.gaussians.motion_mask, 
                                                              camera_center=self.viewpoints[closest_keyframe].camera_center, 
                                                              time_interval=self.gaussians.time_interval)
                            d_xyz2 = d_value2["d_xyz"]
                            ## backward flow
                            render_pkg2 = render_flow(pc=self.gaussians, viewpoint_camera1=viewpoint, viewpoint_camera2=self.viewpoints[closest_keyframe], d_xyz1=dxyz, d_xyz2=d_xyz2, d_rotation1=d_rot, d_scaling1=d_scale, scale_const=None)
                            coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)
                            # using motion_mask
                            dynamic_mask = (~viewpoint.motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()
                            #dynamic_mask |= render_pkg2["render"][2].unsqueeze(0).permute(1, 2, 0).repeat(1, 1, 2).detach().bool()
                            loss_network += flow_weights*l1_loss(flow_back*dynamic_mask, coor1to2_motion*dynamic_mask)
                            
                            ## forward flow
                            render_pkg_back = render_flow(pc=self.gaussians, viewpoint_camera1=self.viewpoints[closest_keyframe], viewpoint_camera2=viewpoint, d_xyz1=d_xyz2, d_xyz2=dxyz, d_rotation1=d_value2["d_rotation"], d_scaling1=d_value2["d_scaling"], scale_const=None)
                            coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
                            # using motion_mask
                            dynamic_mask = (~self.viewpoints[closest_keyframe].motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()
                            #dynamic_mask |= render_pkg_back["render"][2].unsqueeze(0).permute(1, 2, 0).repeat(1, 1, 2).detach().bool()
                            loss_network += flow_weights*l1_loss(flow*dynamic_mask, coor2to1_motion*dynamic_mask)

                        loss_mapping += get_loss_mapping(
                            self.config, image, depth, viewpoint, opacity, rm_dynamic=not (dynamic_network or dynamic_render) , dynamic=dynamic,
                        )
                    else:
                        image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                        gt_image = viewpoint.original_image.cuda()
                        gt_depth = torch.from_numpy(viewpoint.depth).to(
                            dtype=torch.float32, device=image.device
                        )[None]
                        depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
                        l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
                        Ll1 = l1_loss(image, gt_image)
                        loss_mapping += (1.0 - self.opt_params.lambda_dssim) * (Ll1) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                        loss_mapping += 0.1*l1_depth.mean()  #0.1*l1_depth.mean() 
                else:
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, rm_dynamic=not (dynamic_network or dynamic_render), mask=mask
                    )
                #if dynamic_network and self.gaussians.deform_init: 
                #    loss_mapping += pearson_loss(depth, viewpoint)
                
                if dynamic_network and self.gaussians.deform_init:
                    loss_network += 1e-4 * self.gaussians.deform.deform.elastic_loss(t=viewpoint.fid, delta_t=5*self.gaussians.time_interval)
                    #loss_mapping += 1e-5 * self.gaussians.deform.deform.acc_loss(t=viewpoint.fid, delta_t=5*self.gaussians.time_interval)#, cur_time=self.viewpoints[current_window[0]].time)
                    loss_network += 1e-4 * self.gaussians.deform.deform.arap_loss(t=viewpoint.fid, delta_t=5*self.gaussians.time_interval)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            
            loss_mapping.backward(retain_graph=True)
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                            #to_prune = torch.logical_or(torch.logical_and(self.gaussians.dygs==True, (self.gaussians.n_obs >= 1).cuda()), to_prune.cuda())  ##
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized 4DGS-SLAM", tag="Backend")
                        # # make sure we don't split the gaussians, break here.

                    if dynamic_network and self.gaussians.deform_init and self.iteration_count%3000 < 100 and False:
                        # densify
                        self.gaussians.deform.densify(max_grad=0.0006, 
                                                      x=self.gaussians.get_xyz, 
                                                      x_grad=self.gaussians.xyz_gradient_accum / self.gaussians.denom, 
                                                      feature=None, 
                                                      force_dp=True)
                    #vis_render_process(self.gaussians, self.pipeline_params, self.background, self.viewpoints[current_window[0]], 
                    #           self.viewpoints[current_window[0]].uid, self.save_dir, out_dir="map", mask=None, dynamic=True)
                    
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset and i>100
                ) 
                if rm_initdy:
                    update_gaussian = (iters - i-10 == 0)  # 
                    
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True
                    #dygs = self.gaussians.get_dygs_xyz.detach()
                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ) and i>100:
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                
                
                #self.gaussians.network_optimizer.step()
                #self.gaussians.network_optimizer.zero_grad(set_to_none=True)
                # Pose update
                if True: #not (dynamic_network or dynamic_render):
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
                else:
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                    
                if (dynamic_network) and self.gaussians.deform_init:
                    loss_network.backward()
                    self.gaussians.deform.optimizer.step()
                    self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                
                if i>100:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
        return gaussian_split
    
    
    def color_refinement(self, dynamic_network=False):
        Log("Starting color refinement")

        iteration_total = 1500
        for iteration in tqdm(range(1, iteration_total + 1)):
            loss = 0
            viewpoint_idx_stack = list(self.viewpoints.keys())
            for _ in range(10):
                scaling = 0
                viewpoint_cam_idx = viewpoint_idx_stack.pop(
                    random.randint(0, len(viewpoint_idx_stack) - 1)
                )
                viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
                
                if dynamic_network and self.gaussians.deform_init:
                    time_input = self.gaussians.deform.deform.expand_time(viewpoint_cam.fid)
                    N = time_input.shape[0]
                    #ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(iteration)
                    d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, #+ast_noise, 
                                                      iteration=0, feature=None, 
                                                      motion_mask=self.gaussians.motion_mask, 
                                                      camera_center=viewpoint_cam.camera_center, 
                                                      time_interval=self.gaussians.time_interval)
                    dxyz = d_values['d_xyz']
                    d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                    d_opac, d_color=d_values['d_opacity'], d_values["d_color"]
                else:
                    dxyz, d_rot, d_scale, d_opac, d_color = 0, 0, 0, None, None
                    
                render_pkg = render(
                        viewpoint_cam, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                )
                
                image, depth, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["depth"], 
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )
                image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
                gt_image = viewpoint_cam.original_image.cuda()
                gt_depth = torch.from_numpy(viewpoint_cam.depth).to(
                    dtype=torch.float32, device=image.device
                )[None]
                depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
                if dynamic_network:
                    Ll1 = l1_loss(image, gt_image)
                    loss += (1.0 - self.opt_params.lambda_dssim) * (
                        Ll1
                    ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                    loss += 1e-4 * self.gaussians.deform.deform.arap_loss(t=viewpoint_cam.fid, delta_t=5*self.gaussians.time_interval, t_samp_num=8)  #1e-1 * self.gaussians.deform.deform.arap_loss(t=viewpoint_cam.fid, delta_t=20*self.gaussians.time_interval)
                else:
                    Ll1 = l1_loss(image, gt_image, mask=viewpoint_cam.motion_mask)
                    loss += (1.0 - self.opt_params.lambda_dssim) * (
                        Ll1
                    ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image, mask=viewpoint_cam.motion_mask))
                    depth_pixel_mask = viewpoint_cam.motion_mask.view(*gt_depth.shape) * depth_pixel_mask
                    
                l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
                loss += 0.1*l1_depth.mean()  
                #loss += self.gaussians.compute_regulation()
                
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss += 10 * isotropic_loss.mean()
            #if dynamic_network:
            #    loss += 1e-2 * self.gaussians.deform.deform.arap_loss(t=viewpoint_cam.fid, delta_t=20*self.gaussians.time_interval)
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                #self.gaussians.network_optimizer.step()
                #self.gaussians.network_optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
                if dynamic_network and self.gaussians.deform_init:
                    self.gaussians.deform.optimizer.step()
                    self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
        if self.dynamic_model:
            self.gaussians.deform.deform.reg_loss = 0.  # Prevent deepcopy errors
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
    
    ## backend thread
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else: # get info from frondend
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement(dynamic_network=self.dynamic_model)
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system",tag="Backend")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    if self.dynamic_model and self.dystart==0:
                        self.initialize_network(cur_frame_idx, viewpoint)
                    
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    add_new_gaussian = data[5]
                    dynamic_render = data[6]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    #if new_object:
                    #    self.add_next_node(cur_frame_idx, viewpoint, depth_map=depth_map)
                    if add_new_gaussian:
                        self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    
                    if self.dynamic_model and self.dystart==cur_frame_idx:
                        self.initialize_map(cur_frame_idx, viewpoint)
                        self.initialize_network(cur_frame_idx, viewpoint)
                    
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    iter_per_kf = 70
                    #print(iter_per_kf)
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization", tag="Backend")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    if self.dystart > cur_frame_idx:  #
                        self.map_static(self.current_window, iters=int(20))  #
                        self.map_static(self.current_window, prune=True)  #
                    elif add_new_gaussian:
                        self.map(self.current_window, iters=int(200), dynamic_network=self.dynamic_model)
                        self.map(self.current_window, prune=True, dynamic_network=self.dynamic_model)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def map_static(self, current_window, prune=False, iters=1, dynamic_network=False, dynamic_render=False, rm_initdy=True):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        # dygs = self.gaussians.get_dygs_xyz.detach()
        for i in range(iters):
            self.iteration_count += 1
            self.last_sent += 1
            scaling = 0
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                dxyz = 0
                d_rot, d_scale, d_opac, d_color = None, 0, None, None

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz,
                    ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                )

                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[0])
                else:
                    mask = None

                
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity,
                    rm_dynamic=True, #mask=mask
                )

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]

                dxyz = 0
                d_rot, d_scale, d_opac, d_color = None, 0, None, None

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz,
                    ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[0])
                else:
                    mask = None
    
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity,
                    rm_dynamic=True, #mask=mask
                )

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]  # ֻprune
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask  # prune
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized 4DGS-SLAM", tag="Backend")
                        # # make sure we don't split the gaussians, break here.

                    vis_render_process(self.gaussians, self.pipeline_params, self.background,
                                       self.viewpoints[current_window[0]],
                                       self.viewpoints[current_window[0]].uid, self.save_dir, out_dir="map", mask=None,
                                       dynamic=(dynamic_network and self.gaussians.deform_init))

                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                        self.iteration_count % self.gaussian_update_every
                        == self.gaussian_update_offset
                )

                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True
                    # dygs = self.gaussians.get_dygs_xyz.detach()
                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                        not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians", tag="Backend")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                if True:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                if True:  # not (dynamic_network or dynamic_render):
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
                else:
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
        return gaussian_split