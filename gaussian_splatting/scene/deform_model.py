import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, ControlNodeWarp, StaticNetwork
import os
#from utils.system_utils import searchForMaxIteration
from gaussian_splatting.utils.general_utils import get_expon_lr_func
import numpy as np
from gaussian_splatting.utils.general_utils import helper

def searchForMaxIteration(folder):
    if not os.path.exists(folder):
        return None
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "_" in fname]
    return max(saved_iters) if saved_iters != [] else None
    

model_dict = {'mlp': DeformNetwork, 'node': ControlNodeWarp, 'static': StaticNetwork}


class DeformModel:
    def __init__(self, deform_type='node', is_blender=False, d_rot_as_res=True, **kwargs):  # deform_type="node"
        self.deform = model_dict[deform_type](is_blender=is_blender, d_rot_as_res=d_rot_as_res, **kwargs).cuda()
        self.name = self.deform.name
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.d_rot_as_res = d_rot_as_res

    @property
    def reg_loss(self):
        return self.deform.reg_loss

    def step(self, xyz, time_emb, iteration=0, **kwargs):
        return self.deform(xyz, time_emb, iteration=iteration, **kwargs)

    def train_setting(self, training_args):
        l = [
            {'params': group['params'],
             'lr': training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale,
             "name": group['name']}
             for group in self.deform.trainable_parameters()
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale, lr_final=training_args.position_lr_final * training_args.deform_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.deform_lr_max_steps)
        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale
        self.lr_final = training_args.position_lr_final * training_args.deform_lr_scale
        self.lr_delay_mult=training_args.position_lr_delay_mult
        self.max_steps=training_args.deform_lr_max_steps
        #if self.name == 'node':
            #print(dir(self.deform))
        #    self.deform.as_gaussians().training_setup(training_args)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        #if iteration == -1:
        #    loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        #else:
        #    loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/deform.pth")
        if os.path.exists(weights_path):
            self.deform.load_state_dict(torch.load(weights_path))
            return True
        else:
            return False
            
    def extend_node_from_point(self, init_pcl, keep_all=False, force_init=False, reset_bbox=True, **kwargs):
        new_node_params = self.deform.extend_node(init_pcl, keep_all=keep_all, force_init=force_init, reset_bbox=reset_bbox, **kwargs)
        optimizable_tensors = {}
        param_list = self.deform.param_names
        param_idx = np.arange(len(param_list))
        for group in self.optimizer.param_groups:
            if group["name"] != 'nodes':
                continue
            #print("add...")
            for i in param_idx:
                stored_state = self.optimizer.state.get(group['params'][i], None)
                extension_tensor = new_node_params[i]
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                    del self.optimizer.state[group['params'][i]]
                    group["params"][i] = nn.Parameter(torch.cat((group["params"][i], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][i]] = stored_state
                    setattr(self.deform, param_list[i], group["params"][i])
                else:
                    group["params"][i] = nn.Parameter(torch.cat((group["params"][i], extension_tensor), dim=0).requires_grad_(True))
                    setattr(self.deform, param_list[i], group["params"][i])
        #print(len(self.deform.nodes), "   ", len(self.deform._node_radius))
        #self.deform.nodes = optimizable_tensors['nodes']
        #self.deform._node_radius = optimizable_tensors['radius']
        #self.deform._node_weight = optimizable_tensors['weight']

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform" or param_group["name"] == "mlp" or 'node' in param_group['name']:
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )
                param_group['lr'] = lr
                return lr
    
    def densify(self, max_grad, x, x_grad, **kwargs):
        if self.name == 'node':
            self.deform.densify(max_grad=max_grad, optimizer=self.optimizer, x=x, x_grad=x_grad, **kwargs)
        else:
            return
        
    def update(self, iteration):
        self.deform.update(iteration)
