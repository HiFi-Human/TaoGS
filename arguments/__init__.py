#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._resolution = 1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")
        
class MotionOptimizationParams:
    """Stage-1 optimizer settings used by the released motion training."""

    def __init__(self, iterations):
        self.iterations = iterations
        self.position_lr_init = 0.00016 * 2 / 5
        self.position_lr_final = 0.0000016 / 5
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01 / 5
        self.lambda_dssim = 0.2
        self.densification_interval = 300
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.00015
        self.topo_densify_from_iter = 3000
        self.topo_densification_interval = 300

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init_t2 = 0.00016 / 10

        self.position_lr_final_t2 = 0.0000016 / 10

        
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005 
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 500
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000

        self.densify_grad_threshold = 0.0002


        super().__init__(parser, "Optimization Parameters")

class LossParamsS1(ParamGroup):
    def __init__(self, parser):
        self.regular_term = True
        self.alpha_regular = 0.0005
        self.alpha_regular_position = 0.004
        self.graph_term = True
        self.alpha_rigid = 0.025
        self.isotropic_term = True
        self.alpha_isotropic = 0.001
        self.scaling_term = True
        self.alpha_scaling = 1
        self.scaling_threshold_coefficient = 6.0

        self.laplacian_term = True
        self.alpha_laplacian = 2
        self.repulsion_term = False
        self.alpha_repulsion = 10
        super().__init__(parser, "Motion Loss Parameters")


class TopoParams(ParamGroup):
    def __init__(self, parser):
        self.matches_per_ref = 8000
        self.num_refs = 180
        self.nns_per_ref = 1
        self.scaling_factor = 0.001
        self.proj_err_tolerance = 0.002
        self.roma_model = "outdoors"
        super().__init__(parser, "Topology Parameters")


class LossParamsS2():
    def __init__(self):
        self.regular_term = True
        self.alpha_regular = 1e-5
        self.alpha_regular_position = 0.1

        self.graph_term = True
        self.alpha_rigid = 0.0001

        self.isotropic_term = False
        self.scaling_term = True
        self.alpha_scaling = 0.05
        self.scaling_threshold = 5
        self.alpha_isotropic = 0.001
        self.laplacian_term = False
        self.alpha_laplacian = 5
        self.repulsion_term = False
        self.alpha_repulsion = 0.001

        self.father_graph_term = False
        self.alpha_father_rigid = 0.0001
