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

import os
import random
import json
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, cameraList_from_camInfos_parallel, camera_to_JSON
from rich.console import Console
CONSOLE = Console(width=120)
class Scene:

    gaussians : GaussianModel
    gaussians_canonical: GaussianModel
    def __init__(self, args : ModelParams, gaussians_canonical : GaussianModel, shuffle=True, resolution_scales=[1.0], dynamic = False, load_frame_id = -1, ply_path = None, cpc_path = None,
                  parallel_load = False, no_image=False, sparse_view=False, stage = 1,
                  load_point_cloud=True):
        """Load sequence cameras and initialize the frame's Gaussian model."""
        self.model_path = args.model_path
        self.gaussians_canonical = gaussians_canonical
        self.dynamic = dynamic
        if cpc_path == None:
            self.cpc_path = args.model_path
        else:
            self.cpc_path = cpc_path

        self.train_cameras = {}
        self.test_cameras = {}

        if (os.path.exists(os.path.join(args.source_path, "transforms.json"))
                or os.path.isdir(os.path.join(args.source_path, "colmap", "sparse"))
                or os.path.isdir(os.path.join(args.source_path, "sparse"))):
            print("Reading sequence cameras")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, load_frame_id = load_frame_id, ply_path = ply_path, no_image=no_image, sparse_view=sparse_view, stage=stage, load_point_cloud=load_point_cloud)
        else:
            assert False, "Could not recognize scene type!"

        self.gaussians_canonical.masks = scene_info.masks

        if load_point_cloud and scene_info.ply_path is not None:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        for resolution_scale in resolution_scales:
            if not parallel_load:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            else:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos_parallel(scene_info.train_cameras, resolution_scale, args, no_image=no_image)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos_parallel(scene_info.test_cameras, resolution_scale, args, no_image=no_image)
   
        if self.dynamic and load_point_cloud:
            try:
                CONSOLE.log("Loading dynamic point cloud", os.path.join(self.cpc_path,
                                                            "cpc",
                                                            "point_cloud_%d.ply") % load_frame_id)
                self.gaussians_canonical.load_ply(os.path.join(self.cpc_path,
                                                            "cpc",
                                                            "point_cloud_%d.ply") % load_frame_id, self.cameras_extent)
               

            except:
                CONSOLE.log("Could not load dynamic point cloud, creating from scratch")
                self.gaussians_canonical.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        elif load_point_cloud:
            self.gaussians_canonical.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        else:
            CONSOLE.log("Skipping point-cloud initialization; caller will initialize Gaussian attributes")


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians_canonical.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
