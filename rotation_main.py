import os
import math
import torch
import viser.transforms as vst

import einops
from e3nn import o3
from einops import einsum

from scene import Scene
from gaussian_renderer import GaussianModel
from utils.gaussian_utils import GaussianTransformUtils

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    
    return max(saved_iters)

def transform_shs(shs_feat, rotation_matrix):

    ## rotate shs
    shs_feat = shs_feat.to("cuda")
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float().to('cuda') # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix.to('cuda') @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    rot_angles = [r.cpu() for r in rot_angles]
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2]).to('cuda')
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2]).to('cuda')
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2]).to('cuda')

    # rotation of the shs features
    shs_feat[:, 0:3] = D_1 @ shs_feat[:, 0:3]
    shs_feat[:, 3:8] = D_2 @ shs_feat[:, 3:8]
    shs_feat[:, 8:15] = D_3 @ shs_feat[:, 8:15]
    return shs_feat

def rotate_splat(args,dataset,rotation_matrix):
    with torch.no_grad():
        loaded_iter = args.iteration
        if args.iteration==-1:
            loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        rotated_output_path = os.path.join(dataset.model_path.replace("render_out","render_out_rot"),"point_cloud",f"iteration_{loaded_iter}", "point_cloud.ply")
        
        model = GaussianModel(dataset.sh_degree)
        Scene(dataset, model, load_iteration=args.iteration, shuffle=False,only_test=True)

        wigner_D_rotated_extra_shs = transform_shs(model.get_features[:, 1:, :].clone().cpu(), rotation_matrix.cpu())

        wigner_D_rotated_shs = model.get_features.clone().cpu()
        wigner_D_rotated_shs[:, 1:, :] = wigner_D_rotated_extra_shs.cpu()

        rotated_xyz = model.get_xyz @ torch.tensor(so3.as_matrix().T, device=model.get_xyz.device, dtype=torch.float)
        rotated_rotations = torch.nn.functional.normalize(GaussianTransformUtils.quat_multiply(
            model.get_rotation, 
            torch.tensor(so3.as_quaternion_xyzw()[[3, 0, 1, 2]], device=rotated_xyz.device, dtype=torch.float),
        ))

        model._xyz = rotated_xyz 
        model._rotation = rotated_rotations
        model.save_ply(rotated_output_path)


if __name__ == "__main__":

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)

    print("rotating " + args.model_path)

    x_ang, y_ang, z_ang = (90, 180, 0) #rotation angles in degrees (x,y,z)
    roll = math.radians(x_ang)   # Rotation around the x-axis
    pitch = math.radians(y_ang)  # Rotation around the y-axis
    yaw = math.radians(z_ang)    # Rotation around the z-axis

    so3 = vst.SO3.from_rpy_radians(roll, pitch, yaw)
    rotation_matrix = torch.tensor(so3.as_matrix(), dtype=torch.float, device="cuda")

    rotate_splat(args,model.extract(args),rotation_matrix)
