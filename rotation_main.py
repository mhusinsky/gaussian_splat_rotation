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


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def transform_shs(shs_feat, rotation_matrix):

    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]],dtype=torch.float32) 
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rotation_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    
    D1 = o3.wigner_D(1, rotation_angles[0], - rotation_angles[1], rotation_angles[2])
    D2 = o3.wigner_D(2, rotation_angles[0], - rotation_angles[1], rotation_angles[2])
    D3 = o3.wigner_D(3, rotation_angles[0], - rotation_angles[1], rotation_angles[2])

    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
            D1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
            D2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
            D3,
            three_degree_shs,
            "... i j, ... j -> ... i",
        )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

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
        wigner_D_rotated_shs[:, 1:, :] = wigner_D_rotated_extra_shs

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