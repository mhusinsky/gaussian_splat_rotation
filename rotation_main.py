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

def eval_sh_for_each_degree(deg, sh, dirs):
    result_list = []

    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    result_list.append(result)
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result_list.append(-
                           C1 * y * sh[..., 1] +
                           C1 * z * sh[..., 2] -
                           C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result_list.append(+
                               C2[0] * xy * sh[..., 4] +
                               C2[1] * yz * sh[..., 5] +
                               C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                               C2[3] * xz * sh[..., 7] +
                               C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result_list.append(+
                                   C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                                   C3[1] * xy * z * sh[..., 10] +
                                   C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                                   C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                                   C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                                   C3[5] * z * (xx - yy) * sh[..., 14] +
                                   C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result_list.append(+ C4[0] * xy * (xx - yy) * sh[..., 16] +
                                       C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                                       C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                                       C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                                       C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                                       C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                                       C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                                       C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                                       C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result_list

def transform_shs(shs_feat, rotation_matrix):

    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]],dtype=torch.float32) 
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
            D_3,
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