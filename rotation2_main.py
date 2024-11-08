import os
import math
import torch
import viser.transforms as vst

from e3nn import o3
from gaussian_renderer import GaussianModel
from gaussian_utils import GaussianTransformUtils
from argparse import ArgumentParser

def transform_shs(shs_feat, rotation_matrix):
    shs_feat = shs_feat.to("cuda")
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float().to('cuda')
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix.to('cuda') @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    rot_angles = [r.cpu() for r in rot_angles]
    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2]).to('cuda')
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2]).to('cuda')
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2]).to('cuda')

    shs_feat[:, 0:3] = D_1 @ shs_feat[:, 0:3]
    shs_feat[:, 3:8] = D_2 @ shs_feat[:, 3:8]
    shs_feat[:, 8:15] = D_3 @ shs_feat[:, 8:15]
    return shs_feat

def rotate_splat(input_ply, output_ply, rotation_matrix):
    with torch.no_grad():
        model = GaussianModel(sh_degree=3)
        model.load_ply(input_ply)

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
        model._features_rest = wigner_D_rotated_shs[:, 1:, :]
        model.save_ply(output_ply)

if __name__ == "__main__":
    parser = ArgumentParser(description="Rotate a Gaussian Splatting PLY file")
    parser.add_argument("--input", required=True, help="Input PLY file path")
    parser.add_argument("--x_angle", type=float, default=90, help="X rotation angle in degrees")
    parser.add_argument("--y_angle", type=float, default=180, help="Y rotation angle in degrees")
    parser.add_argument("--z_angle", type=float, default=0, help="Z rotation angle in degrees")
    args = parser.parse_args()

    input_path = args.input
    output_path = input_path.replace('.ply', '_rot2.ply')

    roll = math.radians(args.x_angle)
    pitch = math.radians(args.y_angle)
    yaw = math.radians(args.z_angle)

    so3 = vst.SO3.from_rpy_radians(roll, pitch, yaw)
    rotation_matrix = torch.tensor(so3.as_matrix(), dtype=torch.float, device="cuda")

    print(f"Rotating {input_path}")
    print(f"Rotation angles (degrees) - X: {args.x_angle}, Y: {args.y_angle}, Z: {args.z_angle}")
    rotate_splat(input_path, output_path, rotation_matrix)
    print(f"Saved rotated model to {output_path}")
