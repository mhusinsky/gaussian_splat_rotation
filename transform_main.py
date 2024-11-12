import os
import math
import torch
import viser.transforms as vst

from e3nn import o3
from gaussian_renderer import GaussianModel
from gaussian_utils import GaussianTransformUtils
from argparse import ArgumentParser

def filter_by_bounding_box(model, min_bounds, max_bounds):
    xyz = model._xyz
    # Check which points are within bounds for all dimensions
    mask = (xyz[:, 0] >= min_bounds[0]) & (xyz[:, 0] <= max_bounds[0]) & \
           (xyz[:, 1] >= min_bounds[1]) & (xyz[:, 1] <= max_bounds[1]) & \
           (xyz[:, 2] >= min_bounds[2]) & (xyz[:, 2] <= max_bounds[2])
    
    # Apply mask to all model attributes
    model._xyz = model._xyz[mask]
    model._rotation = model._rotation[mask]
    model._scaling = model._scaling[mask]
    model._opacity = model._opacity[mask]
    model._features_dc = model._features_dc[mask]
    model._features_rest = model._features_rest[mask]
    
    return model


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

def transform_splats(input_ply, output_ply, rotation_matrix, translation_offset):
    with torch.no_grad():
        model = GaussianModel(sh_degree=3)
        model.load_ply(input_ply)

        splat_xyz = model.get_xyz

        
        wigner_D_rotated_extra_shs = transform_shs(model.get_features[:, 1:, :].clone().cpu(), rotation_matrix.cpu())
        wigner_D_rotated_shs = model.get_features.clone().cpu()
        wigner_D_rotated_shs[:, 1:, :] = wigner_D_rotated_extra_shs.cpu()
        

        rotated_xyz = splat_xyz @ torch.tensor(so3.as_matrix().T, device=model.get_xyz.device, dtype=torch.float)
        
        translated_xyz = rotated_xyz + translation_offset.to(rotated_xyz.device) 
        
        rotated_rotations = torch.nn.functional.normalize(GaussianTransformUtils.quat_multiply(
            model.get_rotation,
            torch.tensor(so3.as_quaternion_xyzw()[[3, 0, 1, 2]], device=translated_xyz.device, dtype=torch.float),
        ))

        model._xyz = translated_xyz 
        model._rotation = rotated_rotations
        model._features_rest = wigner_D_rotated_shs[:, 1:, :]

        # Apply bounding box filtering
        model = filter_by_bounding_box(model, min_bounds, max_bounds)
        
        model.save_ply(output_ply)

if __name__ == "__main__":
    parser = ArgumentParser(description="Rotate and translate a Gaussian Splatting PLY file")
    parser.add_argument("--input", required=True, help="Input PLY file path")
    parser.add_argument("--euler", nargs=3, type=float, required=True, 
                        metavar=('X', 'Y', 'Z'),
                        help="Rotation angles in degrees (X Y Z)")
    parser.add_argument("--translate", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        metavar=('X', 'Y', 'Z'),
                        help="Translation offset (X Y Z)")
    parser.add_argument("--bbmin", nargs=3, type=float, required=True,
                        metavar=('X', 'Y', 'Z'),
                        help="Minimum bounds of filtering box (X Y Z)")
    parser.add_argument("--bbmax", nargs=3, type=float, required=True,
                        metavar=('X', 'Y', 'Z'),
                        help="Maximum bounds of filtering box (X Y Z)")
    args = parser.parse_args()

    input_path = args.input
    output_path = input_path.replace('.ply', '_rot_trans.ply')

    roll = math.radians(args.euler[0])
    pitch = math.radians(args.euler[1])
    yaw = math.radians(args.euler[2])

    so3 = vst.SO3.from_rpy_radians(roll, pitch, yaw)
    rotation_matrix = torch.tensor(so3.as_matrix(), dtype=torch.float, device="cuda")
    translation_offset = torch.tensor(args.translate, dtype=torch.float, device="cuda")
    min_bounds = torch.tensor(args.bbmin, dtype=torch.float, device="cuda")
    max_bounds = torch.tensor(args.bbmax, dtype=torch.float, device="cuda")

    print(f"Processing {input_path}")
    print(f"Rotation angles (degrees) - X: {args.euler[0]}, Y: {args.euler[1]}, Z: {args.euler[2]}")
    print(f"Translation offset - X: {args.translate[0]}, Y: {args.translate[1]}, Z: {args.translate[2]}")
    print(f"Bounding box min - X: {args.bbmin[0]}, Y: {args.bbmin[1]}, Z: {args.bbmin[2]}")
    print(f"Bounding box max - X: {args.bbmax[0]}, Y: {args.bbmax[1]}, Z: {args.bbmax[2]}")

    transform_splats(input_path, output_path, rotation_matrix, translation_offset)
    print(f"Saved transformed model to {output_path}")
