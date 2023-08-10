import math
from typing import Tuple
import nvdiffrast.torch as dr
import numpy as np
import torch
from torch import Tensor
from tree_segmentation.extension import Mesh, utils
from tree_segmentation.extension import ops_3d

dr.set_log_level(2)


@torch.no_grad()
def render_mesh(
        glctx,
        mesh: Mesh,
        Tw2v: Tensor = None,
        Tw2c: Tensor = None,
        fovy=math.radians(60),
        image_size=1024,
        light_location=(0, 2., 0.),
        background=1.,
        use_face_normal=False,
        default_kads=(0.2, 0.5, 0.1),
) -> Tuple[Tensor, Tensor]:
    device = mesh.v_pos.device
    camera_pos = Tw2v.inverse()[..., :3, 3]
    if camera_pos.ndim == 1:
        camera_pos = camera_pos[None, None, None]
    elif camera_pos.ndim == 2:
        camera_pos = camera_pos[:, None, None, :]
    else:
        raise ValueError
    view_direction = ops_3d.normalize(camera_pos)
    lights = ops_3d.PointLight(
        ambient_color=utils.n_tuple(0.5, 3),
        diffuse_color=utils.n_tuple(1., 3),
        specular_color=utils.n_tuple(0.3, 3),
        device=device)
    lights.location = camera_pos
    if Tw2c is None:
        if Tw2v is not None:
            Tv2c = ops_3d.perspective(fovy=fovy, size=(image_size, image_size), device=device)
            Tw2c = Tv2c @ Tw2v
        else:
            Tw2c = None
    v_pos = mesh.v_pos.float() if Tw2c is None else ops_3d.xfm(mesh.v_pos.float(), Tw2c)
    v_pos = v_pos[None] if v_pos.ndim == 2 else v_pos
    rast, _ = dr.rasterize(glctx, v_pos, mesh.f_pos.int(), (image_size, image_size))
    faces = rast[..., -1].int()
    if getattr(mesh, 'v_clr', None) is not None:
        images, _ = dr.interpolate(mesh.v_clr[None, :, :].contiguous().float(), rast, mesh.f_pos.int())
    else:
        perturbed_nrm = None
        if mesh.f_tex is not None:
            uv, uv_da = dr.interpolate(mesh.v_tex[None], rast, mesh.f_tex.int())
            if mesh.f_mat is not None:
                # f_mat = torch.where(faces == 0, torch.zeros_like(faces), mesh.f_mat[faces - 1])
                f_mat = mesh.f_mat[faces - 1]
            else:
                f_mat = None
            ka = mesh.material['ka'].sample(uv, f_mat=f_mat) if 'ka' in mesh.material else 0
            kd = mesh.material['kd'].sample(uv, f_mat=f_mat) if 'kd' in mesh.material else 0
            ks = mesh.material['ks'].sample(uv, f_mat=f_mat) if 'ks' in mesh.material else 0
            if 'normal' in mesh.material:
                perturbed_nrm = mesh.material['normal'].sample(uv, f_mat=f_mat)
        else:
            ka = v_pos.new_full((3,), default_kads[0])
            kd = v_pos.new_full((3,), default_kads[1])
            ks = v_pos.new_full((3,), default_kads[2])
        if not use_face_normal and mesh.f_nrm is not None and mesh.f_tng is not None:
            nrm = ops_3d.compute_shading_normal(mesh, camera_pos, rast, perturbed_nrm)
        else:
            nrm = ops_3d.compute_shading_normal_face(mesh, camera_pos, rast, None)
        points, _ = dr.interpolate(mesh.v_pos[None].float(), rast, mesh.f_pos.int())
        images = ops_3d.Blinn_Phong(nrm, lights(points), view_direction, (ka, kd, ks))
    images = dr.antialias(images, rast, v_pos, mesh.f_pos.int()).clamp(0, 1)
    images = torch.where(rast[..., -1:] > 0, images, torch.full_like(images, background))
    if Tw2c.ndim == 2:
        return images[0, :, :, :3], faces[0]
    else:
        return images, faces


def choose_best_views(
    glctx,
    mesh: Mesh,
    Tw2c: Tensor,
    N=100,
    image_size=1024,
    num_split=10,
):
    # get faces
    device = Tw2c.device
    M = Tw2c.size(0)
    assert M >= N
    tri_ids = []
    for i in range(0, M, num_split):
        v_pos = ops_3d.xfm(mesh.v_pos, Tw2c[i:i + num_split])
        rast, _ = dr.rasterize(glctx, v_pos, mesh.f_pos.int(), (image_size, image_size))
        tri_ids.append(rast[..., -1].int())
    tri_ids = torch.cat(tri_ids, dim=0)
    face_indices = [torch.unique(tri_ids[i]) for i in range(M)]
    del tri_ids
    #  init
    cnt = torch.zeros(mesh.f_pos.shape[0] + 1, device=device)
    choosed = []
    for i in np.random.choice(M, N, replace=False):
        cnt[face_indices[i]] += 1
        choosed.append(i)
    # score = torch.std(cnt)
    score = torch.mean(cnt)
    best_choosed = choosed.copy()
    # choose best
    best_score = score
    for i in range(M * 10):
        # if i % 100 == 0:
        #     print(f'step: {i}, score: {score.item():.4f}, best: {best_score.item():.4f}')
        x = np.random.choice(choosed, 1)[0]
        y = np.random.choice(M, 1)[0]
        cnt[face_indices[x]] -= 1
        cnt[face_indices[y]] += 1

        # new_score = torch.std(cnt)
        new_score = torch.mean(cnt)
        if best_score < new_score:
            best_score = new_score
            best_choosed = choosed.copy()
        if new_score < score or np.random.rand() < 0:
            score = new_score
            choosed.pop(choosed.index(x))
            choosed.append(y)
        else:
            cnt[face_indices[x]] += 1
            cnt[face_indices[y]] -= 1
    print('Best the average number of viewed faces:', best_score)
    return torch.tensor(best_choosed, device=device, dtype=torch.long)


def random_camera_position(mesh: Mesh, bbox: Tensor = None, N=100, min_dist=0.1) -> Tensor:
    import trimesh
    device = mesh.v_pos.device
    mesh = mesh.to_trimesh()
    if bbox is None:
        AABB = mesh.AABB.cpu().numpy()
        box_min = AABB[0].numpy() - min_dist
        box_max = AABB[1].numpy() - min_dist
    else:
        bbox = bbox.cpu().numpy()
        box_min = bbox[0]
        box_max = bbox[1]
    points = np.random.rand(N, 3) * (box_max - box_min) + box_min
    _, distance, _, = trimesh.proximity.closest_point(mesh, points)
    points = points[distance > min_dist]
    eye = torch.from_numpy(points).float().to(device)
    return eye