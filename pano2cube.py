import torch
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from scipy.ndimage import map_coordinates

to_pil_image = transforms.ToPILImage()

# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube(nn.Module):
    def __init__(self, equ_h: int, equ_w: int, cube_length: int, FOV: float = 90.0, device = 'cuda'):
        """
        Initialize the equirectangular to cubemap converter.
        Args:
            equ_h, equ_w: Height and width of the equirectangular panorama.
            cube_length: Side length (in pixels) of each output cubemap face.
            FOV: Field of view for each cubemap face (in degrees, default 90).
        """
        super().__init__()
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.cube_length = cube_length
        self.FOV = FOV
        self.device = device
        # Pre-compute the normalized sampling grid for each of the 6 cube faces.
        # We will store a grid of shape (6, face_w, face_w, 2) with values in [-1,1].
        # (We don't assign device here; we'll move it to the input's device in forward.)
        self.face_grids = self._create_face_grids(cube_length, FOV)
        
    def _create_face_grids(self, face_w: int, FOV: float):
        """Create the normalized sampling grids (lon, lat) for all six faces."""
        # Compute tangent of half FOV (for mapping to plane)
        t = float(torch.tan(torch.tensor(FOV * 0.5 * torch.pi/180.0)))  # FOV in radians half-angle
        # Create normalized coordinate vectors for face pixels (range -1 to 1).
        u = torch.linspace(-1.0, 1.0, face_w)  # horizontal coordinate (left=-1, right=+1)
        v = torch.linspace(-1.0, 1.0, face_w)  # vertical coordinate (top=-1, bottom=1)
        # Note: v is reversed so that +1 corresponds to top (north pole) and -1 to bottom (south).
        # Meshgrid to get coordinate matrix for face
        v_map, u_map = torch.meshgrid(v, u, indexing='ij')  # shape (face_w, face_w)
        # Prepare a container for 6 face grids
        grids = torch.empty((6, face_w, face_w, 2), dtype=torch.float32)
        # For each face, compute direction vectors and then spherical coords
        # 1. Front face (center looking toward +Z)
        x = (u_map * t)
        y = (v_map * t)
        z = torch.ones_like(u_map)
        grids[0] = self._xyz_to_lonlat_grid(x, y, z)
        # 2. Right face (center looking toward +X)
        x = torch.ones_like(u_map)
        y = (v_map * t)
        z = -(u_map * t)
        grids[1] = self._xyz_to_lonlat_grid(x, y, z)
        # 3. Back face (center looking toward -Z)
        x = -(u_map * t)
        y = (v_map * t)
        z = -torch.ones_like(u_map)
        grids[2] = self._xyz_to_lonlat_grid(x, y, z)
        # 4. Left face (center looking toward -X)
        x = -torch.ones_like(u_map)
        y = (v_map * t)
        z = (u_map * t)
        grids[3] = self._xyz_to_lonlat_grid(x, y, z)
        # 5. Up face (center looking toward -Y)
        x = (u_map * t)
        y = -torch.ones_like(u_map)
        z = (v_map * t)
        grids[4] = self._xyz_to_lonlat_grid(x, y, z)
        # 6. Down face (center looking toward +Y)
        x = (u_map * t)
        y = torch.ones_like(u_map)
        z = -(v_map * t)
        grids[5] = self._xyz_to_lonlat_grid(x, y, z)
        return grids

    def _xyz_to_lonlat_grid(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """
        Convert 3D direction vectors (x, y, z) on a unit sphere to a normalized grid of (lon, lat) in [-1,1].
        """
        # Normalize the vectors to unit length (to lie on the unit sphere)
        inv_norm = torch.rsqrt(x * x + y * y + z * z)  # 1/sqrt(x^2+y^2+z^2)
        x_norm = x * inv_norm
        y_norm = y * inv_norm
        z_norm = z * inv_norm
        # Longitude (theta): arctan2(X, Z) – horizontal angle around Y-axis, range [-pi, pi]
        lon = torch.atan2(x_norm, z_norm)  # shape (face_w, face_w)
        # Latitude (phi): asin(Y) – vertical angle from equator, range [-pi/2, pi/2]
        lat = torch.asin(y_norm)          # shape (face_w, face_w)
        # Normalize to [-1,1] range for grid_sample:
        # lon in [-pi, pi] -> divide by pi
        # lat in [-pi/2, pi/2] -> divide by (pi/2)
        lon_norm = lon / torch.pi        # normalized longitude
        lat_norm = lat / (0.5 * torch.pi)  # normalized latitude
        # Note: We do *not* invert lat here because v_map was defined with top=+1, bottom=-1.
        # Combine into grid: shape (face_w, face_w, 2) as (x_coord, y_coord)
        grid = torch.stack([lon_norm, lat_norm], dim=-1)
        return grid

    def forward(self, img_tensor: torch.Tensor, depth_tensor: torch.Tensor = None, mode: str = 'bilinear'):
        """
        Perform the equirectangular to cubemap conversion.
        Args:
            img_tensor: Input panorama image tensor of shape (B, C, H, W) in float32.
            depth_tensor: (Optional) Depth map tensor of shape (B, 1, H, W) corresponding to the panorama.
            mode: Sampling mode for grid_sample ('bilinear' or 'nearest').
        Returns:
            If depth_tensor is not provided: returns cubemap image tensor of shape (B, 6, face_w, face_w, C).
            If depth_tensor is provided: returns a tuple (cubemap_image, cubemap_depth).
        """
        # Ensure input is float32
        if img_tensor.dtype != torch.float32:
            img_tensor = img_tensor.to(torch.float32)
        B, C, H, W = img_tensor.shape
        assert H == self.equ_h and W == self.equ_w, "Input image dimensions do not match the expected size."
        # Move precomputed grids to the input device, if not already
        grids = self.face_grids.to(self.device)
        # Sample the image for each face using grid_sample
        # We iterate over faces to avoid duplicating the input (keeping memory usage low).
        img_faces = []
        for face_idx in range(6):
            face_grid = grids[face_idx:face_idx+1].repeat(B, 1, 1, 1)  # shape (1, face_w, face_w, 2)
            face_pixels = F.grid_sample(img_tensor, face_grid, mode=mode, align_corners=True)
            # face_pixels: shape (B, C, face_w, face_w)
            img_faces.append(face_pixels)
        cubemap_img = torch.stack(img_faces, dim=1)  # (B, 6, C, face_w, face_w)
        if depth_tensor is not None:
            # Process depth map similarly
            if depth_tensor.dtype != torch.float32:
                depth_tensor = depth_tensor.to(torch.float32)
            dH, dW, dC = depth_tensor.shape
            assert dH == H and dW == W, "Depth map size must match image size."
            depth_NCHW = depth_tensor.unsqueeze(0)  # (1, 1, H, W)
            depth_faces = []
            for face_idx in range(6):
                face_grid = grids[face_idx:face_idx+1]  # (1, face_w, face_w, 2)
                # For depth, you might choose 'nearest' to avoid interpolating depth values, but we use the given mode.
                face_depth = F.grid_sample(depth_NCHW, face_grid, mode=mode, align_corners=True)
                depth_face = face_depth.squeeze(0)  # (1, face_w, face_w)
                depth_faces.append(depth_face)
            cubemap_depth = torch.stack(depth_faces, dim=0)  # (6, 1, face_w, face_w)
            return cubemap_img, cubemap_depth
        else:
            return cubemap_img

class Cube2Equirec(nn.Module):
    def __init__(self, cube_length, equ_h, device='cuda'):
        super().__init__()
        self.device = device

        self.cube_length = cube_length
        self.equ_h = equ_h
        equ_w = equ_h * 2
        self.equ_w = equ_w
        theta = (np.arange(equ_w) / (equ_w-1) - 0.5) * 2 *np.pi
        phi = (np.arange(equ_h) / (equ_h-1) - 0.5) * np.pi
        
        theta, phi = np.meshgrid(theta, phi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(phi)
        z = np.cos(theta) * np.cos(phi)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

        planes = np.asarray([
                    [0, 0, 1, -1], # z =  1
                    [1, 0, 0, -1], # x =  1
                    [0, 0, 1,  1], # z = -1
                    [1, 0, 0,  1], # x = -1
                    [0, 1, 0,  1],  # y = -1
                    [0, 1, 0, -1], # y =  1
                ])
        r_lst = np.array([
                [0, 0, 0], # z =  1
                [0, -0.5, 0], # x =  1
                [0, 1, 0], # z = -1
                [0, 0.5, 0], # x = -1
                [-0.5, 0, 0], # y = -1
                [0.5, 0, 0] # y =  1
            ]) * np.pi
        
        f = cube_length / 2.0
        self.K = np.array([
                [f, 0, (cube_length-1)/2.0],
                [0, f, (cube_length-1)/2.0],
                [0, 0, 1]
            ])
        self.R_lst = [cv2.Rodrigues(x)[0] for x in r_lst]

        extrinsics = []
        for R in self.R_lst:
            E_ext = torch.eye(4, dtype=torch.float32)
            E_ext[:3, :3] = torch.tensor(R)
            E_ext = torch.round(E_ext * 1e8) / 1e8
            extrinsics.append(E_ext)
        self.extrinsics = torch.stack(extrinsics, dim=0).to(self.device)

        masks, XYs = self._intersection(xyz, planes)

        for i in range(6):
            self.register_buffer('mask_%d'%i, masks[i])
            self.register_buffer('XY_%d'%i, XYs[i])
    
    def forward(self, x, mode='bilinear'):
        assert mode in ['nearest', 'bilinear']
        assert x.shape[0] % 6 == 0
        equ_count = x.shape[0] // 6
        equi = torch.zeros(equ_count, x.shape[1], self.equ_h, self.equ_w, device=self.device)
        for i in range(6):
            now = x[i::6, ...]
            mask = getattr(self, 'mask_%d'%i).to(self.device)
            mask = mask[None, ...].repeat(equ_count, x.shape[1], 1, 1)

            XY = (getattr(self, 'XY_%d'%i)[None, None, :, :].repeat(equ_count, 1, 1, 1) / (self.cube_length-1) - 0.5).to(self.device) * 2
            sample = F.grid_sample(now, XY, mode=mode, align_corners=True)[..., 0, :]
            equi[mask] = sample.view(-1)

        return equi

    def _intersection(self, xyz, planes):
        abc = planes[:, :-1]
        
        depth = -planes[:, 3][None, None, ...] / np.dot(xyz, abc.T)
        depth[depth < 0] = np.inf
        arg = np.argmin(depth, axis=-1)
        depth = np.min(depth, axis=-1)


        pts = depth[..., None] * xyz
        
        mask_lst = []
        mapping_XY = []
        for i in range(6):
            mask = arg == i
            mask = np.tile(mask[..., None], [1, 1, 3])

            XY = np.dot(np.dot(pts[mask].reshape([-1, 3]), self.R_lst[i].T), self.K.T)
            XY = np.clip(XY[..., :2].copy() / XY[..., 2:], 0, self.cube_length-1)
            mask_lst.append(mask[..., 0])
            mapping_XY.append(XY)
        mask_lst = [torch.BoolTensor(x) for x in mask_lst]
        mapping_XY = [torch.FloatTensor(x) for x in mapping_XY]

        return mask_lst, mapping_XY



if __name__ == '__main__':
    path = '/home/qiwei/program/Omni-Scene/vigor1.jpg'
    height = 512
    width = 1024
    cube_length = 256
    E2C = Equirec2Cube(equ_h=height, equ_w=width, cube_length=cube_length)
    C2E = Cube2Equirec(cube_length=256, equ_h=height)
    # load rgb
    rgb = Image.open(path).convert("RGB")

    # 定义转换流程：先 Resize，再转为 tensor
    transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=Image.BICUBIC),  # 指定高度为 512，宽度为 1024
        transforms.ToTensor()            # 转为 [C, H, W] 格式，并归一化到 [0,1]
    ])

    rgb = transform(rgb).to('cuda')

    # E2C
    cube_rgb = E2C(rgb[:1].unsqueeze(0))
    for b in range(len(cube_rgb[0])):
        cube = to_pil_image(cube_rgb[0][b])
        cube.save(f'cube_{b}.png')
    
    # C2E
    panorama_rgb = C2E(cube_rgb)
    panorama = to_pil_image(panorama_rgb[0])
    panorama.save('panorama.png')

    print(cube_rgb.shape)




# Based on https://github.com/sunset1995/py360convert
class Equirec2CubeUniFuse:
    def __init__(self, equ_h, equ_w, face_w):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w

        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]

    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [F R B L U D] format.
        '''
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front face (z = 0.5)
        self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0 * self.face_w:1 * self.face_w, 2] = 0.5

        # Right face (x = 0.5)
        self.xyz[:, 1 * self.face_w:2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5

        # Back face (z = -0.5)
        self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w:3 * self.face_w, 2] = -0.5

        # Left face (x = -0.5)
        self.xyz[:, 3 * self.face_w:4 * self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5

        # Up face (y = 0.5)
        self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w:5 * self.face_w, 1] = 0.5

        # Down face (y = -0.5)
        self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5 * self.face_w:6 * self.face_w, 1] = -0.5

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        # pad_l = e_img[:, [0]]
        # pad_r = e_img[:, [-1]]
        # e_img = np.concatenate([e_img, pad_l, pad_r], 1)

        return map_coordinates(e_img, [self.coor_y, self.coor_x],
                               order=order, mode='wrap')[..., 0]

    def run(self, equ_img, equ_dep=None):

        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)

        if equ_dep is not None:
            cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
                                 for i in range(equ_dep.shape[2])], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        if equ_dep is not None:
            return cube_img, cube_dep
        else:
            return cube_img