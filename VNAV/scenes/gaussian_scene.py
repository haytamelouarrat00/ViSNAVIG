import torch
import numpy as np
from plyfile import PlyData
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .base_scene import BaseScene

class GaussianScene(BaseScene):
    """
    Scene handler for 3D Gaussian Splatting using the diff-gaussian-rasterization library.
    More memory-efficient for large reconstructions.
    """

    def __init__(self, width: int = 640, height: int = 480, device: str = "cuda"):
        self.width = width
        self.height = height
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Gaussian parameters
        self.means = None
        self.quats = None
        self.scales = None
        self.opacities = None
        self.shs = None

    def load(self, path: str) -> None:
        """Loads a Gaussian Splatting model from a .ply file."""
        print(f"Loading 3DGS model (diff-gaussian) from {path}...")
        plydata = PlyData.read(path)
        v = plydata['vertex']
        
        self.means = torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1), device=self.device, dtype=torch.float32).contiguous()
        self.opacities = torch.sigmoid(torch.tensor(v['opacity'], device=self.device, dtype=torch.float32)[:, None]).contiguous()
        
        scale_names = [p.name for p in v.properties if p.name.startswith('scale_')]
        self.scales = torch.exp(torch.tensor(np.stack([v[s] for s in scale_names], axis=-1), device=self.device, dtype=torch.float32)).contiguous()
        
        rot_names = ['rot_0', 'rot_1', 'rot_2', 'rot_3'] 
        self.quats = torch.tensor(np.stack([v[r] for r in rot_names], axis=-1), device=self.device, dtype=torch.float32)
        self.quats = torch.nn.functional.normalize(self.quats, p=2, dim=-1).contiguous()
        
        # SHs: diff-gaussian-rasterization expects (N, 1, 3) for degree 0
        self.shs = torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1), device=self.device, dtype=torch.float32)[:, None, :].contiguous()
        
        print(f"Loaded {len(self.means)} Gaussians.")

    def _run_rasterizer(self, pose: np.ndarray, intrinsics: np.ndarray):
        if self.means is None:
            raise RuntimeError("Scene not loaded. Call load() first.")
            
        # 1. Compute Matrices (World-to-Camera)
        T_wc = torch.tensor(pose, device=self.device, dtype=torch.float32)
        
        # Projection Matrix (OpenGL style)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        W, H = self.width, self.height
        
        fov_x = 2 * np.arctan(W / (2 * fx))
        fov_y = 2 * np.arctan(H / (2 * fy))
        
        # Construct full projection matrix (OpenGL style, matching 3DGS)
        tan_fovx = np.tan(fov_x / 2)
        tan_fovy = np.tan(fov_y / 2)
        z_near, z_far = 0.01, 100.0
        
        P = torch.zeros((4, 4), device=self.device)
        P[0, 0] = 1.0 / tan_fovx
        P[1, 1] = 1.0 / tan_fovy
        P[2, 2] = z_far / (z_far - z_near)
        P[2, 3] = -(z_far * z_near) / (z_far - z_near)
        P[3, 2] = 1.0
        
        # 3DGS expects transposed matrices
        viewmatrix = T_wc.transpose(0, 1)
        projmatrix = P.transpose(0, 1)
        full_proj = viewmatrix @ projmatrix

        # 2. Rasterization Settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(H),
            image_width=int(W),
            tanfovx=tan_fovx,
            tanfovy=tan_fovy,
            bg=torch.zeros(3, device=self.device),
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_proj,
            sh_degree=0,
            campos=torch.inverse(T_wc)[:3, 3],
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # 3. Rasterize
        raster_out = rasterizer(
            means3D=self.means,
            means2D=torch.zeros_like(self.means, device=self.device, requires_grad=True),
            shs=self.shs,
            colors_precomp=None,
            opacities=self.opacities,
            scales=self.scales,
            rotations=self.quats,
            cov3D_precomp=None
        )
        return raster_out

    def render(self, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Renders the splats using diff-gaussian-rasterization."""
        raster_out = self._run_rasterizer(pose, intrinsics)
        rendered_image = raster_out[0]

        # Clamp and convert to numpy
        img = (torch.clamp(rendered_image, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        return img

    def render_depth(self, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Renders the metric depth map (metres) of the splats.

        The 2024+ Inria diff-gaussian-rasterizer emits alpha-composited *inverse*
        depth at ``raster_out[2]`` (see ``forward.cu:395`` — ``expected_invdepth``).
        We invert it to per-pixel depth and mask pixels with no Gaussian coverage
        (invdepth ≈ 0) as 0, which downstream mask-based consumers treat as invalid.
        """
        raster_out = self._run_rasterizer(pose, intrinsics)

        if len(raster_out) > 2 and raster_out[2].numel() > 0:
            invdepth = raster_out[2].squeeze(0).detach().cpu().numpy().astype(np.float32)
            depth = np.zeros_like(invdepth)
            valid = invdepth > 1e-6
            depth[valid] = 1.0 / invdepth[valid]
            return depth

        import warnings
        warnings.warn("This version of diff-gaussian-rasterization does not return depth maps natively. Returning empty depth.")
        return np.zeros((self.height, self.width), dtype=np.float32)
