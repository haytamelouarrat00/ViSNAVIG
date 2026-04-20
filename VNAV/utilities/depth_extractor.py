import torch
import numpy as np
import cv2

try:
    from moge.model.v2 import MoGeModel
except ImportError as e:
    import warnings
    warnings.warn(f"MoGe is not installed. Please install it using: pip install git+https://github.com/microsoft/MoGe.git\nActual Error: {e}")
    MoGeModel = None

class MoGe2DepthExtractor:
    """
    Wrapper for Microsoft's MoGe-2 depth estimation model.
    Optimized for high-speed inference in a visual servoing loop.
    """
    def __init__(self, variant="vits", device="cuda", half_precision=True):
        """
        Initializes the MoGe-2 model.
        
        Args:
            variant (str): "vits", "vitb", or "vitl". "vits" is the fastest (~35M params).
            device (str): Compute device ("cuda" or "cpu").
            half_precision (bool): Whether to use FP16 for faster inference and lower memory.
        """
        if MoGeModel is None:
            raise RuntimeError("MoGe is not installed.")
            
        model_name = f"Ruicheng/moge-2-{variant}-normal"
        print(f"Loading MoGe-2 Depth Extractor ({model_name})...")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half_precision = half_precision
        
        # Load the model from huggingface
        self.model = MoGeModel.from_pretrained(model_name).to(self.device)
        
        if self.half_precision and self.device.type == "cuda":
            self.model = self.model.half()
            
        self.model.eval()
        print("MoGe-2 loaded successfully.")

    @torch.no_grad()
    def get_depth(self, image: np.ndarray, downscale_factor: float = 0.5) -> np.ndarray:
        """
        Infers the metric depth map from an RGB image.
        
        Args:
            image (np.ndarray): HxWx3 RGB image (uint8 or float).
            downscale_factor (float): Factor to resize the image before inference for speed.
                                      The output depth map is resized back to the original size.
                                      
        Returns:
            np.ndarray: HxW depth map in metric scale (meters).
        """
        orig_h, orig_w = image.shape[:2]
        
        # Downscale for speed
        if downscale_factor < 1.0:
            new_w = int(orig_w * downscale_factor)
            new_h = int(orig_h * downscale_factor)
            img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = image

        # Ensure image is RGB and float32/float16 in [0, 1]
        if img_resized.dtype == np.uint8:
            img_tensor = img_resized.astype(np.float32) / 255.0
        else:
            img_tensor = img_resized.astype(np.float32)

        tensor_dtype = torch.float16 if (self.half_precision and self.device.type == "cuda") else torch.float32
        
        # (3, H, W)
        input_tensor = torch.tensor(img_tensor, dtype=tensor_dtype, device=self.device).permute(2, 0, 1)

        # Run inference
        # infer() returns a dict with 'depth', 'points', 'normal', 'mask', 'intrinsics'
        output = self.model.infer(input_tensor)
        
        depth_map = output['depth'].detach().cpu().numpy().astype(np.float32)
        
        # Upscale depth map back to original resolution so pixel coordinates match
        if downscale_factor < 1.0:
            depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
        return depth_map

if __name__ == "__main__":
    # Simple test script
    import time
    extractor = MoGe2DepthExtractor(variant="vits")
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    _ = extractor.get_depth(dummy_img, downscale_factor=0.5)
    
    # Timing
    t0 = time.time()
    depth = extractor.get_depth(dummy_img, downscale_factor=0.5)
    t1 = time.time()
    
    print(f"Depth shape: {depth.shape}, Min: {depth.min():.2f}, Max: {depth.max():.2f}")
    print(f"Inference time (with 0.5x downscale): {(t1 - t0) * 1000:.1f} ms")
