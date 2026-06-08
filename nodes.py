import sys
import os
import torch
import numpy as np
import trimesh as tm
from PIL import Image
import folder_paths

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meshflow.pipelines import MeshFlowPipeline
from meshflow.utils.mesh import Mesh

_PIPELINE_CACHE = {}

def get_pipeline(model_path, device, dtype, compile_models, num_verts, image_size):
    cache_key = (model_path, device, dtype, compile_models, num_verts, image_size)
    if cache_key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE.clear()
        pipeline = MeshFlowPipeline.from_pretrained(
            model_path=model_path,
            device=device,
            dtype=dtype,
            compile_models=compile_models,
            num_verts=num_verts,
        )
        if pipeline._visual_encoder_cfg is not None:
            pipeline._visual_encoder_cfg["image_size"] = image_size
            from meshflow.models.condition_encoder import make_empty_visual_embeds, resolve_visual_embed_dim
            embed_dim = resolve_visual_embed_dim(
                pipeline._visual_encoder_cfg,
                fallback=1024,
            )
            pipeline._empty_visual_embeds = make_empty_visual_embeds(image_size, embed_dim)

            hub_dir = pipeline._visual_encoder_cfg.get("hub_dir", "")
            if not os.path.exists(hub_dir):
                home_hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "facebookresearch_dinov3_main")
                pipeline._visual_encoder_cfg["hub_dir"] = home_hub_dir
            
            local_weights = os.path.join(folder_paths.models_dir, "facebook", "dinov3-vitl16-pretrain-lvd1689m", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
            if os.path.exists(local_weights):
                pipeline._visual_encoder_cfg["hub_weights"] = local_weights
            else:
                hub_weights = pipeline._visual_encoder_cfg.get("hub_weights")
                if hub_weights and not os.path.exists(hub_weights):
                    pipeline._visual_encoder_cfg["hub_weights"] = None
        _PIPELINE_CACHE[cache_key] = pipeline
    return _PIPELINE_CACHE[cache_key]

class MeshFlowRemesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "The input 3D model to be remeshed."}),
                "model_name": (["meshflow", "meshflow_w_num_verts_control"], {"default": "meshflow", "tooltip": "Select the MeshFlow model to use. 'meshflow' is the standard model. 'meshflow_w_num_verts_control' allows dynamic control of the generated mesh resolution."}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of diffusion sampling steps. Higher values can increase detail but take longer."}),
                "sampler": (["euler", "midpoint", "heun", "rk4"], {"default": "heun", "tooltip": "ODE solver for flow-matching sampling. Higher-order solvers (heun, rk4) produce more accurate trajectories; heun uses 2 model evals per step, rk4 uses 4."}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Classifier-Free Guidance (CFG) scale for visual conditioning. Only effective when reference_image is connected."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for sampling latents."}),
                "base_num_verts": ([1024, 2048, 4096, 8192, 16384], {"default": 4096, "tooltip": "The base resolution/point count of the loaded model checkpoint (sequence length)."}),
                "points": ("INT", {"default": 4096, "min": 1024, "max": 16384, "step": 256, "tooltip": "Target resolution (points/vertices) of the generated output mesh."}),
                "image_size": ([512, 1024, 2048], {"default": 512, "tooltip": "Resolution to resize and center crop the reference image to before processing."}),
                "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Computation device to run the model on (cuda or cpu)."}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16", "tooltip": "Precision model dtype (fp16, bf16, or fp32)."}),
                "compile": ("BOOLEAN", {"default": False, "tooltip": "Whether to use torch.compile on CUDA for faster inference."}),
                "use_rmbg": ("BOOLEAN", {"default": False, "tooltip": "Enable automatic background removal and foreground cropping for the reference image."}),
                "fill_holes": ("BOOLEAN", {"default": False, "tooltip": "Automatically closes and triangulates boundary loops in the final mesh topology."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional reference image for image-conditioned generation."}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "remesh"
    CATEGORY = "MeshFlow"

    def remesh(self, trimesh, model_name, steps, sampler, guidance_scale, seed, base_num_verts, points, image_size, device, dtype, compile, use_rmbg, fill_holes, reference_image=None):
        model_path = os.path.join(folder_paths.models_dir, "facebook", "meshflow", model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"MeshFlow model path not found at {model_path}. Please download and place the config.yaml and model.pth in that directory.")

        pipeline = get_pipeline(model_path, device, dtype, compile, base_num_verts, image_size)
        pipeline.use_rmbg = use_rmbg

        if reference_image is not None:
            hub_dir = pipeline._visual_encoder_cfg.get("hub_dir")
            if hub_dir and not os.path.exists(hub_dir):
                os.makedirs(os.path.dirname(hub_dir), exist_ok=True)
                import subprocess
                subprocess.run(["git", "clone", "https://github.com/facebookresearch/dinov3.git", hub_dir], check=True)

        if isinstance(trimesh, tm.Scene):
            parts = []
            for _, node in trimesh.graph.to_flattened().items():
                name = node["geometry"]
                if name in trimesh.geometry and isinstance(trimesh.geometry[name], tm.Trimesh):
                    parts.append(trimesh.geometry[name].copy().apply_transform(node["transform"]))
            trimesh = tm.util.concatenate(parts)

        mesh_in = Mesh(
            verts=torch.tensor(trimesh.vertices, dtype=torch.float32),
            faces=torch.tensor(trimesh.faces, dtype=torch.int64),
            device=torch.device(device)
        )
        mesh_in.preprocess()
        mesh_in.normalize(normalize_by="bsphere", size=2.0)

        pil_image = None
        if reference_image is not None:
            img_tensor = reference_image[0]
            if img_tensor.shape[-1] == 4:
                img_tensor = img_tensor[..., :3]
            img_np = (img_tensor * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_image = Image.fromarray(img_np)

        if pil_image is None:
            guidance_scale = 1.0

        out_mesh, latents = pipeline.run(
            mesh=mesh_in,
            image=pil_image,
            steps=steps,
            sampler=sampler,
            guidance_scale=guidance_scale,
            seed=seed,
            num_verts=points,
            return_latent=True
        )

        if fill_holes:
            out_mesh = pipeline.decode_latent(latents, fill_holes=True)

        return (out_mesh.to_trimesh(),)

NODE_CLASS_MAPPINGS = {
    "MeshFlowRemesh": MeshFlowRemesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshFlowRemesh": "MeshFlow Remesh"
}
