# ComfyUI-MeshFlow

A ComfyUI custom node wrapper for Meta's **MeshFlow** pipeline, providing state-of-the-art artist-like mesh remeshing directly within ComfyUI. 

This node allows you to remesh 3D models to a clean, well-distributed vertex structure, optionally conditioned on a reference image. It interfaces seamlessly with the `TRIMESH` type used by standard 3D custom nodes (such as ComfyUI-PhantyForge).

---

## Tribute & Credits

This project is a wrapper around the research and official implementation of:

**MeshFlow: A Diffusion Model for Artist-Like Mesh Generation**  
*Facebook Research (Meta)*  
[Paper](https://arxiv.org/abs/2411.08271) | [Official Codebase](https://github.com/facebookresearch/meshflow)

We give our deepest thanks and tribute to the original creators and authors at Meta for open-sourcing this incredible model and making high-quality, artist-like mesh diffusion generation accessible to the community.

---

## Features

- **Mesh Remeshing**: Transforms irregular input geometry into a clean, uniform vertex layout.
- **Vertex Control**: Choose between the standard `meshflow` model and the `meshflow_w_num_verts_control` model to dynamically control target vertex counts.
- **Image Conditioning**: Guide the mesh generation using a reference image (automatically handled via DINOv3 visual encoder).
- **Hole Filling**: Triangulates and closes open boundaries automatically for clean output topology.
- **Offline Mode**: Supports pre-downloaded local weights for both MeshFlow and DINOv3 to run completely offline.

---

## Model Setup

Before using the node, you need to place the model files in the appropriate directories.

### 1. MeshFlow Weights
Place the checkpoints under your ComfyUI models directory as follows:

```
ComfyUI/models/facebook/meshflow/
├── meshflow/
│   ├── config.yaml
│   └── model.pth
└── meshflow_w_num_verts_control/
    ├── config.yaml
    └── model.pth
```

### 2. DINOv3 Weights (Offline Support)
To run the DINOv3 visual encoder offline without requesting Meta's CDN, place the Vit-L/16 model weights file `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` here:

```
ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m/
└── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

---

## Node Configuration

### Inputs

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `trimesh` | `TRIMESH` | The input 3D model (Trimesh / Scene) to be remeshed. |
| `model_name` | `COMBO` | Choose between `meshflow` (standard 4096 vertices) and `meshflow_w_num_verts_control` (dynamic vertex counts). |
| `steps` | `INT` | Number of diffusion sampling steps (Default: 28). |
| `guidance_scale` | `FLOAT` | CFG scale for image conditioning (Only active if `reference_image` is connected). |
| `seed` | `INT` | Random seed for sampling latents. |
| `num_verts` | `COMBO` | Target vertex resolution (1024 to 8192). Only active with the `meshflow_w_num_verts_control` model. |
| `device` | `COMBO` | Execution device: `cuda` or `cpu`. |
| `dtype` | `COMBO` | Computation precision: `fp16`, `bf16`, or `fp32`. |
| `compile` | `BOOLEAN` | Uses `torch.compile` on CUDA for faster inference speeds. |
| `use_rmbg` | `BOOLEAN` | Enables automatic background removal and foreground cropping for the reference image. |
| `fill_holes` | `BOOLEAN` | Closes and triangulates boundary loops in the final output mesh. |
| `reference_image` | `IMAGE` *(Optional)* | Optional reference image to guide generation. |

### Outputs

| Name | Type | Description |
| :--- | :--- | :--- |
| `trimesh` | `TRIMESH` | The clean, remeshed output mesh. |
