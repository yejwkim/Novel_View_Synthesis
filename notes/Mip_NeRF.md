# Notes on *Mip-NeRF — Barron et al., CVPR 2021*

## Terminologies
- **Conical Frustum**: 3D geometric shape created by cutting cones perpendicular to their axis
- **Integrated Positional Encoding (IPE)**: Frustum approximation (multivariate Gaussian) + integral over positional encodings
- **Anti-aliasing**: Computer graphics technique to overcome aliasing issue; Smooths jagged edges of diagonals / curves
- **Supersampling**: Render multiple samples per pixel and average them
- **Pre-filtering**: Filter high-frequency detail before rendering to avoid aliasing (e.g. mipmapping)
- **Mipmap**: Image / texture map representation at a set of lower scales

## NeRF Limitation (Motivation for Mip-NeRF)
- Excessively blurred or aliased when image inputs are at different resolutions
- Supersampling by having multiple rays per pixel too costly

## Key Idea
- Extension of NeRF, with each ray sampled as 3D conical frustum (instead of infinitesimal point)
- Performs anti-aliasing via IPE
    - Mitigates aliasing in varying viewing distances and resolutions
- Inspired by mipmapping
    - Nearby regions preserve high-frequency detail
    - Distant regions are smoothly low-pass filtered
- Mip-NeRF does **continuous analytical filtering** during rendering (instead of discrete)
- Pipeline:
    - Each ray interval = conical frustum
    - Approximate frustum as multivariate Gaussian
    - Apply IPE:
        - Integrate positional encodings over the Gaussian region
    - Feed encoded features into NeRF MLP
    - Render anti-aliased radiance field