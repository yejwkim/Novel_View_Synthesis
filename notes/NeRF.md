# Notes on *NeRF: Neural Radiance Fields — Mildenhall et al., ECCV 2020*

## Terminologies
- **View Synthesis**: Image generation from new camera viewpoint
- **MLP**: Multilayer perceptron; Fully-connected neural network
    - Improvement from single-layer perceptron by introducing hidden layers to allow for nonlinear functions / continous mappings
- **Radiance**: Directional light; Amount of light traveling through a point in a particular direction
- **Radiance Field** $L(x,y,z,\theta,\phi)$: Specifies how much light is emitted for every point in space and viewing direction
- **Volume Density**: Amount of matter / opacity at a point in space
- **Volume Rendering**: Combining radiance & density along camera rays to generate an image

## Major Improvements
- **Continuous neural scene representation**
    - Scene representation via continuous 5D radiance field using an MLP
    - Avoids discretized voxel grids, overcoming the storage costs
    - Allows for view-dependent appearance
- **Differentiable Volume Rendering**
    - Classical volume rendering used to synthesize images from the continuous field
    - Gradient descent can be used for optimization, as field is now continuous (previous works were discrete)
    - Eliminated the usage of explicit meshes or ground-truth 3D geometry
- **Higher-Quality NVS**
    - More photorealistic renderings
    - Better captures complex geometry in real scenes
- **Efficient Representation Learning**
    - *Positional Encoding*: Enables modeling of high-frequency details
    - *Hierarchical Sampling*: Improves rendering efficiency by focusing computation on important regions

## Key Idea
- Model: $F_\theta:(x,y,z,\theta,\phi)\rightarrow(c,\sigma)$ ($\mathbb{R}^5\rightarrow\mathbb{R}^4$)
    - $(x,y,z)$: Position
    - $(\theta, \phi)$: Direction
    - $c$: View-dependent Emitted Radiance (RGB); Depends on position & direction
    - $\sigma$: Volume Density; Depends only on position
- Pipeline
    1. Generate a set of 3D points by marching camera rays through the scene
    2. Use such points and their corresponding 2D viewing direction $(\theta, \phi)$ as input to neural network to produce an output set of colors (radiance) and densities
        - MLP will be used as the neural network
    3. Use classical volume rendering techniques to convert set of radiances and densities into a 2D image
    4. Use gradient descent as a means for model optimization by minimizing RMSE between synthesized (rendered) color and ground truth pixel color
- Why MLP, not CNN?
    - Purpose of NeRF is not on image understanding, but on continuous coordinate function approximation
    - Avoids discretized voxel grids
- Finding Volume Density (+ Geometry Feature)
    - Input: 3D coordinate
    - 8 fully-connected hidden layers, with 256 hidden units (channels) and ReLU activation per layer
    - Output: Volume density & 256-dimensional feature vector
- Finding Radiance
    - Concatenate feature vector from previous step with 3D Cartesian unit vector of direction
    - Input: Concatenated feature vector
    - 1 fully-connected hidden layer (with 128 channels & ReLU)
    - Output: View dependent RGB color (radiance)
- Classical Volume Rendering
    - Render a pixel by accumulating color contributions along a camera ray
    - Estimate the integral via stratified sampling + quadrature
- Positional Encoding
    - Sinusoidal mapping of coordinates into higher-dimensional multi-frequency features
    - Helps MLP learn high-frequency scene details that raw coordinates struggle to represent
- Hierarchical Sampling
    - Coarse network
        - Uniformly sample points along rays
        - Estimates where important scene content exists
    - Fine network
        - Allocates more samples to high-density / high-importance regions
    - Focuses computation near visible geometry
    - Reduces unnecessary sampling near empty space

## Limitations
- **Scene-specific**: Each scene requires its own training process