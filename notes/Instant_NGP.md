# Notes on *Instant-NGP — Müller et al., SIGGRAPH 2022*

## Terminologies
- **Graphics Primitive**: Basic geometric rendering element (e.g. voxel, MLP)
- **Multiresolution Hash Encoding**

## Background
- Encoding that maps neural network inputs to higher-dimensional space is needed
- Previous encodings depend on heuristics & structural modifications
    - Complicate training process
    - May be inapplicable to certain tasks
    - Limit performance on GPUs
- Previous encodings:
    - Frequency encoding: Map coordinates into many sine / cosine frequencies $\rightarrow$ Fixed encoding
    - One-blob encoding: Continuous alternative approach of one-hot encoding $\rightarrow$ Limited detail
    - Parametric encoding: Encoding itself contains learnable parameters $\rightarrow$ Requires large (wasteful) memory
    - Sparse parametric encoding: Use learnable parameters, but access small subset for each query $\rightarrow$ Dense grid is wasteful

## Major Improvements
- **Adaptivity**
    - Coarse Resolution: 1:1 mapping from grid points to array entires
    - Fine Resolution: Array treated as hash table & indexed using a spatial hash function
        - Hash collisions cause colliding training gradients to average, prioritizing sparse areas
    - **No structural updates needed**
- **Efficiency**
    - Lookups are $O(1)$
    - Do not require control flow
    - Hash tables can be queried in parallel

## Key Idea
- Map grids to corresponding fixed-size arrays of feature vectors
- Only Two Configurations (for Tuning):
    - Number of parameters $T$
    - Desired finest resolution $N_{\max}$
- Total trainable parameters = encoding parameters + neural network parameters
- Multiresolution Hierarchy:
    - Multiple resolution levels, from coarse to fine grids
    - Coarse = Global structure
    - Fine = Local / High-frequency detail
- Interpolation
    - Retrieve neighboring grid vertex features, and interpolate them
    - For continuity & smooth outputs