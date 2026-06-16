# SpaVCCA
Data and code for SpaVCCA

Spatial transcriptomics enables the characterization of gene expression within intact tissue architecture, but 
identifying domains across batches, platforms, or conditions remains challenging due to technical variability and 
batch effects. Existing methods often fail to simultaneously achieve effective batch correction and preservation of 
spatial structure. Here, we propose SpaVCCA, a unified framework for spatial transcriptomics integration that combines 
a graph convolutional Variational Autoencoder (VAE) with a Canonical Correlation Analysis (CCA)-based alignment loss 
and a graph contrastive objective. The graph convolutional encoder captures local spatial dependencies, while the CCA 
loss aligns latent representations across batches. The contrastive loss further preserves local neighborhood structure 
during integration. We evaluate SpaVCCA on diverse datasets, including different platforms (Visium, Stereo-seq, MERFISH) 
with multiple slices, cross-technology integration, and 3D spatial data. Compared to existing state-of-the-art (SOTA) 
methods, SpaVCCA delivers substantial performance advancements. Overall, SpaVCCA provides a scalable and effective 
solution for integrating spatial transcriptomics data, characterizing spatially organized cellular states and 
supporting applications in drug development and disease research. 

Tutorials can be accesssed in: https://github.com/momoGZH/SpaVCCA/tree/main/tutorials
Dataset can be accesssed in: https://drive.google.com/drive/folders/17BUyqHunRjCwRoUw-i4QZBmq6dUyNqGZ?usp=sharing
