---
date: '0'
title: 'VISNAV: Visual Servoing Navigation Framework'
github: 'https://github.com/haytamelouarrat00/ViSNAVIG'
cover: './demo.png'
tech:
  - PyTorch
  - 3D Gaussian Splatting
  - XFeat
  - MoGe-2
  - Open3D
  - Python
showInProjects: true
---

A research framework for visual servoing navigation in photo-realistic 3D scene representations. A virtual camera is
placed inside a 3D Gaussian Splatting, mesh, or NeRF scene and iteratively moved to minimize the visual error between
its rendered view and a real target image — replicating real-world camera navigation purely from image feedback. The
system implements both Feature-Based Visual Servoing (FBVS), using XFeat sparse matching and MoGe-2 monocular depth to
build an IBVS interaction matrix in SE(3), and Direct Visual Servoing (DVS) based on dense photometric error
minimization, achieving sub-millimeter translational and sub-degree angular convergence accuracy.