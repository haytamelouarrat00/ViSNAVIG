#!/bin/bash
set -euo pipefail

# --- UPDATE THESE PATHS ---
INPUT_DIR="/home/haytam-elourrat/VISNAV/DATA/kitchen/images"
OUT_DIR="/home/haytam-elourrat/VISNAV/DATA/kitchen/mesh"
# --------------------------

DB_PATH="$OUT_DIR/database.db"
SPARSE_DIR="$OUT_DIR/sparse"
MVS_DIR="$OUT_DIR/mvs"

mkdir -p "$SPARSE_DIR" "$MVS_DIR"

# 1. COLMAP SfM
colmap feature_extractor --database_path "$DB_PATH" --image_path "$INPUT_DIR" --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera 1
colmap sequential_matcher --database_path "$DB_PATH"
colmap mapper --database_path "$DB_PATH" --image_path "$INPUT_DIR" --output_path "$SPARSE_DIR"
colmap image_undistorter --image_path "$INPUT_DIR" --input_path "$SPARSE_DIR/0" --output_path "$MVS_DIR" --output_type COLMAP

# 2. OpenMVS Meshing
cd "$MVS_DIR"
InterfaceCOLMAP -i sparse -o scene.mvs
DensifyPointCloud scene.mvs
ReconstructMesh scene_dense.mvs
