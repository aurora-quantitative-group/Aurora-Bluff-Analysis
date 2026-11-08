#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
project_dir=$(pwd)
brew_prefix=$(brew --prefix)
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install cmake boost dlib openblas opencv@4
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ui.txt
if [ ! -d .local/OpenFace ]; then
  mkdir -p .local
  git clone --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git .local/OpenFace
fi
model_dir=.local/OpenFace/lib/local/LandmarkDetector/model/patch_experts
while read -r model link; do
  if [ ! -s "$model_dir/$model" ]; then
    curl -fL --retry 2 "$link" -o "$model_dir/$model.download"
    mv "$model_dir/$model.download" "$model_dir/$model"
  fi
done <<'MODELS'
cen_patches_0.25_of.dat https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1
cen_patches_0.35_of.dat https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1
cen_patches_0.50_of.dat https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1
cen_patches_1.00_of.dat https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1
MODELS
OpenBLAS_HOME="$brew_prefix/opt/openblas" cmake -S "$project_dir/.local/OpenFace" -B "$project_dir/.local/OpenFace/build" \
  -DOpenCV_DIR="$brew_prefix/opt/opencv@4/lib/cmake/opencv4" \
  -DCMAKE_PREFIX_PATH="$brew_prefix" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build .local/OpenFace/build --target FeatureExtraction -j 4
