#!/usr/bin/env bash
set -euo pipefail
release_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
set +u
source /opt/ros/humble/setup.bash
set -u
cd "$release_root/sim_runtime/navigation"
extra=()
if [[ -n "${G1_CXX_RUNTIME:-}" ]]; then extra+=("-DG1_CXX_RUNTIME=$G1_CXX_RUNTIME"); fi
colcon build --packages-select serial local_planner --cmake-args -DBUILD_TESTING=OFF "${extra[@]}"
