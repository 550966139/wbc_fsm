#!/usr/bin/env bash
# Standalone contract/safety tests need only a C++17 compiler. Set
# NLOHMANN_JSON_INCLUDE_DIR to also test Config23 without CMake.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
build_dir=${1:-"$project_dir/build-port-tests"}
mkdir -p -- "$build_dir"
compiler=${CXX:-c++}
flags=(-std=c++17 -O2 -DNDEBUG -Wall -Wextra -Wpedantic -I"$project_dir/include" -I"$project_dir/tests")

for name in policy23 safety; do
  "$compiler" "${flags[@]}" "$project_dir/tests/${name}_test.cpp" -o "$build_dir/${name}_test"
  "$build_dir/${name}_test"
done

if [[ -n ${NLOHMANN_JSON_INCLUDE_DIR:-} ]]; then
  flags+=(-I"$NLOHMANN_JSON_INCLUDE_DIR")
fi
if printf '#include <nlohmann/json.hpp>\n' | "$compiler" "${flags[@]}" -x c++ -E - >/dev/null 2>&1; then
  "$compiler" "${flags[@]}" "-DPROJECT_ROOT_DIR=\"$project_dir\"" \
    "$project_dir/tests/config23_test.cpp" -o "$build_dir/config23_test"
  "$build_dir/config23_test"
else
  printf '%s\n' 'SKIP: Config23 tests require nlohmann/json.hpp (set NLOHMANN_JSON_INCLUDE_DIR).'
fi
