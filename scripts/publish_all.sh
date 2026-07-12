#!/usr/bin/env bash
# Bootstrap the FIRST coordinated PyPI release of the Cogno ecosystem.
#
# The libs depend on each other by name, so they must be published in dependency
# order (a dependency must exist on PyPI before a dependent can be installed).
# For ongoing releases prefer per-repo Trusted Publishing (.github/workflows/publish.yml
# fired by a GitHub Release) — this script is only for the initial bootstrap, run
# locally with a PyPI API token.
#
# Usage:
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-…                # a PyPI API token (account-scoped for the first run)
#   ./scripts/publish_all.sh --dry-run          # build every lib, upload nothing
#   ./scripts/publish_all.sh                    # build + upload in dependency order
#   ./scripts/publish_all.sh --test             # upload to TestPyPI instead
#
# Assumes every repo is checked out as a sibling directory of cogno-anima
# (the local dir for cogno-anima is `cogno-core`).
set -euo pipefail

DRY_RUN=0
REPOSITORY_URL=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --test) REPOSITORY_URL="https://test.pypi.org/legacy/" ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Dependency order: homeo → synapse → anima → {cortex, mcp, soma}; independents anywhere.
ORDER=(
  cogno-homeo
  cogno-synapse
  cogno-core            # project name: cogno-anima
  cogno-engram
  cogno-cortex
  cogno-mcp
  cogno-meter
  cogno-persona
  cogno-observability
  cogno-herald
  cogno-aegis
  cogno-gateway
  cogno-soma
  cogno-praxis
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the git/ dir holding all repos
echo "ecosystem root: $ROOT"
python3 -m pip install --quiet --upgrade build twine

for repo in "${ORDER[@]}"; do
  dir="$ROOT/$repo"
  [ -d "$dir" ] || { echo "!! missing $dir — skipping"; continue; }
  echo "==================== $repo ===================="
  rm -rf "$dir/dist"
  ( cd "$dir" && python3 -m build >/dev/null )
  twine check "$dir"/dist/*
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "   [dry-run] built, not uploading:"; ls -1 "$dir"/dist
    continue
  fi
  if [ -n "$REPOSITORY_URL" ]; then
    twine upload --repository-url "$REPOSITORY_URL" "$dir"/dist/*
  else
    twine upload "$dir"/dist/*
  fi
  echo "   uploaded $repo ✓"
done

echo "done."
