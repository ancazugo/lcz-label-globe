#!/usr/bin/env bash
# Build the globe and publish it to the `deploy` branch.
#
# The payload is ~63 MB of binaries regenerated from GeoPackages under /maps,
# which CI cannot reach. Committing that to main would rewrite tens of MB on
# every build, so main carries source only and the built site lives on an
# orphan-ish `deploy` branch that is force-pushed -- its history never grows.
#
# Netlify: connect the repo, set the production branch to `deploy`, publish
# directory `.`, build command empty.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=label_site
WT=.deploy-wt

# SQLITE_USE_OGR_VFS: gpkg reads fail with "database is locked" otherwise when
# the filesystem is full. NUMPY_MADVISE_HUGEPAGE: large allocations stall ~40 s
# on this host without it.
SQLITE_USE_OGR_VFS=YES NUMPY_MADVISE_HUGEPAGE=0 \
  python src/build_label_site.py --output-dir "$OUT" "$@"

test -f "$OUT/index.html"        || { echo "no index.html -- build failed"; exit 1; }
test -f "$OUT/data/manifest.json" || { echo "no manifest -- build failed"; exit 1; }

git worktree remove --force "$WT" 2>/dev/null || true
git worktree add -B deploy "$WT"

rsync -a --delete --exclude .git "$OUT"/ "$WT"/
cp deploy/netlify.toml "$WT"/netlify.toml

git -C "$WT" add -A
git -C "$WT" commit -m "build $(date -u +%FT%TZ)" || echo "nothing changed"
git -C "$WT" push -f origin deploy
git worktree remove --force "$WT"
echo "pushed to deploy"
