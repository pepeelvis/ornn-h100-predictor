#!/bin/bash
# Daily driver for the Ornn H100 index predictor. Invoked by the
# com.ornn.h100predictor LaunchAgent at 15:00 America/New_York, ahead of
# the index's ~16:30 ET publish. Safe to also run by hand.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"
source .venv/bin/activate
cd src
python predict_today.py
python build_dashboard.py
cd "$ROOT"

# Publish via a dedicated, standalone clone (~/.ornn-publish-clone), not this
# worktree. This worktree's .git is a linked worktree sharing a primary
# checkout with several unrelated Clawdmeter project workspaces; concurrent
# git activity from any of those siblings (e.g. a checkpoint commit) can
# transiently break git resolution here ("fatal: not a git repository") --
# happened 2026-08-09, then again 2026-08-20 where it outlasted the retry
# guard and silently skipped a day's publish. The dedicated clone has its
# own independent .git, so sibling-workspace contention can't reach it.
PUBLISH_CLONE="$HOME/.ornn-publish-clone"
publish() {
    git -C "$PUBLISH_CLONE" pull --ff-only origin main
    cp "$ROOT/predictions_log.csv" "$PUBLISH_CLONE/predictions_log.csv"
    cp "$ROOT/lighter_basis_log.csv" "$PUBLISH_CLONE/lighter_basis_log.csv"
    rsync -a --delete "$ROOT/docs/" "$PUBLISH_CLONE/docs/"
    if [ -n "$(git -C "$PUBLISH_CLONE" status --porcelain predictions_log.csv lighter_basis_log.csv docs/)" ]; then
        git -C "$PUBLISH_CLONE" add predictions_log.csv lighter_basis_log.csv docs/
        git -C "$PUBLISH_CLONE" commit -m "Daily update: $(date +%Y-%m-%d)"
    fi
    # Unconditional: also catches a prior attempt that committed locally but
    # failed to push (retry below would otherwise see a clean tree and no-op).
    git -C "$PUBLISH_CLONE" push origin main
}
if ! publish; then
    echo "Publish failed, retrying in 10s..." >&2
    sleep 10
    publish
fi

# Best-effort: sync this worktree's local copy of just the generated files to
# what just published, so a future manual session here isn't looking at a
# stale/diverged version of them. predict_today.py/build_dashboard.py above
# always leave these paths locally modified (build_dashboard.py stamps a
# fresh "last updated" time every run), so a plain `git pull` would conflict
# on them virtually every day -- `checkout <ref> -- <paths>` sidesteps that
# by directly overwriting just these three known-fully-generated paths with
# origin's version (safe: nothing hand-authored ever lives in them), without
# touching HEAD or any unrelated uncommitted work elsewhere in the worktree
# (e.g. in-progress src/ edits from a manual session). Not on the critical
# path -- publish already succeeded above -- so a failure here is just a
# warning, never fatal to today's run.
(git fetch origin main && git checkout origin/main -- predictions_log.csv lighter_basis_log.csv docs/) \
    || echo "Warning: could not sync worktree's local copy after publish (non-fatal, publish itself succeeded)" >&2
