#!/usr/bin/env bash
# Moves aside the leftovers of the FIRST probe run, the one collected from
# models/udr_s42/ before explicit seeding was fixed.
#
# Those files are not wrong, they are just from a different policy than the
# one the paper reports. Keeping them in results/ next to the current ones is
# how a stale number ends up in a table by accident, and it is the kind of
# thing that is hard to explain at an oral exam.
#
# Nothing is deleted: everything goes to results/legacy_udr_s42/ with a note
# explaining what it is, so the history stays auditable.

set -euo pipefail

DEST="results/legacy_udr_s42"
mkdir -p "$DEST"

STALE=(
  "results/probe_robustness_control.json"   # dataset: probe_dataset_control_450.npz
  "results/probe_results_control.json"      # dataset: probe_dataset_control_450.npz
  "results/probe_controls_control.json"     # no dataset field, same run as above
)

moved=0
for f in "${STALE[@]}"; do
  if [ -f "$f" ]; then
    mv "$f" "$DEST/"
    echo "moved: $f -> $DEST/"
    moved=$((moved + 1))
  else
    echo "already gone: $f"
  fi
done

# the .npz of the old family, if still on disk
for f in probe_dataset_450.npz probe_dataset_control_450.npz; do
  if [ -f "$f" ]; then
    mv "$f" "$DEST/"
    echo "moved: $f -> $DEST/"
    moved=$((moved + 1))
  fi
done

cat > "$DEST/README.md" << 'EOF'
# Legacy results: first probe run (models/udr_s42)

These files come from the first RecurrentPPO + UDR run, trained before the
`--seed` argument was added to `train_udr.py`. That run therefore has an
undefined seed and is not part of the 6-seed group reported in the paper.

The probe dataset and the reservoir control were originally collected from
that checkpoint. They were later recollected from `models/udr_lstm_s42`, which
is part of the 6-seed group, and only the recollected versions are reported.

Kept for auditability. Do not use these numbers.

Mapping:
  probe_robustness_control.json  <- probe_dataset_control_450.npz  (actor: udr_s42)
  probe_results_control.json     <- probe_dataset_control_450.npz  (actor: udr_s42)
  probe_controls_control.json    <- same run, no dataset field recorded
EOF

echo
echo "Moved $moved file(s). Wrote $DEST/README.md"
echo
echo "Note: results/probe_robustness_control_e7.json and _e8.json are NOT"
echo "moved. Their .npz files were deleted after analysis, so their"
echo "provenance cannot be verified from disk. Run the sweep with"
echo "WITH_CONTROLS=1 to regenerate them verifiably."
