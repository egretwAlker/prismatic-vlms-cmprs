#!/usr/bin/env bash
# Submit LLaVA v1.5 data download as an HTCondor batch job.
#
# Usage (from prismatic-vlms/):
#   bash scripts/submit_download.sh
set -euo pipefail

condor_submit_bid 100 - <<'EOF'
universe       = vanilla
executable     = /home/txia/venvs/salaadpp/bin/python
arguments      = "scripts/preprocess_hf.py --root_dir /fast/txia/salaadpp"
initialdir     = /home/txia/code/salaadpp/prismatic-vlms
request_cpus   = 4
request_memory = 16384
output         = /fast/txia/salaadpp/preprocess.out
error          = /fast/txia/salaadpp/preprocess.err
log            = /fast/txia/salaadpp/preprocess.log
queue
EOF

echo "Submitted. Monitor with:"
echo "  tail -f /fast/txia/salaadpp/preprocess.err"
echo "  tail -f /fast/txia/salaadpp/preprocess.out"
