#!/usr/bin/env bash
# Launch the SegmentART annotation server.
#
#   ./run.sh              # foreground, port 8501
#   PORT=8888 ./run.sh    # foreground, custom port
#
# When running on a remote machine, forward the port from your workstation:
#   ssh -L <port>:localhost:<port> <user>@<host>
# then open http://localhost:<port> in a browser.

set -euo pipefail

PORT="${PORT:-8501}"

exec python -m streamlit run run.py \
    --server.port "$PORT" \
    --server.enableCORS false \
    --server.enableXsrfProtection false
