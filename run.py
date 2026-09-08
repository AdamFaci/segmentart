# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""SegmentART launcher.

    streamlit run run.py

Behind a Jupyter / JupyterHub proxy:
    streamlit run run.py \
        --server.port 8501 \
        --server.baseUrlPath "/user/<username>/proxy/8501/" \
        --server.enableCORS false \
        --server.enableXsrfProtection false
"""
from segmentart.frontend.app import main

main()
