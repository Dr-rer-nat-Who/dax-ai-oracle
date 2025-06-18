#!/bin/bash
set -e

# Install Node.js dependencies
npm i

# Install Python requirements
if command -v pip >/dev/null 2>&1; then
    pip install -r requirements.txt
else
    echo "pip not found" >&2
    exit 1
fi
