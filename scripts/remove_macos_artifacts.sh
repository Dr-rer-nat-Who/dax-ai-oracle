#!/bin/bash
# Remove macOS resource forks and metadata files
set -e
find . -name '*.DS_Store' -type f -print -delete
find . -name '__MACOSX' -type d -print0 | xargs -0 -r rm -rf
