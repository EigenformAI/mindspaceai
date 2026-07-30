#!/usr/bin/env bash
# Serve the projector locally. It fetches its data over HTTP, so opening
# index.html from the filesystem will not work — file:// is blocked by CORS.
cd "$(dirname "$0")"
PORT="${1:-8133}"
echo "http://localhost:$PORT/"
python3 -m http.server "$PORT"
