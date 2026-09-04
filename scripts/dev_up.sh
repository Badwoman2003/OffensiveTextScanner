#!/usr/bin/env bash
# Spin up the full stack for local development.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "[dev_up] .env created from .env.example"
fi

echo "[dev_up] starting redis + backend + workers"
docker compose up -d redis backend worker-cpu flower

echo "[dev_up] tailing backend logs (ctrl-c to stop)"
docker compose logs -f backend worker-cpu
