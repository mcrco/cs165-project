#!/bin/bash
set -e

ELAN_DIR="$HOME/.elan"

if [ ! -d "$ELAN_DIR" ]; then
  echo "Installing elan..."
  ( export HOME=/root; curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y )
fi

export PATH="/root/.elan/bin:$PATH"

echo "elan version:"
elan --version