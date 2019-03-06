#!/usr/bin/env bash

# To use this script you need BayesElo installed.

echo "$(cat <<'EOF'
readpgn game_history.pgn
elo
mm
exactdist
ratings >ratings.txt
EOF
)" | ../BayesElo/bayeselo
