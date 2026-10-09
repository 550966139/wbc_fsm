#!/bin/bash
# cap_session.sh — one-command hardware capture session (two passive listeners).
# Usage: bash ~/wbc_fsm/scripts/cap_session.sh <tag> [minutes]
# Artifacts: ~/wbc_fsm/logs/hw_cap/<tag>/{hw_<tag>.npz, hw_<tag>_sport.npz, *.log}
# Read-only: the listeners subscribe and never publish. Safe to run any time.
set -u
TAG=${1:?tag}
MIN=${2:-20}
DUR=$((MIN * 60))
DIR=$HOME/wbc_fsm/logs/hw_cap/$TAG
mkdir -p "$DIR"
export PYTHONPATH=/home/unitree/unitree_sdk2_python/unitree_sdk2_python-master
PY=python3

pkill -f "cap_hw.py .*hw_${TAG}.npz" 2>/dev/null
pkill -f "cap_hw_sport.py .*hw_${TAG}_sport.npz" 2>/dev/null
sleep 1

date +%s.%N > "$DIR/t_start"
nohup $PY $HOME/wbc_fsm/scripts/cap_hw.py "$DUR" "$DIR/hw_${TAG}.npz" > "$DIR/cap_hw.log" 2>&1 &
P1=$!
nohup $PY $HOME/wbc_fsm/scripts/cap_hw_sport.py "$DUR" "$DIR/hw_${TAG}_sport.npz" > "$DIR/cap_sport.log" 2>&1 &
P2=$!
echo "capture started tag=$TAG pid=$P1/$P2 dur=${MIN}min -> $DIR"
echo "watch: tail -f $DIR/cap_hw.log"
