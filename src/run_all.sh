#!/bin/bash
# Full GPU pipeline: main collection -> probe training data -> MASK replication
cd "$(dirname "$0")"
export TORCH_DISABLE_NATIVE_JIT=1
python run_main.py > ../logs/run_main.log 2>&1
python probe_data.py > ../logs/probe_data.log 2>&1
python mask_run.py > ../logs/mask_run.log 2>&1
echo FINISHED > ../logs/pipeline_done
