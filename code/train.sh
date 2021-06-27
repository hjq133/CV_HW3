#!/bin/bash


spring.submit run -p $1 -r --gpu -n1 --cpus-per-task=5 \
"python -u run.py \
  --task=1 \
  2>&1 | tee log.train "
