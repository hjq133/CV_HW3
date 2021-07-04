#!/bin/bash


spring.submit run -p $1 -x SH-IDC1-10-5-36-95 --gpu -n1 --cpus-per-task=5 \
"python -u run.py \
  --task=2 "
