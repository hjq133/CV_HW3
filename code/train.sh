#!/bin/bash


spring.submit run -p $1 -x SH-IDC1-10-5-36-95,SH-IDC1-10-5-36-96,SH-IDC1-10-5-37-32 --gpu -n1 --cpus-per-task=5 \
"python -u run.py \
  --task=1 \
  2>&1 | tee log.train "
