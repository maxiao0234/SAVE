# Spatial Aggregation Vector Encoding (SAVE)
### SAVE: Encoding Spatial Interactions for Vision Transformers

This repository contains pytorch supported code and configuration of the proposed aggregation encoding for vision transformers.

## Contents
- [Introduction](#Introduction)
- [Classification](#Classification)
- [Citing](#Citing)

## Introduction

## Classification
This implementation is based on the [Deit](https://github.com/facebookresearch/deit) backbone.

```
# An example for training on 8GPUs:
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model save_deit_t16_224 --batch-size 64
```

## Citing
- TO DO
