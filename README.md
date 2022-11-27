# Spatial Aggregation Vector Encoding (SAVE)

Spatial Aggregation Vector Encoding (SAVE) is a method for establishing spatial information for vectors (elements of input tokens, queries, keys, or values) in vision transformers. It can be plug-and-play in vectors, even with other position encoding methods. It aggregates part of surrounding vectors with spatial contextual connections by establishing two-dimensional relationships.This repository contains pytorch supported code and configuration of the proposed aggregation encoding for vision transformers:

<img src="figs/fig1.png">

## Classification
This implementation is based on the [Deit](https://github.com/facebookresearch/deit) backbone.

```
# An example for training on 8GPUs:
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model save_deit_t16_224 --batch-size 64
```

To modify the SAVE configurations, go to `models.py`.

## Citing
- TO DO
