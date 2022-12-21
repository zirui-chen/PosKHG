# PosKHG: A Position-Aware Knowledge Hypergraph Model for Link Prediction

This repository is a formal implementation of our paper "PosKHG: A Position-Aware Knowledge Hypergraph Model for Link Prediction".

To install requirements:

```setup
python 3.7.10
numpy 1.21.5
pytorch 1.12.0
```

To train (and evaluate) the model in the paper, run this command:

```
python main.py --dataset JF17K --num_iterations 200 --batch_size 64 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4 --eval_step 1 --valid_patience 10 -ary 2 -ary 3 -ary 4 -ary 5 -ary 6
```
