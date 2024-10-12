
'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

 -- if to run an experiment with only one negative sample, run the following command:
    python main.py --layers=1 --u_top_k=15 --decay=1e-03 --i_top_k=50  --model=lightGCN  --epochs=201

 -- if to run an experiment with multiple negative samples, run the following command:

    python main.py --layers=1 --u_top_k=15 --decay=1e-03 --i_top_k=50  --model=lightGCN  --epochs=201 --neg_samples=10