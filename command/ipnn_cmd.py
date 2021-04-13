# coding=utf-8
# python gridsearch.py --in_f ipnn_cmd.py --total_cmd -1 --repeat 1 --grid_workers 4 --out_csv ipnn.csv --skip_cmd 0 --cuda 0,1,2,3

[0, 1e-4, 1e-5, 1e-6]  # l2
["'[256, 128, 64]'", "'[128, 64]'", "'[64]'"]  # layers
[4]  # eval_batch_size
'python main.py --model_name IPNN --dataset ml100k-5-1 --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
