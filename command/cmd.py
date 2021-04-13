# coding=utf-8
# python gridsearch.py --in_f cmd.py --total_cmd -1 --repeat 1 --grid_workers 4 --out_csv cmd.csv --skip_cmd 0 --cuda 0,1,2,3

# [0.1, 0.01, 0.001]  # lr
# [1e-3, 1e-4]  # l2
# 'python main.py --model_name BiasedMF --max_epoch 1 --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'


[0.001]  # lr
[0, 1e-4, 1e-5, 1e-6]  # l2
["'[128, 64]'", "'[64]'"]  # layers
[16]  # eval_batch_size
# 'python main.py --model_name BiasedMF --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
# 'python main.py --model_name GRU4Rec --val_metrics ndcg@10 --test_metrics ndcg@ 5.10.20.50.100,hit@10,recall@10.20,precision@10'
'python main.py --model_name DeepFM --dataset ml100k-5-1 --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
