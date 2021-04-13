# coding=utf-8
# python gridsearch.py --in_f cmd.py --total_cmd -1 --repeat 1 --grid_workers 4 --out_csv cmd.csv --skip_cmd 0 --cuda 0,1,2,3


['ml1m-5-1']  # dataset
[1e-3, 1e-4, 1e-5, 1e-6, 0.0]  # l2
# 'python main.py --model_name Popular --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
'python main.py --model_name BiasedMF --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
'python main.py --model_name NCF --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
# 'python main.py --model_name NCF --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10 --layers [64,32]'
'python main.py --model_name NeuMF --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
# 'python main.py --model_name NeuMF --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10 --layers [64,32]'
'python main.py --model_name ConvNCF --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
'python main.py --model_name FISM --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
'python main.py --model_name GRU4Rec --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10 --max_his 20'
# 'python main.py --model_name GRU4Rec --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
'python main.py --model_name WideDeep --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
# 'python main.py --model_name WideDeep --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10 --layers [64,32]'
'python main.py --model_name NFM --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
# 'python main.py --model_name NFM --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10 --layers [64,32]'
'python main.py --model_name DeepFM --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10'
# 'python main.py --model_name DeepFM --val_metrics ndcg@10 --test_metrics ndcg@5.10.20.50.100,hit@10,recall@10.20,precision@10 --layers [64,32]'

