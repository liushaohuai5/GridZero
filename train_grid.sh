set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export LD_LIBRARY_PATH=./lib64/
export PYTHONIOENCODING=utf8

python ./experiments/gridsim_v2.py --env_id grid --task_name balance \
  --exp_name powergrid --num_gpus 6 \
  --replay_buffer_size 1000 --num_workers 20 --batch_worker_num 16 --n_parallel 8 \
  --num_simulations 50 --mcts_num_policy_samples 13 --mcts_num_random_samples 4 \
  --mcts_num_expert_samples 4 --PER True \
  --lr_init 0.01 --max_moves 288 --ssl_target 0 --checkpoint_interval 100 --target_update_interval 200 \
  --test_interval 2000 --log_interval 2000 --imitation_log_std -2.0 --explore_scale 1.0 #> debug.txt