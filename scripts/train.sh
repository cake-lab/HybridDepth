cd ../

rm -rf lightning_logs/version*

# export CUDA_VISIBLE_DEVICES=0

python cli_run.py fit --config configs/config_fit_DDFF12.yaml


