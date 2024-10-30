cd ../

rm -rf lightning_logs/version*

# export CUDA_VISIBLE_DEVICES=1

python cli_run.py test  --config configs/config_test_DDFF12.yaml
