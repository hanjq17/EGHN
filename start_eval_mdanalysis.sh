log_dir=YOUR_LOG_DIR
mkdir -p $log_dir
python3 -u eval_mdanalysis.py --outf $log_dir --model_dir ${MODEL_PATH} 2>&1 | tee $log_dir/out.log
echo "Success"
