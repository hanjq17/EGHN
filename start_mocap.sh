log_dir=YOUR_LOG_DIR
mkdir -p $log_dir
python3 -u main_mocap.py --outf $log_dir 2>&1 | tee $log_dir/out.log

echo "Success"
echo "END"