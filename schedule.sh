mkdir -p logs/txt

python  train_tdeed.py  --model KOVO_big | tee logs/txt/KOVO_big.log
python  train_tdeed.py  --model KOVOTF_big | tee logs/txt/KOVOTF_big.log
python  train_tdeed.py  --model KOVOMAM_big | tee logs/txt/KOVOMAM_big.log