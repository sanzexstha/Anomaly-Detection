export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset SMAP  --data_path dataset/SMAP --input_c 25    --output_c 25
python main.py --anormly_ratio 1  --num_epochs 1        --batch_size 32     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20




------
