export CUDA_VISIBLE_DEVICES=7

python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55
python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 32     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20


python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 32  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55

#
#python main.py --anormly_ratio 0.85 --num_epochs 1   --batch_size 256  --win_size 100  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55
#python main.py --anormly_ratio 0.85  --num_epochs 10      --batch_size 256  --win_size 100   --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55
