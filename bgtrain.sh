# source this file to train

# nohup python train.py --data_path 'data/covers/classes' --save_path 'output2' --parallel &

nohup python main.py --batch_size 64 --imsize 128 --dataset custom --adv_loss hinge --parallel true --total_step 30000 &

