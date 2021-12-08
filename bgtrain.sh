# source this file to train

nohup python main.py --batch_size 64 --imsize 128 --dataset custom --adv_loss hinge --parallel true --total_step 36000 --pretrained_model 4695 --version sagan_noise &