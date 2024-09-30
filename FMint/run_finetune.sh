# seed=1 && stamp="20240801-173416" && echo "seed=$seed, stamp=$stamp" &&
seed=1 && stamp="20240805" && echo "seed=$seed, stamp=$stamp" &&

# CUDA_VISIBLE_DEVICES=7 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/dampedharmonic_oscillator' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_dampedharmonic_oscillator_$stamp.log 2>&1 

# CUDA_VISIBLE_DEVICES=7 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/vander_pol' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_vander_pol_$stamp.log 2>&1

# CUDA_VISIBLE_DEVICES=7 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/lorenz_attractor' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_lorenz_attractor_$stamp.log 2>&1

# CUDA_VISIBLE_DEVICES=7 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/lotka_volterra' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_lotka_volterra_$stamp.log 2>&1

# # ============= OOD ==================

# CUDA_VISIBLE_DEVICES=7 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/expo_decay' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_expo_decay_$stamp.log 2>&1

# CUDA_VISIBLE_DEVICES=7 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/law_cooling' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_law_cooling_$stamp.log 2>&1

# # =========== finetuning ===========

# CUDA_VISIBLE_DEVICES=1 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/falling_object_1000' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_log/Finetune_falling_object_1000_$stamp.log 2>&1


CUDA_VISIBLE_DEVICES=0 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/fitzhugh_nagumo_1000' \
  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 900000 \
  --save_freq 100 > Finetune_log/Finetune_fitzhugh_nagumo_1000_$stamp.log 2>&1

CUDA_VISIBLE_DEVICES=0 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/pendulum_gravity_1000' \
  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 900000 \
  --save_freq 100 > Finetune_log/Finetune_pendulum_gravity_1000_$stamp.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/drivendamped_pendulum_1000' \
  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 900000 \
  --save_freq 100 > Finetune_log/Finetune_drivendamped_pendulum_1000_$stamp.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/rossler_1000' \
  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 900000 \
  --save_freq 100 > Finetune_log/Finetune_rossler_1000_$stamp.log 2>&1


# # ============= not trained yet ==============

# CUDA_VISIBLE_DEVICES=7 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/thomas' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_thomas_$stamp.log 2>&1

# CUDA_VISIBLE_DEVICES=7 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 25 --train_data_dirs '/export/jyuan98/learn2correct/data_preparation/duffing' \
#   --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 900000 \
#   --save_freq 100 > Finetune_duffing _$stamp.log 2>&1


echo 'Done.'