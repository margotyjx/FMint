# ICON-LM
bs=200
# seed=1 && stamp="20240801-173416" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 500000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/new_FMint'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/icon_lm_learn_s$seed-$stamp \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_icon_lm_learn_s$seed-$stamp.log 2>&1 


# # seed=1 && stamp="20240831-141954" && echo "seed=$seed, stamp=$stamp" &&
seed=1 && stamp="20240831-141954" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/lotka_volterra'\
 --analysis_dir /export/jyuan98/learn2correct/analysis/lotka_volterra \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_lotka_volterra.log 2>&1 

# 20240805-012702
seed=1 && stamp="20240831-155209" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 2000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/lorenz_attractor'\
 --analysis_dir /export/jyuan98/learn2correct/analysis/lorenz_attractor\
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_lorenz_attractor.log 2>&1

seed=1 && stamp="20240831-161552" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/dampedharmonic_oscillator'\
 --analysis_dir /export/jyuan98/learn2correct/analysis/dampedharmonic_oscillator \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_dampedharmonic_oscillator.log 2>&1 

#  20240831-141816
seed=1 && stamp="20240831-141816" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/vander_pol'\
 --analysis_dir /export/jyuan98/learn2correct/analysis/vander_pol \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_vander_pol.log 2>&1 

# ========================= finetune ==============================
# # 20240805-013835
# seed=1 && stamp="20240806-013916" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/expo_decay'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/expo_decay-$stamp \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_expo_decay.log 2>&1

# # 20240805-014159
# seed=1 && stamp="20240806-014234" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/law_cooling'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/law_cooling \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_law_cooling.log 2>&1 

# # ========================== 100 sample==============================
# seed=1 && stamp="20240831-194650" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/falling_object' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/falling_object \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_falling_object.log 2>&1 

# seed=1 && stamp="20240831-200043" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/pendulum_gravity'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/pendulum_gravity \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_pendulum_gravity.log 2>&1 

# seed=1 && stamp="20240831-195143" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/fitzhugh_nagumo' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/fitzhugh_nagumo \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_fitzhugh_nagumo.log 2>&1 

# # 20240805-014015
# seed=1 && stamp="20240831-200549" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/drivendamped_pendulum' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/drivendamped_pendulum \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_drivendamped_pendulum.log 2>&1 

# # 20240805-014015
# seed=1 && stamp="20240831-201059" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 2000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/rossler' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/rossler \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_rossler.log 2>&1 

#  # ========================== 25 sample==============================
# seed=1 && stamp="20240907-171954" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/falling_object_25' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/falling_object_25 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_falling_object_25.log 2>&1 

# seed=1 && stamp="20240907-174457" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/pendulum_gravity_25'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/pendulum_gravity_25 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_pendulum_gravity_25.log 2>&1 

# seed=1 && stamp="20240907-172539" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/fitzhugh_nagumo_25' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/fitzhugh_nagumo_25 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_fitzhugh_nagumo_25.log 2>&1 

# # 20240805-014015
# seed=1 && stamp="20240907-174948" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/drivendamped_pendulum_25' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/drivendamped_pendulum_25 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_drivendamped_pendulum_25.log 2>&1 

# # 20240805-014015
# seed=1 && stamp="20240907-175627" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 2000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/rossler_25' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/rossler_25 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_rossler_25.log 2>&1 

#  # ========================== 50 sample==============================
# seed=1 && stamp="20240907-164129" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/falling_object_50' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/falling_object_50 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_falling_object_50.log 2>&1 

# seed=1 && stamp="20240907-175550" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/pendulum_gravity_50'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/pendulum_gravity_50 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_pendulum_gravity_50.log 2>&1 

# seed=1 && stamp="20240907-173506" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/fitzhugh_nagumo_50' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/fitzhugh_nagumo_50 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_fitzhugh_nagumo_50.log 2>&1 

# # 20240805-014015
# seed=1 && stamp="20240907-173723" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/drivendamped_pendulum_50' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/drivendamped_pendulum_50 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_drivendamped_pendulum_50.log 2>&1 

# # 20240805-014015
# seed=1 && stamp="20240907-164922" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 2000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/rossler_50' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/rossler_50 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_rossler_50.log 2>&1 

#  # ========================== 200 sample==============================
# seed=1 && stamp="20240907-163848" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/falling_object_200' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/falling_object_200 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_falling_object_200.log 2>&1 

# seed=1 && stamp="20240907-163158" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/pendulum_gravity_200'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/pendulum_gravity_200 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_pendulum_gravity_200.log 2>&1 

# seed=1 && stamp="20240907-162552" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/fitzhugh_nagumo_200' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/fitzhugh_nagumo_200 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_fitzhugh_nagumo_200.log 2>&1 

# # 20240805-014015
# seed=1 && stamp="20240907-163537" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/drivendamped_pendulum_200' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/drivendamped_pendulum_200 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_drivendamped_pendulum_200.log 2>&1 

# # 20240805-014015
# seed=1 && stamp="20240907-163913" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 2000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/rossler_200' \
#  --analysis_dir /export/jyuan98/learn2correct/analysis/rossler_200 \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_rossler_200.log 2>&1 

# ========================== 1000 sample==============================
seed=1 && stamp="20240907-222447" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/falling_object_1000' \
 --analysis_dir /export/jyuan98/learn2correct/analysis/falling_object_1000 \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_falling_object_1000.log 2>&1 

seed=1 && stamp="20240907-224211" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/pendulum_gravity_1000'\
 --analysis_dir /export/jyuan98/learn2correct/analysis/pendulum_gravity_1000 \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_pendulum_gravity_1000.log 2>&1 

seed=1 && stamp="20240907-223314" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/fitzhugh_nagumo_1000' \
 --analysis_dir /export/jyuan98/learn2correct/analysis/fitzhugh_nagumo_1000 \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_fitzhugh_nagumo_1000.log 2>&1 

# 20240805-014015
seed=1 && stamp="20240907-224723" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/drivendamped_pendulum_1000' \
 --analysis_dir /export/jyuan98/learn2correct/analysis/drivendamped_pendulum_1000 \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_drivendamped_pendulum_1000.log 2>&1 

# 20240805-014015
seed=1 && stamp="20240907-225251" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 2000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/rossler_1000' \
 --analysis_dir /export/jyuan98/learn2correct/analysis/rossler_1000 \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_rossler_1000.log 2>&1 


  


echo "Done."