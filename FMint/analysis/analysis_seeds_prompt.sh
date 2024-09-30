# ICON-LM
bs=200
# seed=1 && stamp="20240924-124527" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/lotka_volterra'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/lotka_volterra_caption \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_lotka_volterra_caption.log 2>&1 

# seed=1 && stamp="20240924-104543" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 2000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/lorenz_attractor'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/lorenz_attractor_caption\
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_lorenz_attractor_caption.log 2>&1

# seed=1 && stamp="20240924-102230" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/dampedharmonic_oscillator'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/dampedharmonic_oscillator_caption \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_dampedharmonic_oscillator_caption.log 2>&1 


# seed=1 && stamp="20240924-103411" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' --restore_step 1000 \
#  --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/vander_pol'\
#  --analysis_dir /export/jyuan98/learn2correct/analysis/vander_pol_caption \
#  --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
#  --batch_size $bs > out_analysis_vander_pol_caption.log 2>&1 

# ========================= finetune ==============================
seed=1 && stamp="20240926-233837" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/falling_object_100' \
 --analysis_dir /export/jyuan98/learn2correct/analysis/falling_object_caption \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_falling_object_caption.log 2>&1 

seed=1 && stamp="20240928-100915" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/pendulum_gravity_100'\
 --analysis_dir /export/jyuan98/learn2correct/analysis/pendulum_gravity_caption \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_pendulum_gravity_caption.log 2>&1 

seed=1 && stamp="20240928-094714" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/fitzhugh_nagumo_100' \
 --analysis_dir /export/jyuan98/learn2correct/analysis/fitzhugh_nagumo_caption \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_fitzhugh_nagumo_caption.log 2>&1 

# 20240805-014015
seed=1 && stamp="20240928-102059" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 1000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/drivendamped_pendulum_100' \
 --analysis_dir /export/jyuan98/learn2correct/analysis/drivendamped_pendulum_caption \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_drivendamped_pendulum_caption.log 2>&1 

# 20240805-014015
seed=1 && stamp="20240928-103244" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=7 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' --restore_step 2000 \
 --test_data_dirs '/export/jyuan98/learn2correct/data_preparation/rossler_100' \
 --analysis_dir /export/jyuan98/learn2correct/analysis/rossler_caption \
 --restore_dir /export/jyuan98/learn2correct/save/user/ckpts/icon_lm/$stamp \
 --batch_size $bs > out_analysis_rossler_caption.log 2>&1 

echo "Done."