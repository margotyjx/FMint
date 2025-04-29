gpu=1

dir=FMINT
testeqns=10
testquests=5
traineqns=10

# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types expo_decay --length 500 --dt 0.05 --nv_step 10 --seed 102 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types law_cooling --length 500 --dt 0.05 --nv_step 10 --seed 103 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types lotka_volterra --length 20000 --dt 0.001 --nv_step 100 --seed 104 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 10000 --dt 0.01 --eqn_types vander_pol --nv_step 10 --seed 106 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 10000 --dt 0.001 --eqn_types dampedharmonic_oscillator --nv_step 100 --seed 107 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 5000 --dt 0.005 --eqn_types fitzhugh_nagumo --nv_step 100 --seed 108 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 1000 --dt 0.01 --eqn_types falling_object --nv_step 20 --seed 109 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 1000 --dt 0.01 --eqn_types pendulum_gravity --nv_step 20 --seed 110 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 1000 --dt 0.01 --eqn_types drivendamped_pendulum --nv_step 20 --seed 111 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testquests --quests $testquests --eqn_types lorenz_attractor --length 20000 --dt 0.001 --nv_step 100 --seed 112 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testquests --quests $testquests --eqn_types rossler_attractor --length 20000 --dt 0.001 --nv_step 100 --seed 113 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode test --name test --eqns $testquests --quests $testquests --eqn_types duff_equa --length 20000 --dt 0.001 --nv_step 100 --seed 114 &&


# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types expo_decay --length 500 --dt 0.05 --nv_step 10 --seed 2 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types law_cooling --length 500 --dt 0.05 --nv_step 10 --seed 3 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types lotka_volterra --length 20000 --dt 0.001 --nv_step 100 --seed 4 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 10000 --dt 0.01 --eqn_types vander_pol --nv_step 10 --seed 6 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 10000 --dt 0.001 --eqn_types dampedharmonic_oscillator --nv_step 100  --seed 7 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 5000 --dt 0.005 --eqn_types fitzhugh_nagumo --nv_step 100 --seed 8 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 1000 --dt 0.01 --eqn_types falling_object --nv_step 20 --seed 9 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 1000 --dt 0.01 --eqn_types pendulum_gravity --nv_step 20 --seed 10 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 1000 --dt 0.01 --eqn_types drivendamped_pendulum --nv_step 20 --seed 11 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types lorenz_attractor --length 20000 --dt 0.001 --nv_step 100 --seed 12 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types rossler_attractor --length 20000 --dt 0.001 --nv_step 100 --seed 13 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_neuralvec.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types duff_equa --length 20000 --dt 0.001 --nv_step 100 --seed 14 &&


echo "Done"


