import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial
import sys
sys.path.append('../')
import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt

import data_preparation.data_1st_order as ode_1
import data_preparation.data_2nd_order as ode_2
import data_preparation.data_3rd_order as ode_3
import data_dynamics as dyn
import data_writetfrecord as datawrite
import data_utils



def generate_expo_decay(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step, testname, testquests):
	'''du/dt = -k * u(t)'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	coeffs_k = jax.random.uniform(next(rng), (eqns,), minval = 0.2, maxval = 0.5)

	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, coeff_k in enumerate(coeffs_k):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_1.generate_one_dyn(key = next(rng), ode_batch_fn = ode_1.expo_decay_batch_fn, 
                                    dt = dt, length = length, num = num, init_range = (50,100),
                                    params = coeff_k, k = nv_step)
            # ts_expand (100, 19, 1)
            # selected_u (100, 19, 1)
            # errors (100, 19, 1)

			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(coeff_k))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# 'train'
	datawrite.write_NeurVec_1st_order_tfrecord(name = name, eqn_type = "expo_decay", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, coeff_k in enumerate(coeffs_k):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_1.generate_one_dyn(key = next(rng), ode_batch_fn = ode_1.expo_decay_batch_fn, 
                                    dt = dt, length = length, num = 25, init_range = (50,100),
                                    params = coeff_k, k = nv_step)

			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(coeff_k))
			all_eqn_captions.append(None)
		utils.print_dot(i)

	datawrite.write_NeurVec_1st_order_tfrecord(name = testname, eqn_type = "expo_decay", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)



def generate_law_cooling(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step, testname,testquests):
	'''du/dt = -k * (u - u_env)'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	coeffs_k = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 1.0)
	T_envs = jax.random.uniform(next(rng), (eqns,), minval = 15, maxval = 25)

	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (coeff_k, T_env) in enumerate(zip(coeffs_k, T_envs)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_1.generate_one_dyn(key = next(rng), ode_batch_fn = ode_1.heat_transfer_batch_fn, 
					dt = dt, length = length, num = num, init_range = (0,100),
					params = [coeff_k, T_env], k = nv_step)

			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}".format(coeff_k, T_env))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_1st_order_tfrecord(name = name, eqn_type = "law_cooling", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	# test
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (coeff_k, T_env) in enumerate(zip(coeffs_k, T_envs)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_1.generate_one_dyn(key = next(rng), ode_batch_fn = ode_1.heat_transfer_batch_fn, 
					dt = dt, length = length, num = 25, init_range = (0,100),
					params = [coeff_k, T_env], k = nv_step)

			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}".format(coeff_k, T_env))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_1st_order_tfrecord(name = testname, eqn_type = "law_cooling", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	

def generate_lotka_volterra(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	'''dx/dt = \alpha * x - \beta * x * y
		dy/dt = \delta * x * y - \gamma * y
	
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	alpha_s = jax.random.uniform(next(rng), (eqns,), minval = 0.05, maxval = 1.2)
	beta_s = jax.random.uniform(next(rng), (eqns,), minval = 0.005, maxval = 0.15)
	gamma_s = jax.random.uniform(next(rng), (eqns,), minval = 0.05, maxval = 1.2)
	delta_s = jax.random.uniform(next(rng), (eqns,), minval = 0.005, maxval = 0.15)

	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (alpha, beta, gamma, delta) in enumerate(zip(alpha_s, beta_s, gamma_s, delta_s)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.lotka_volterra_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(10,100),(5,50)],
					params = [alpha, beta, gamma, delta], k = nv_step)
			
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			# print('ts_expand',ts_expand.shape)
			# print('selected_u',selected_u.shape)
			# print('errors',errors.shape)
			# exit()

			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(alpha, beta,gamma,delta))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "lotka_volterra", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	

	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (alpha, beta, gamma, delta) in enumerate(zip(alpha_s, beta_s, gamma_s, delta_s)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.lotka_volterra_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(10,100),(5,50)],
					params = [alpha, beta, gamma, delta], k = nv_step)
			
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))

			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(alpha, beta,gamma,delta))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "lotka_volterra", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)



def generate_harmonic_oscillator(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	'''d^x/dt^2 = -\omega^2 * x
	
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	omega_s = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 10.0)


	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, omega in enumerate(omega_s):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.harmonic_oscillator_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(-1.0,1.0),(-1.0,1.0)],
					params = omega, k = nv_step)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))

			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(omega))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "harmonic_oscillator", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	# test
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, omega in enumerate(omega_s):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.harmonic_oscillator_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(-1.0,1.0),(-1.0,1.0)],
					params = omega, k = nv_step)

			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))

			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(omega))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "harmonic_oscillator", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	


def generate_van_der_pol(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	'''d^2x/dt^2 - \mu * (1 - x^2) * dx/dt + x = 0
	
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	eps_s = jax.random.uniform(next(rng), (eqns,), minval = 0.05, maxval = 10.0)


	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, eps in enumerate(eps_s):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.vander_pol_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(-1,1),(-0.5,0.5)],
					params = eps, k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(eps))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "vander_pol", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	# test
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, eps in enumerate(eps_s):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.vander_pol_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(-1,1),(-0.5,0.5)],
					params = eps, k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(eps))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "vander_pol", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	


def generate_damped_harmonic_oscillator(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	'''d^2x/dt^2 + 2 zeta * omega * dx/dt + omega^2 * x = 0
	
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	zeta_s = jax.random.uniform(next(rng), (eqns,), minval = .1, maxval = 5.0)
	omega_s = jax.random.uniform(next(rng), (eqns,), minval = .5, maxval = 10.0)
# zeta \in (0.1, 2.0), omega \in (0.5, 5.0)
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (zeta,omega) in enumerate(zip(zeta_s, omega_s)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.dampedharmonic_oscillator_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(-2,2),(-0.1,0.1)],
					params = [zeta, omega], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}".format(zeta,omega))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "dampedharmonic_oscillator", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (zeta,omega) in enumerate(zip(zeta_s, omega_s)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.dampedharmonic_oscillator_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(-2,2),(-0.1,0.1)],
					params = [zeta, omega], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}".format(zeta,omega))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "dampedharmonic_oscillator", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	


def generate_driven_damped_pendulum(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	'''d^2x/dt^2 + b * dx/dt + c * sin(x) = A * cos(omega * t)
	
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	b_s = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.2)
	c_s = jax.random.uniform(next(rng), (eqns,), minval = 1.0, maxval = 2.)
	A_s = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.5)
	omega_s = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 3.)



	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (b,c,A,omega) in enumerate(zip(b_s, c_s, A_s, omega_s)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.drivendamped_pendulum_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(-0.25 * jnp.pi,0.25 * jnp.pi),(-0.5,0.5)],
					params = [b,c,A,omega], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(b,c,A,omega))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "drivendamped_pendulum", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (b,c,A,omega) in enumerate(zip(b_s, c_s, A_s, omega_s)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.drivendamped_pendulum_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(-0.25 * jnp.pi,0.25 * jnp.pi),(-0.5,0.5)],
					params = [b,c,A,omega], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(b,c,A,omega))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "drivendamped_pendulum", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)

def generate_fitzhugh_nagumo(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	'''d^2x/dt^2 + 2 zeta * omega * dx/dt + omega^2 * x = 0
	
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	I_s = jax.random.uniform(next(rng), (eqns,), minval = 0., maxval = 2.0)
	eps_s = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 0.1)
	a_s = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 1.2)
	b_s = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.3)

	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (I, eps, a, b) in enumerate(zip(I_s, eps_s, a_s, b_s)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.fitzhugh_nagumo_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(-1,1),(-0.5,0.5)],
					params = [I, eps, a, b], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(I, eps, a, b))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "fitzhugh_nagumo", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	# test
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (I, eps, a, b) in enumerate(zip(I_s, eps_s, a_s, b_s)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.fitzhugh_nagumo_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(-1,1),(-0.5,0.5)],
					params = [I, eps, a, b], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(I, eps, a, b))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "fitzhugh_nagumo", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	

def generate_falling_object(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	'''
	dxdt = v
	dvdt = g - c * v
	
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	c_s = jax.random.uniform(next(rng), (eqns,), minval = 0.01, maxval = 2.0)


	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, c in enumerate(c_s):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.falling_object_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(0,100),(0,2)],
					params = c, k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(c))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "falling_object", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, c in enumerate(c_s):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.falling_object_batch_fn, 
					dt = dt, length = length, num =100, init_range = [(0,100),(0,2)],
					params = c, k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(c))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "falling_object", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	

def generate_pendulum_gravity(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	'''
	dxdt = v
	dvdt = -g/l * sin(x)
	
	'''
	# how to type pi in jax? 

	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	l_s = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 2.0)
	b_s = jax.random.uniform(next(rng), (eqns,), minval = 0.05, maxval = 1.0)



	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i,(l,b) in enumerate(zip(l_s,b_s)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.pendulum_gravity_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(0,0.25 * jnp.pi),(-0.25 * jnp.pi,0.25 * jnp.pi)],
					params = [l,b], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}".format(l,b))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "pendulum_gravity", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i,(l,b) in enumerate(zip(l_s,b_s)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.pendulum_gravity_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(0,0.25 * jnp.pi),(-0.25 * jnp.pi,0.25 * jnp.pi)],
					params = [l,b], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}".format(l,b))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "pendulum_gravity", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	



def generate_lorenz(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	''' 
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	sigma_s = jax.random.uniform(next(rng), (eqns,), minval = 5.0, maxval = 15.0)
	rho_s = jax.random.uniform(next(rng), (eqns,), minval = 100., maxval = 140.)
	beta_s = jax.random.uniform(next(rng), (eqns,), minval = 1.5, maxval = 3.5)


	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i,(sigma,rho,beta) in enumerate(zip(sigma_s, rho_s, beta_s)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_3.generate_one_dyn(key = next(rng), ode_batch_fn = ode_3.lorenz_attractor_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(-5,5),(-5,5),(0,25)],
					params = [sigma, rho, beta], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}".format(sigma,rho,beta))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_3rd_order_tfrecord(name = name, eqn_type = "lorenz_attractor", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i,(sigma,rho,beta) in enumerate(zip(sigma_s, rho_s, beta_s)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_3.generate_one_dyn(key = next(rng), ode_batch_fn = ode_3.lorenz_attractor_batch_fn, 
					dt = dt, length = length, num = 100, init_range = [(-5,5),(-5,5),(0,25)],
					params = [sigma, rho, beta], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}".format(sigma,rho,beta))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_3rd_order_tfrecord(name = testname, eqn_type = "lorenz_attractor", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	

def generate_thomas(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	''' 
    dx/dt = sin(y) - b * x
    dy/dt = sin(z) - b * y
    dz/dt = sin(x) - b * z
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	b_s = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.3)



	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, b in enumerate(b_s):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_3.generate_one_dyn(key = next(rng), ode_batch_fn = ode_3.thomas_attractor_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(-1,1),(-1,1),(-1,1)],
					params = b, k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(b))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_3rd_order_tfrecord(name = name, eqn_type = "thomas_attractor", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, b in enumerate(b_s):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_3.generate_one_dyn(key = next(rng), ode_batch_fn = ode_3.thomas_attractor_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(-1,1),(-1,1),(-1,1)],
					params = b, k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}".format(b))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_3rd_order_tfrecord(name = testname, eqn_type = "thomas_attractor", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
def generate_rossler(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step,testname,testquests):
	''' 
    dx/dt = -y - z
    dy/dt = x + a * y
    dz/dt = b + z * (x - c)
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	a_s = jax.random.uniform(next(rng), (eqns,), minval = 0.25, maxval = 0.4)
	b_s = jax.random.uniform(next(rng), (eqns,), minval = 0.25, maxval = 0.3)
	c_s = jax.random.uniform(next(rng), (eqns,), minval = 9., maxval = 13.)



	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i,(a,b,c) in enumerate(zip(a_s,b_s,c_s)):
	# for i, b in enumerate(b_s):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_3.generate_one_dyn(key = next(rng), ode_batch_fn = ode_3.rossler_attractor_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(-1,1),(-1,1),(-1,1)],
					params = [a,b,c], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}".format(a,b,c))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_3rd_order_tfrecord(name = name, eqn_type = "rossler_attractor", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i,(a,b,c) in enumerate(zip(a_s,b_s,c_s)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_3.generate_one_dyn(key = next(rng), ode_batch_fn = ode_3.rossler_attractor_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(-1,1),(-1,1),(-1,1)],
					params = [a,b,c], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}".format(a,b,c))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_3rd_order_tfrecord(name = testname, eqn_type = "rossler_attractor", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
def generate_duff_equa(seed, eqns, quests, length, dt, num, caption_mode, name,nv_step, testname,testquests):
	'''
	d^2x/dt^2 + delta * dx/dt + alpha * x + beta * x^3 = gamma * cos(omega * t)
	
	'''
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	delta_s = jax.random.uniform(next(rng), (eqns,), minval = 0.1, maxval = 0.3)
	alpha_s = jax.random.uniform(next(rng), (eqns,), minval = 0, maxval = 0.5)
	beta_s = jax.random.uniform(next(rng), (eqns,), minval = 1, maxval = 3)
	gamma_s = jax.random.uniform(next(rng), (eqns,), minval = 0.3, maxval = 0.7)
	omega_s = jax.random.uniform(next(rng), (eqns,), minval = 0.5, maxval = 2.5)



	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (delta, alpha, beta, gamma, omega) in enumerate(zip(delta_s, alpha_s, beta_s, gamma_s, omega_s)):
		for j in range(quests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.duffing_equa_batch_fn, 
					dt = dt, length = length, num = num, init_range = [(0,3),(-0.5,0.5)],
					params = [delta, alpha, beta, gamma, omega], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(delta, alpha, beta, gamma, omega))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = name, eqn_type = "duffing_equa", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)
	
	all_ts = []; all_ys = []; all_errors = []; all_params = []; all_eqn_captions = []
	for i, (delta, alpha, beta, gamma, omega) in enumerate(zip(delta_s, alpha_s, beta_s, gamma_s, omega_s)):
		for j in range(testquests):
			# each of shape (num, length, 1)
			ts_expand,selected_u, errors = ode_2.generate_one_dyn(key = next(rng), ode_batch_fn = ode_2.duffing_equa_batch_fn, 
					dt = dt, length = length, num = 25, init_range = [(0,3),(-0.5,0.5)],
					params = [delta, alpha, beta, gamma, omega], k = nv_step)
			selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
			errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
			all_ts.append(ts_expand)
			all_ys.append(selected_u)
			all_errors.append(errors)
			all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(delta, alpha, beta, gamma, omega))
			all_eqn_captions.append(None)
		utils.print_dot(i)
		
	# name is 'train' or 'test'
	datawrite.write_NeurVec_2nd_order_tfrecord(name = testname, eqn_type = "duffing_equa", 
				all_params = all_params, all_eqn_captions = all_eqn_captions,
				all_ts = all_ts, all_ys = all_ys, all_errors = all_errors,alpha=nv_step)

	
def main(argv):
	for key, value in FLAGS.__flags.items():
			print(value.name, ": ", value._value, flush=True)
	
	
	name = '{}/{}'.format(FLAGS.dir, FLAGS.name)
	testname = '{}/{}'.format(FLAGS.dir, 'test')

	if not os.path.exists(FLAGS.dir):
		os.makedirs(FLAGS.dir)
		
	if 'expo_decay' in FLAGS.eqn_types:
		generate_expo_decay(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)
		

	if 'law_cooling' in FLAGS.eqn_types:
		generate_law_cooling(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)
		

	if 'lotka_volterra' in FLAGS.eqn_types:
		generate_lotka_volterra(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)
		
	if 'fitzhugh_nagumo' in FLAGS.eqn_types:
		generate_fitzhugh_nagumo(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)      
		

	if 'vander_pol' in FLAGS.eqn_types:
		generate_van_der_pol(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)   

	if 'drivendamped_pendulum' in FLAGS.eqn_types:
		generate_driven_damped_pendulum(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)   
		
	if 'dampedharmonic_oscillator' in FLAGS.eqn_types:
		generate_damped_harmonic_oscillator(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)   

	if 'falling_object' in FLAGS.eqn_types:
		generate_falling_object(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)  
		
	if 'pendulum_gravity' in FLAGS.eqn_types:
		generate_pendulum_gravity(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)  
		
	if 'lorenz_attractor' in FLAGS.eqn_types:
		generate_lorenz(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)  

	if 'thomas_attractor' in FLAGS.eqn_types:
		generate_thomas(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname, testquests=FLAGS.testquests)  
		

	if 'rossler_attractor' in FLAGS.eqn_types:
		generate_rossler(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step,
							testname = testname,testquests=FLAGS.testquests)  
		
	if 'duff_equa' in FLAGS.eqn_types:
		generate_duff_equa(seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, 
							dt = FLAGS.dt, num = FLAGS.num, caption_mode = FLAGS.caption_mode, name = name,nv_step=FLAGS.nv_step)
		







if __name__ == "__main__":

	import tensorflow as tf
	import os
	os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
	tf.config.set_visible_devices([], device_type='GPU')

	FLAGS = flags.FLAGS
	flags.DEFINE_string('caption_mode', None, 'mode for caption')
	flags.DEFINE_integer('num', 100, 'number of systems in each equation') # fixed paramter
	flags.DEFINE_integer('quests', 25, 'number of questions in each operator')
	flags.DEFINE_integer('testquests', 25, 'number of questions in each operator')
	flags.DEFINE_integer('eqns', 100, 'number of equations')
	flags.DEFINE_integer('length', 40, 'length of trajectory and control')
	flags.DEFINE_integer('nv_step', 5, 'step size of NeurVec')
	flags.DEFINE_integer('mfc_iters', 1000, 'iterations for solving mfc')
	flags.DEFINE_float('mfc_tol', 1e-10, 'res tolerance for solving mfc')
	flags.DEFINE_boolean('mfc_verbose', False, 'verbose for solving mfc')
	flags.DEFINE_float('dt', 0.02, 'time step in dynamics')
	flags.DEFINE_float('dx', 0.02, 'time step in dynamics')
	flags.DEFINE_integer('nu_nx_ratio', 1, 'nu_nx_ratio in mfc_hj')
	flags.DEFINE_string('name', 'data', 'name of the dataset')
	flags.DEFINE_string('dir', '.', 'name of the directory to save the data')
	flags.DEFINE_list('eqn_types', [], 'list of equations for data generation')
	flags.DEFINE_list('write', [], 'list of features to write')

	flags.DEFINE_integer('seed', 1, 'random seed')

	app.run(main)
