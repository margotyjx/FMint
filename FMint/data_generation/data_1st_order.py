import jax
import jax.numpy as jnp
from collections import namedtuple
from functools import partial
import data_utils
from einshape import jax_einshape as einshape

def rk4_step(y,t, dt, rhs):
	k1 = dt * rhs(t,y)
	k2 = dt * rhs(t, y + 0.5 * k1)
	k3 = dt * rhs(t , y + 0.5 * k2)
	k4 = dt * rhs(t, y +  k3)
	y_next = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
	return y_next

def euler_step(y, t, dt, rhs):
	y_next = y + dt * rhs(t, y)
	return y_next



@partial(jax.jit, static_argnums=(-2,-1,))
def expo_decay_fn(init, control,ts,dt, coeff_k ,k, step_fn):
	rhs = lambda t, y: -coeff_k * y

	f = partial(step_fn, rhs = rhs, dt = dt)
	def scan_step_fn(y,t):
		y_next = f(y,t)
		return y_next,y_next
	

	# traj is solution with fine time step dt
	_, traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	

	# Define a function to simulate using large time steps
	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(T, t):
			T_next = step_fn(T, t, k * dt, rhs)
			return T_next, T_next

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj
	
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x = large_traj



	def predict_next(y,t):
		return step_fn(y,t,k*dt, rhs)
	
	# selected_indices = jnp.arange(0, len(traj), k)
	# selected_u = traj[selected_indices]
	# selected_times = ts[selected_indices]


	selected_times = ts[::k][:-1]
	large_x = large_traj[:-1]
	predictions = jax.vmap(predict_next)(large_x, selected_times)

	selected_u = traj[::k][1:]

	errors = selected_u - predictions


	return selected_times,large_x, errors

@partial(jax.jit, static_argnums=(-2,-1,))
def heat_transfer_fn(init, control, ts, dt, params, k, step_fn):

	coeff_k, T_env = params


	rhs = lambda t, T: -coeff_k * (T - T_env)  # du/dt = -k * (u - u_env)

	f = partial(step_fn, rhs = rhs, dt = dt)
	def scan_step_fn(y,t):
		y_next = f(y,t)
		return y_next,y_next
	

	# traj is solution with fine time step dt
	_, traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	

	# Define a function to simulate using large time steps
	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(T, t):
			T_next = step_fn(T, t, k * dt, rhs)
			return T_next, T_next

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj
	
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x = large_traj



	def predict_next(y,t):
		return step_fn(y,t,k*dt, rhs)
	
	# selected_indices = jnp.arange(0, len(traj), k)
	# selected_u = traj[selected_indices]
	# selected_times = ts[selected_indices]


	selected_times = ts[::k][:-1]
	large_x = large_traj[:-1]
	predictions = jax.vmap(predict_next)(large_x, selected_times)

	selected_u = traj[::k][1:]

	errors = selected_u - predictions


	return selected_times,large_x, errors



expo_decay_batch_fn = jax.jit(jax.vmap(expo_decay_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))
heat_transfer_batch_fn = jax.jit(jax.vmap(heat_transfer_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))

# @partial(jax.jit, static_argnames=('ode_batch_fn','sub_batch_fn','ts','length','num','k'))
@partial(jax.jit, static_argnames=('ode_batch_fn','length','num','k'))
def generate_one_dyn(key, ode_batch_fn, dt, length, num,  init_range, params,k):
	'''
	generate data for dynamics
	@param 
		key: jax.random.PRNGKey
		ode_batch_fn: e.g. ode_auto_const_batch_fn, jitted function
		dt: float, time step
		length: int, length of time series
		num: int, number of samples
		k_sigma, k_l: float, kernel parameters
		init_range: tuple, range of initial values
		coeffs: tuple, coefficients of the dynamics, will be unpacked and passed to ode_batch_fn
		control: 2D array (num, length), control signal, if None, generate with Gaussian process
	@return
		ts: 2D array (num, length, 1), time series
		control: 2D array (num, length, 1), control signal
		traj: 2D array (num, length, 1), trajectory
	'''
	ts = jnp.arange(length) * dt
	ts_expand = einshape("i->ji", ts, j = num) # 100,50
	control = ts_expand


	
	key, subkey1 = jax.random.split(key, num = 2)
	init = jax.random.uniform(subkey1, (num,), minval = init_range[0], maxval = init_range[1])
	print('init',init)

	# traj[0] = init, final is affected by control[-1]



	selected_times, selected_u, errors = ode_batch_fn(init, control,ts,dt, params ,k, euler_step)


	# e.g., traj (2,50), selected_times (2, 10), selected_u (2,9), errors (2,9)


	return selected_times[...,None],selected_u[...,None], errors[...,None]


if __name__ == "__main__":
	from jax.config import config
	config.update('jax_enable_x64', True)
	import haiku as hk
	import matplotlib.pyplot as plt
	
	seed = 2
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	coeff_k = jax.random.uniform(next(rng), minval = 0.2, maxval = 0.5)
	print('coeff_k',coeff_k)

	selected_times, selected_u, errors = generate_one_dyn(key = next(rng), ode_batch_fn = expo_decay_batch_fn
							, dt=0.05, length = 500, num =1,  init_range = (100,200), params = coeff_k
							,k = 10)
	# test du/dt = u, with ground truth u = exp(t)

	print('selected_times',selected_times)

	print('selected_sol',selected_u[0])
	print('errors',errors[0])


	x = selected_u[0, :, 0]


	# Time points
	time = jnp.arange(selected_u.shape[1])

	# Plotting
	plt.figure(figsize=(10, 6))
	plt.plot(time, x, label='x')

	plt.xlabel('Time')

	plt.legend()
	plt.grid(True)
	plt.savefig('law_cooling.png')

	# print('ts_expand',ts_expand.shape)
	# print('traj',traj.shape)
	# print('selected_u',selected_u.shape)
	# print('errors',errors.shape)

	
	# init = 1
	# dt = 0.02
	# ts = jnp.arange(50) * dt
	# control = ts



	# traj,selected_times,selected_u, errors = ode_auto_const_fn(init, ts,dt,-1.0, 10, rk4_step)
	

	# assert jnp.allclose(traj, jnp.exp(ts))
