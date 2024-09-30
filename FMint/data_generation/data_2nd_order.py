import jax
import jax.numpy as jnp
from collections import namedtuple
from functools import partial
import data_utils
from einshape import jax_einshape as einshape

def rk4_step(y,t, dt, rhs):
	y = jnp.array(y)
	k1 = dt * rhs(t,y)
	k2 = dt * rhs(t + 0.5 * dt, y + 0.5 * k1)
	k3 = dt * rhs(t + 0.5 * dt, y + 0.5 * k2)
	k4 = dt * rhs(t + dt , y +  k3)
	y_next = y + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
	return y_next

def euler_step(y, t, dt, rhs):
	y_next = y + dt * rhs(t, y)
	return y_next



@partial(jax.jit, static_argnums=(-2,-1,))
def lotka_volterra_fn(init, control,ts,dt, params ,k, step_fn):
	alpha, beta, gamma, delta = params

	
	def rhs(t, state):
		x, y = state
		dxdt = alpha * x - beta * x * y
		dydt = delta * x * y - gamma * y
		return jnp.array([dxdt, dydt])
	
	f = partial(step_fn, rhs = rhs, dt = dt)

	def scan_step_fn(state,t): # only need to pass state and t
		state_array = jnp.array(state)
		next_state_array = f(state_array, t)
		next_state = (next_state_array[0], next_state_array[1])
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	fine_x, fine_v = fine_traj


	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(state, t):
			state_array = jnp.array(state)
			next_state_array = step_fn(state, t, k * dt, rhs)
			next_state = (next_state_array[0], next_state_array[1])
			return next_state, next_state

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj

	# Compute large step trajectory using scan for efficiency
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x, large_v = large_traj

	def compute_large_step(idx):
		state = (large_x[idx], large_v[idx])
		big_step_state = step_fn(state, control[idx * k], k * dt, rhs)
		actual_x = fine_x[(idx + 1) * k]
		actual_v = fine_v[(idx + 1) * k]
		error_x = actual_x - big_step_state[0]
		error_v = actual_v - big_step_state[1]
		return jnp.array([error_x, error_v])

	indices = jnp.arange(len(control) // k - 1)
	errors = jax.lax.map(compute_large_step, indices)
	errors = errors.transpose()

	##############################################################

	selected_indices = jnp.arange(0, len(fine_x), k)
	print('selected_indices',selected_indices)


	selected_times = ts[selected_indices]
	# selected_states_x = fine_x[selected_indices]
	
	# selected_states_v = fine_v[selected_indices]


	selected_states = jnp.stack([large_x, large_v], axis = 0)



	return selected_times[:-1],selected_states[:,:-1], errors




@partial(jax.jit, static_argnums=(-2,-1,))
def vander_pol_fn(init, control,ts,dt, eps ,k, step_fn):

	
	def rhs(t, state):
		x, v = state
		dxdt = 1/eps * (x - 1/3 * x**3 - v)
		dvdt = eps * x
		return jnp.array([dxdt, dvdt])
	
	f = partial(step_fn, rhs = rhs, dt = dt)

	def scan_step_fn(state,t): # only need to pass state and t
		state_array = jnp.array(state)
		next_state_array = f(state_array, t)
		next_state = (next_state_array[0], next_state_array[1])
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	fine_x, fine_v = fine_traj


	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(state, t):
			state_array = jnp.array(state)
			next_state_array = step_fn(state, t, k * dt, rhs)
			next_state = (next_state_array[0], next_state_array[1])
			return next_state, next_state

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj

	# Compute large step trajectory using scan for efficiency
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x, large_v = large_traj

	def compute_large_step(idx):
		state = (large_x[idx], large_v[idx])
		big_step_state = step_fn(state, control[idx * k], k * dt, rhs)
		actual_x = fine_x[(idx + 1) * k]
		actual_v = fine_v[(idx + 1) * k]
		error_x = actual_x - big_step_state[0]
		error_v = actual_v - big_step_state[1]
		return jnp.array([error_x, error_v])

	indices = jnp.arange(len(control) // k - 1)
	errors = jax.lax.map(compute_large_step, indices)
	errors = errors.transpose()

	##############################################################

	selected_indices = jnp.arange(0, len(fine_x), k)
	print('selected_indices',selected_indices)


	selected_times = ts[selected_indices]
	# selected_states_x = fine_x[selected_indices]
	
	# selected_states_v = fine_v[selected_indices]


	selected_states = jnp.stack([large_x, large_v], axis = 0)



	return selected_times[:-1],selected_states[:,:-1], errors




@partial(jax.jit, static_argnums=(-2,-1,))
def dampedharmonic_oscillator_fn(init, control,ts,dt, params ,k, step_fn):

	zeta, omega = params
	def rhs(t, state):
		x, v = state
		dxdt = v
		dvdt = -2 * zeta * omega * v - omega**2 * x
		return jnp.array([dxdt, dvdt])
	

	f = partial(step_fn, rhs = rhs, dt = dt)

	def scan_step_fn(state,t): # only need to pass state and t
		state_array = jnp.array(state)
		next_state_array = f(state_array, t)
		next_state = (next_state_array[0], next_state_array[1])
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	fine_x, fine_v = fine_traj


	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(state, t):
			state_array = jnp.array(state)
			next_state_array = step_fn(state, t, k * dt, rhs)
			next_state = (next_state_array[0], next_state_array[1])
			return next_state, next_state

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj

	# Compute large step trajectory using scan for efficiency
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x, large_v = large_traj

	def compute_large_step(idx):
		state = (large_x[idx], large_v[idx])
		big_step_state = step_fn(state, control[idx * k], k * dt, rhs)
		actual_x = fine_x[(idx + 1) * k]
		actual_v = fine_v[(idx + 1) * k]
		error_x = actual_x - big_step_state[0]
		error_v = actual_v - big_step_state[1]
		return jnp.array([error_x, error_v])

	indices = jnp.arange(len(control) // k - 1)
	errors = jax.lax.map(compute_large_step, indices)
	errors = errors.transpose()

	##############################################################

	selected_indices = jnp.arange(0, len(fine_x), k)
	print('selected_indices',selected_indices)


	selected_times = ts[selected_indices]
	# selected_states_x = fine_x[selected_indices]
	
	# selected_states_v = fine_v[selected_indices]


	selected_states = jnp.stack([large_x, large_v], axis = 0)



	return selected_times[:-1],selected_states[:,:-1], errors




@partial(jax.jit, static_argnums=(-2,-1,))
def drivendamped_pendulum_fn(init, control,ts,dt, params ,k, step_fn):

	b, c, A, omega = params
	def rhs(t, state):
		x, v = state
		dxdt = v
		dvdt = -b * v - c * jnp.sin(x) + A * jnp.cos(omega * t)
		return jnp.array([dxdt, dvdt])
	

	f = partial(step_fn, rhs = rhs, dt = dt)

	def scan_step_fn(state,t): # only need to pass state and t
		state_array = jnp.array(state)
		next_state_array = f(state_array, t)
		next_state = (next_state_array[0], next_state_array[1])
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	fine_x, fine_v = fine_traj


	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(state, t):
			state_array = jnp.array(state)
			next_state_array = step_fn(state, t, k * dt, rhs)
			next_state = (next_state_array[0], next_state_array[1])
			return next_state, next_state

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj

	# Compute large step trajectory using scan for efficiency
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x, large_v = large_traj

	def compute_large_step(idx):
		state = (large_x[idx], large_v[idx])
		big_step_state = step_fn(state, control[idx * k], k * dt, rhs)
		actual_x = fine_x[(idx + 1) * k]
		actual_v = fine_v[(idx + 1) * k]
		error_x = actual_x - big_step_state[0]
		error_v = actual_v - big_step_state[1]
		return jnp.array([error_x, error_v])

	indices = jnp.arange(len(control) // k - 1)
	errors = jax.lax.map(compute_large_step, indices)
	errors = errors.transpose()

	##############################################################

	selected_indices = jnp.arange(0, len(fine_x), k)
	print('selected_indices',selected_indices)


	selected_times = ts[selected_indices]
	# selected_states_x = fine_x[selected_indices]
	
	# selected_states_v = fine_v[selected_indices]


	selected_states = jnp.stack([large_x, large_v], axis = 0)



	return selected_times[:-1],selected_states[:,:-1], errors


@partial(jax.jit, static_argnums=(-2,-1,))
def fitzhugh_nagumo_fn(init, control,ts,dt, params ,k, step_fn):

	I, eps, a , b = params
	def rhs(t, state):
		v, w = state
		dvdt = v - v**3/3 - w + I
		dwdt = eps * (v + a - b * w)
		return jnp.array([dvdt, dwdt])
	

	f = partial(step_fn, rhs = rhs, dt = dt)

	def scan_step_fn(state,t): # only need to pass state and t
		state_array = jnp.array(state)
		next_state_array = f(state_array, t)
		next_state = (next_state_array[0], next_state_array[1])
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	fine_x, fine_v = fine_traj


	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(state, t):
			state_array = jnp.array(state)
			next_state_array = step_fn(state, t, k * dt, rhs)
			next_state = (next_state_array[0], next_state_array[1])
			return next_state, next_state

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj

	# Compute large step trajectory using scan for efficiency
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x, large_v = large_traj

	def compute_large_step(idx):
		state = (large_x[idx], large_v[idx])
		big_step_state = step_fn(state, control[idx * k], k * dt, rhs)
		actual_x = fine_x[(idx + 1) * k]
		actual_v = fine_v[(idx + 1) * k]
		error_x = actual_x - big_step_state[0]
		error_v = actual_v - big_step_state[1]
		return jnp.array([error_x, error_v])

	indices = jnp.arange(len(control) // k - 1)
	errors = jax.lax.map(compute_large_step, indices)
	errors = errors.transpose()

	##############################################################

	selected_indices = jnp.arange(0, len(fine_x), k)
	print('selected_indices',selected_indices)


	selected_times = ts[selected_indices]
	# selected_states_x = fine_x[selected_indices]
	
	# selected_states_v = fine_v[selected_indices]


	selected_states = jnp.stack([large_x, large_v], axis = 0)



	return selected_times[:-1],selected_states[:,:-1], errors


@partial(jax.jit, static_argnums=(-2,-1,))
def falling_object_fn(init, control,ts,dt, params ,k, step_fn):

	c = params
	def rhs(t, state):
		x, v = state
		dxdt = v
		dvdt = 9.81 - c * v
		return jnp.array([dxdt, dvdt])
	

	f = partial(step_fn, rhs = rhs, dt = dt)

	def scan_step_fn(state,t): # only need to pass state and t
		state_array = jnp.array(state)
		next_state_array = f(state_array, t)
		next_state = (next_state_array[0], next_state_array[1])
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	fine_x, fine_v = fine_traj


	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(state, t):
			state_array = jnp.array(state)
			next_state_array = step_fn(state, t, k * dt, rhs)
			next_state = (next_state_array[0], next_state_array[1])
			return next_state, next_state

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj

	# Compute large step trajectory using scan for efficiency
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x, large_v = large_traj

	def compute_large_step(idx):
		state = (large_x[idx], large_v[idx])
		big_step_state = step_fn(state, control[idx * k], k * dt, rhs)
		actual_x = fine_x[(idx + 1) * k]
		actual_v = fine_v[(idx + 1) * k]
		error_x = actual_x - big_step_state[0]
		error_v = actual_v - big_step_state[1]
		return jnp.array([error_x, error_v])

	indices = jnp.arange(len(control) // k - 1)
	errors = jax.lax.map(compute_large_step, indices)
	errors = errors.transpose()

	##############################################################

	selected_indices = jnp.arange(0, len(fine_x), k)
	print('selected_indices',selected_indices)


	selected_times = ts[selected_indices]
	# selected_states_x = fine_x[selected_indices]
	
	# selected_states_v = fine_v[selected_indices]


	selected_states = jnp.stack([large_x, large_v], axis = 0)



	return selected_times[:-1],selected_states[:,:-1], errors



@partial(jax.jit, static_argnums=(-2,-1,))
def pendulum_gravity_fn(init, control,ts,dt, params ,k, step_fn):

	l, b = params
	def rhs(t, state):
		x, v = state
		dxdt = v
		dvdt = -9.81/l * jnp.sin(x) - b * v
		return jnp.array([dxdt, dvdt])
	

	f = partial(step_fn, rhs = rhs, dt = dt)

	def scan_step_fn(state,t): # only need to pass state and t
		state_array = jnp.array(state)
		next_state_array = f(state_array, t)
		next_state = (next_state_array[0], next_state_array[1])
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	fine_x, fine_v = fine_traj


	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(state, t):
			state_array = jnp.array(state)
			next_state_array = step_fn(state, t, k * dt, rhs)
			next_state = (next_state_array[0], next_state_array[1])
			return next_state, next_state

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj

	# Compute large step trajectory using scan for efficiency
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x, large_v = large_traj

	def compute_large_step(idx):
		state = (large_x[idx], large_v[idx])
		big_step_state = step_fn(state, control[idx * k], k * dt, rhs)
		actual_x = fine_x[(idx + 1) * k]
		actual_v = fine_v[(idx + 1) * k]
		error_x = actual_x - big_step_state[0]
		error_v = actual_v - big_step_state[1]
		return jnp.array([error_x, error_v])

	indices = jnp.arange(len(control) // k - 1)
	errors = jax.lax.map(compute_large_step, indices)
	errors = errors.transpose()

	##############################################################

	selected_indices = jnp.arange(0, len(fine_x), k)
	print('selected_indices',selected_indices)


	selected_times = ts[selected_indices]
	# selected_states_x = fine_x[selected_indices]
	
	# selected_states_v = fine_v[selected_indices]


	selected_states = jnp.stack([large_x, large_v], axis = 0)



	return selected_times[:-1],selected_states[:,:-1], errors

@partial(jax.jit, static_argnums=(-2,-1,))
def duffing_equa_fn(init, control,ts,dt, params ,k, step_fn):

	delta, alpha, beta, gamma, omega = params
	def rhs(t, state):
		x, v = state
		dxdt = v
		dvdt = gamma * jnp.cos(omega * t) - delta * v - alpha * x - beta * x ** 3 
		return jnp.array([dxdt, dvdt])
	

	f = partial(step_fn, rhs = rhs, dt = dt)

	def scan_step_fn(state,t): # only need to pass state and t
		state_array = jnp.array(state)
		next_state_array = f(state_array, t)
		next_state = (next_state_array[0], next_state_array[1])
		return next_state, next_state
	

	# traj is solution with fine time step dt
	_, fine_traj = jax.lax.scan(scan_step_fn, init, control) #(num, length), e.g. (100,50)
	fine_x, fine_v = fine_traj


	def simulate_large_steps_with_scan(init_state, control, dt, k, step_fn, rhs):
		# Adjust the control times for large steps
		large_step_control = control[::k]

		# Define a modified scan function that applies the large step
		def large_step_scan_fn(state, t):
			state_array = jnp.array(state)
			next_state_array = step_fn(state, t, k * dt, rhs)
			next_state = (next_state_array[0], next_state_array[1])
			return next_state, next_state

		# Use lax.scan to process each large step
		_, large_traj = jax.lax.scan(large_step_scan_fn, init_state, large_step_control)
		return large_traj

	# Compute large step trajectory using scan for efficiency
	large_traj = simulate_large_steps_with_scan(init, control, dt, k, step_fn, rhs)
	large_x, large_v = large_traj

	def compute_large_step(idx):
		state = (large_x[idx], large_v[idx])
		big_step_state = step_fn(state, control[idx * k], k * dt, rhs)
		actual_x = fine_x[(idx + 1) * k]
		actual_v = fine_v[(idx + 1) * k]
		error_x = actual_x - big_step_state[0]
		error_v = actual_v - big_step_state[1]
		return jnp.array([error_x, error_v])

	indices = jnp.arange(len(control) // k - 1)
	errors = jax.lax.map(compute_large_step, indices)
	errors = errors.transpose()

	##############################################################

	selected_indices = jnp.arange(0, len(fine_x), k)
	print('selected_indices',selected_indices)


	selected_times = ts[selected_indices]
	# selected_states_x = fine_x[selected_indices]
	
	# selected_states_v = fine_v[selected_indices]


	selected_states = jnp.stack([large_x, large_v], axis = 0)



	return selected_times[:-1],selected_states[:,:-1], errors

duffing_equa_batch_fn = jax.jit(jax.vmap(duffing_equa_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))
lotka_volterra_batch_fn = jax.jit(jax.vmap(lotka_volterra_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))
vander_pol_batch_fn = jax.jit(jax.vmap(vander_pol_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))
dampedharmonic_oscillator_batch_fn = jax.jit(jax.vmap(dampedharmonic_oscillator_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))
fitzhugh_nagumo_batch_fn = jax.jit(jax.vmap(fitzhugh_nagumo_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))
falling_object_batch_fn = jax.jit(jax.vmap(falling_object_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))
pendulum_gravity_batch_fn = jax.jit(jax.vmap(pendulum_gravity_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))
drivendamped_pendulum_batch_fn = jax.jit(jax.vmap(drivendamped_pendulum_fn, [0,0, None, None, None, None, None], (0,0,0)), static_argnums=(-2,-1,))

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


	
	key, subkey1, subkey2 = jax.random.split(key, num = 3)
	init_x = jax.random.uniform(subkey1, (num,), minval = init_range[0][0], maxval = init_range[0][1])
	init_y = jax.random.uniform(subkey2, (num,), minval = init_range[1][0], maxval = init_range[1][1])


	init = (init_x,init_y)

	print('init x and y',init)
	# traj[0] = init, final is affected by control[-1]

	selected_times, selected_u, errors = ode_batch_fn(init, control,ts,dt, params ,k, rk4_step)



	# e.g., traj (2,50), selected_times (2, 10), selected_u (2,9), errors (2,9)


	return selected_times[...,None],selected_u[...,None], errors[...,None]


if __name__ == "__main__":
	from jax.config import config
	config.update('jax_enable_x64', True)
	import haiku as hk
	import matplotlib.pyplot as plt
	
	seed = 116
	rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
	a = jax.random.uniform(next(rng),  minval = 0.5, maxval = 2.0)
	b = jax.random.uniform(next(rng), minval = 3.0, maxval = 5.0)


	delta = jax.random.uniform(next(rng), minval = 0.1, maxval = 0.5)
	alpha = jax.random.uniform(next(rng), minval = -1, maxval = 1)
	beta = jax.random.uniform(next(rng), minval = 1, maxval = 5)
	gamma = jax.random.uniform(next(rng), minval = 0.3, maxval = 1.5)
	omega = jax.random.uniform(next(rng), minval = 1.0, maxval = 2.0)


	selected_times, selected_u, errors = generate_one_dyn(key = next(rng), ode_batch_fn = pendulum_gravity_batch_fn
							, dt=0.005, length =5000, num =1,  init_range = [(0,2),(1,5)], params = [a,b]
							,k = 100)
	# test du/dt = u, with ground truth u = exp(t)
	selected_u = jnp.transpose(jnp.squeeze(selected_u, axis = -1),(0,2,1))
	errors = jnp.transpose(jnp.squeeze(errors, axis = -1),(0,2,1))
	print('selected_times',selected_times.shape)

	print('selected_sol',selected_u.shape)
	print('errors',errors[0])

	x = selected_u[0, :, 0]
	v = selected_u[0, :, 1]
	# print('x',x)
	# print('v',v)
	# exit()

	# Time points
	time = jnp.arange(selected_u.shape[1])

	# Plotting
	plt.figure(figsize=(10, 6))
	plt.plot(time, x, label='1d')
	plt.plot(time, v, label='2d')
	# plt.plot(time, errors[0, :, 0], label='error 1d')
	plt.xlabel('Time')

	plt.legend()
	plt.grid(True)
	plt.savefig('brusselator.png')

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
