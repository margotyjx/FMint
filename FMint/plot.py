import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import utils
import tensorflow as tf
import os
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_pred(equation, prompt, mask, query, ground_truth, pred, to_tfboard = True):
	'''
	plot the figure
	@param 
		equation: string
		prompt: 2D array, [len(prompt), prompt_dim]
		mask: 1D array, [len(prompt)]
		query: 2D array, [len(qoi), qoi_k_dim]
		ground_truth: 2D array, [len(qoi), qoi_v_dim]
		pred: 2D array, [len(qoi), qoi_v_dim]
	@return
		the figure
	'''
	figure = plt.figure(figsize=(6, 4))
	plt.subplot(1,1,1)
	plt.plot(query[:,0], ground_truth[:,0], 'k-', label='Ground Truth')
	plt.plot(query[:,0], pred[:,0], 'r--', label='Prediction')
	plt.xlabel('key')
	plt.ylabel('value')
	plt.title("eqn:{}, qoi".format(equation))
	plt.legend()
	plt.tight_layout()
	if to_tfboard:
		return utils.plot_to_image(figure)
	else:
		return figure

def plot_subfig(ax, t, x_true, x_pred = None, mask = None):
	'''
	plot the subfigure (only plot the first dimension)
	@param 
		ax: the subfigure
		t: 2D array, [len(t), k_dim]
		x_pred: 1D array, [len(t), v_dim]
		x_true: 1D array, [len(t), v_dim]
	@return
		the subfigure
	'''
	if mask is not None:
		t = t[mask]
		x_true = x_true[mask]
		if x_pred is not None:
			x_pred = x_pred[mask]
	ax.plot(t[:,0], x_true[:,0], 'ko', markersize=3, label='GT')
	if x_pred is not None:
		ax.plot(t[:,0], x_pred[:,0], 'ro', markersize=1, label='Pred')
		ax.legend()
	ax.set_xlabel('key')
	ax.set_ylabel('value')
	

def plot_all(equation, prompt, mask, query, query_mask, ground_truth, pred = None,
						demo_num = 5, k_dim = 1, v_dim = 1,
						to_tfboard = True, ):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		prompt: 2D array, [len(prompt), prompt_dim]
		mask: 1D array, [len(prompt)]
		query: 2D array, [len(qoi), query_dim]
		query_mask: 1D array, [len(qoi)]
		ground_truth: 2D array, [len(qoi), qoi_v_dim]
		pred: 2D array, [len(qoi), qoi_v_dim], skip if None
		demo_num: int, number of demos
		k_dim: int, max dim for k in prompt
		v_dim: int, max dim for v in prompt
	@return
		the figure
	'''
	fig_col_half_num = 2
	fig_row_num = demo_num // fig_col_half_num + 1
	fig, axs = plt.subplots(fig_row_num, 2* fig_col_half_num, figsize=(12, 8))
	fig.subplots_adjust(hspace=0.5, wspace=0.5)
	fig.suptitle("eqn:{}".format(equation))
	_, prompt_dim = jnp.shape(prompt)
	if prompt_dim != k_dim + v_dim + demo_num + 1:
		raise ValueError("Error in plot: prompt_dim != k_dim + v_dim + demo_num + 1")
	for i in range(demo_num):
		row_ind = i // fig_col_half_num
		col_ind = i % fig_col_half_num
		# plot demo cond
		mask_cond = jnp.abs(prompt[:, -demo_num - 1 + i] - 1) < 0.01 # around 1
		demo_cond_k = prompt[mask_cond, :k_dim]
		demo_cond_v = prompt[mask_cond, k_dim:k_dim+v_dim]
		ax = axs[row_ind, 2*col_ind]
		plot_subfig(ax, demo_cond_k, demo_cond_v)
		ax.set_title("demo {}, cond".format(i))
		# plot demo qoi
		mask_qoi = jnp.abs(prompt[:, -demo_num - 1 + i] + 1) < 0.01 # around -1
		demo_qoi_k = prompt[mask_qoi, :k_dim]
		demo_qoi_v = prompt[mask_qoi, k_dim:k_dim+v_dim]
		ax = axs[row_ind, 2*col_ind+1]
		plot_subfig(ax, demo_qoi_k, demo_qoi_v)
		ax.set_title("demo {}, qoi".format(i))
	# plot pred
	mask_cond = jnp.abs(prompt[:, -1] - 1) < 0.01 # around 1
	cond_k = prompt[mask_cond, :k_dim]
	cond_v = prompt[mask_cond, k_dim:k_dim+v_dim]
	row_ind = demo_num // fig_col_half_num
	col_ind = demo_num % fig_col_half_num
	ax = axs[row_ind, 2*col_ind]
	plot_subfig(ax, cond_k, cond_v)
	ax.set_title("quest, cond")
	ax = axs[row_ind, 2*col_ind+1]
	plot_subfig(ax, query, ground_truth, pred, query_mask.astype(bool))
	ax.set_title("quest, qoi")
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig
	

def get_plot_k_index(k_mode, equation):
	if ("lorenz" in equation) or ("thomas" in equation) or ("rossler" in equation):
		k_index = 0
		qoi_k_Dflag = 3
	elif ("law" in equation) or ("expo" in equation):
		k_index = 0
		qoi_k_Dflag = 1
	else:
		k_index = 0
		qoi_k_Dflag = 2
	# if ("oscillator" in equation) or ("lotka_volterra" in equation) or ("vander_pol" in equation) or ("duffing" in equation) or ("falling" in equation) or ("nagumo" in equation) or ("pendulum" in equation):
	# 	k_index = 0
	# 	qoi_k_2D_flag = True
	# else:
	# 	k_index = 0
	# 	qoi_k_2D_flag = False
	return k_index, qoi_k_Dflag

def plot_all_in_one(equation, caption, prompt, mask, query, query_mask, ground_truth, pred,
									config, to_tfboard = True, ):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		prompt: 2D array, [len(prompt), prompt_dim]
		mask: 1D array, [len(prompt)]
		query: 2D array, [len(qoi), query_dim]
		query_mask: 1D array, [len(qoi)]
		ground_truth: 2D array, [len(qoi), qoi_v_dim]
		pred: 2D array, [len(qoi), qoi_v_dim]
		demo_num: int, number of demos
		k_dim: int, max dim for k in prompt
		v_dim: int, max dim for v in prompt
		k_mode: the mode for the key
	@return
		the figure
	'''
	plt.close('all')
	demo_num, k_dim, v_dim, k_mode, = config['demo_num'], config['k_dim'], config['v_dim'], config['k_mode']
	cond_k_index, qoi_k_index = get_plot_k_index(k_mode, equation)
	prompt = np.array(prompt)
	mask = np.array(mask)
	query = np.array(query)
	query_mask = np.array(query_mask)
	ground_truth = np.array(ground_truth)
	pred = np.array(pred)

	# check consistency between mask and prompt
	assert np.sum(mask) == np.sum(np.abs(prompt[:, (k_dim + v_dim):]))

	fig, axs = plt.subplots(3, 1, figsize=(10, 8))
	fig.subplots_adjust(hspace=0.5, wspace=0.0)
	fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot cond quest
	mask_cond_quest = np.abs(prompt[:, -1] - 1) < 0.01 # around 1
	# cond_quest = prompt[mask_cond_quest, :k_dim+v_dim]  # [cond_len_in_use, k_dim+v_dim]
	axs[0].plot(prompt[mask_cond_quest, cond_k_index], prompt[mask_cond_quest,k_dim], 'k+', markersize=7, label='cond quest')
	# plot pred
	query_mask = query_mask.astype(bool)
	axs[1].plot(query[query_mask,qoi_k_index], ground_truth[query_mask,0], 'k+', markersize=7, label='ground truth')
	axs[2].plot(query[query_mask,qoi_k_index], ground_truth[query_mask,0], 'k+', markersize=7, label='ground truth')
	axs[2].plot(query[query_mask,qoi_k_index], pred[query_mask,0], 'r+', markersize=7, label='pred')
	cond_mask_nonzero_num = []
	qoi_mask_nonzero_num = []
	for i in range(demo_num):
		mask_cond_i = np.abs(prompt[:, -demo_num-1+i] - 1) < 0.01 # around 1
		mask_qoi_i = np.abs(prompt[:, -demo_num-1+i] + 1) < 0.01 # around -1
		if np.sum(mask_cond_i) > 0 and np.sum(mask_qoi_i) > 0:  # demo that is used
			cond_mask_nonzero_num.append(np.sum(mask_cond_i))
			qoi_mask_nonzero_num.append(np.sum(mask_qoi_i))
			# NOTE: we don't need mask because prompt is multiplied by mask when constructed
			# cond_i = prompt[mask_cond_i, :k_dim+v_dim] # [cond_len_in_use, k_dim+v_dim]
			# qoi_i = prompt[mask_qoi_i, :k_dim+v_dim] # [qoi_len_in_use, k_dim+v_dim]
			axs[0].plot(prompt[mask_cond_i,cond_k_index], prompt[mask_cond_i,k_dim], 'o', markersize=3, label='cond {}'.format(i), alpha = 0.5)
			axs[1].plot(prompt[mask_qoi_i,qoi_k_index], prompt[mask_qoi_i,k_dim], 'o', markersize=3, label='qoi {}'.format(i), alpha = 0.5)
	cond_mask_nonzero_num = np.array(cond_mask_nonzero_num)
	qoi_mask_nonzero_num = np.array(qoi_mask_nonzero_num)
	axs[0].set_xlabel('key'); axs[0].set_ylabel('value')
	axs[1].set_xlabel('key'); axs[1].set_ylabel('value')
	axs[2].set_xlabel('key'); axs[2].set_ylabel('value')
	axs[0].set_title("cond, {} demos mask nonzero num: {}, quest mask nonzero num: {}".format(
									cond_mask_nonzero_num.shape[0], cond_mask_nonzero_num, np.sum(mask_cond_quest)))
	axs[1].set_title("demo qoi, {} demos mask nonzero num: {}".format(qoi_mask_nonzero_num.shape[0], qoi_mask_nonzero_num))
	axs[2].set_title("quest qoi, mask nonzero num: {}".format(np.sum(query_mask)))
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig
	
	

def plot_data(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 2:
		fig, axs = plt.subplots(8, 1, figsize=(16, 12))
	elif qoi_k_Dflag == 3:
		fig, axs = plt.subplots(8, 1, figsize=(16, 12))
	else:
		fig, axs = plt.subplots(4, 1, figsize=(12, 8))
	fig.subplots_adjust(hspace=1.2, wspace=0.0)
	caption = ""
	fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot cond quest
	
	# cond_quest = prompt[mask_cond_quest, :k_dim+v_dim]  # [cond_len_in_use, k_dim+v_dim]
	for i in range(len(data.demo_cond_mask)):
		axs[0].plot(data.demo_cond_k[i, data.demo_cond_mask[i,:].astype(bool), k_index], 
								data.demo_cond_v[i, data.demo_cond_mask[i,:].astype(bool), 0], 'o', markersize=3, label='u1, cond {}'.format(i), alpha = 0.5)
		if qoi_k_Dflag >= 2:
			axs[4].plot(data.demo_cond_k[i, data.demo_cond_mask[i,:].astype(bool), k_index], 
								data.demo_cond_v[i, data.demo_cond_mask[i,:].astype(bool), 1], 'o', markersize=3, label='u2, cond {}'.format(i), alpha = 0.5)
		
		axs[1].plot(data.demo_qoi_k[i, data.demo_qoi_mask[i,:].astype(bool), k_index],
								data.demo_qoi_v[i, data.demo_qoi_mask[i,:].astype(bool), 0], 'o', markersize=3, label='u1 qoi {}'.format(i), alpha = 0.5)
		if qoi_k_Dflag >= 2:
			axs[5].plot(data.demo_qoi_k[i, data.demo_qoi_mask[i,:].astype(bool), k_index],
								data.demo_qoi_v[i, data.demo_qoi_mask[i,:].astype(bool), 1], 'o', markersize=3, label='u2 qoi {}'.format(i), alpha = 0.5)
		
	axs[0].plot(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), k_index], 
							data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='cond quest')
	if qoi_k_Dflag >= 2:
		axs[4].plot(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), k_index], 
							data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1], 'k+', markersize=7, label='cond quest')
	
	axs[1].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='u1 qoi quest')
	if qoi_k_Dflag >= 2:
		axs[5].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 1], 'k+', markersize=7, label='u2 qoi quest')
		
	# plot pred for both dimensions 
	axs[2].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'k+', markersize=7, label='u1 qoi quest')
	axs[3].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 0], 'r+', markersize=7, label='u1 qoi pred')
	if qoi_k_Dflag >= 2:
		axs[6].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 1], 'k+', markersize=7, label='u2 qoi quest')
		axs[7].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 1], 'r+', markersize=7, label='u2 qoi pred')

	demo_cond_len = np.sum(data.demo_cond_mask, axis=1)
	demo_qoi_len = np.sum(data.demo_qoi_mask, axis=1)
	quest_cond_len = np.sum(data.quest_cond_mask, axis=1)
	quest_qoi_len = np.sum(data.quest_qoi_mask, axis=1)

	axs[0].set_xlabel('key'); axs[0].set_ylabel('value')
	axs[1].set_xlabel('key'); axs[1].set_ylabel('value')
	axs[2].set_xlabel('key'); axs[2].set_ylabel('value')
	axs[3].set_xlabel('key'); axs[3].set_ylabel('value')
	axs[0].set_title("u1 cond, demo len: {}, quest len: {}".format(demo_cond_len, quest_cond_len))
	axs[1].set_title("u1 qoi, demo len: {}, quest len: {}".format(demo_qoi_len, quest_qoi_len))
	axs[2].set_title("u1 quest qoi ground truth, len: {}".format(quest_qoi_len))
	axs[3].set_title("u1 quest qoi prediction, len: {}".format(quest_qoi_len))

	if qoi_k_Dflag >= 2:
		axs[4].set_xlabel('key'); axs[3].set_ylabel('value')
		axs[5].set_xlabel('key'); axs[4].set_ylabel('value')
		axs[6].set_xlabel('key'); axs[5].set_ylabel('value')
		axs[7].set_xlabel('key'); axs[7].set_ylabel('value')

		axs[4].set_title("u2 cond, demo len: {}, quest len: {}".format(demo_cond_len, quest_cond_len))
		axs[5].set_title("u2 qoi, demo len: {}, quest len: {}".format(demo_qoi_len, quest_qoi_len))
		axs[6].set_title("u2 quest qoi ground truth, len: {}".format(quest_qoi_len))
		axs[7].set_title("u2 quest qoi prediction, len: {}".format(quest_qoi_len))
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig

def correction_plot_data(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 3:
		fig, axs = plt.subplots(3, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.5, wspace=0.0)
	elif qoi_k_Dflag == 2:
		fig, axs = plt.subplots(2, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.5, wspace=0.0)
	else:
		fig, axs = plt.subplots(1, 1, figsize=(12, 8))
	
	caption = ""
	# fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot pred for both dimensions 
	if qoi_k_Dflag >= 2:
		axs[0].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'g+', markersize=12, label='ground truth error')
		axs[0].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 0], 'co', markersize=9, label='predicted error', alpha = 0.5)
		axs[1].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 1], 'g+', markersize=12, label='ground truth error')
		axs[1].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 1], 'co', markersize=9, label='predicted error', alpha = 0.5)
		if qoi_k_Dflag == 3:
			axs[2].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 2], 'g+', markersize=12, label='ground truth error')
			axs[2].plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 2], 'co', markersize=9, label='predicted error', alpha = 0.5)
		
	else:
		axs.plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							label[0, data.quest_qoi_mask[0,:].astype(bool), 0], 'g+', markersize=12, label='ground truth error')
		axs.plot(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index],
							pred[data.quest_qoi_mask[0,:].astype(bool), 0], 'co', markersize=9, label='predicted error', alpha = 0.5)

	if qoi_k_Dflag >= 2:
		axs[0].set_xlabel('time')
		axs[0].legend(loc = "upper right", fontsize="25")
		axs[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		axs[1].set_xlabel('time')
		axs[1].legend(loc = "upper right", fontsize="25")
		axs[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[1].yaxis.set_minor_locator(ticker.LinearLocator(8))
		if qoi_k_Dflag == 3:
			axs[2].set_xlabel('time')
			axs[2].legend(loc = "upper right", fontsize="25")
			axs[2].yaxis.set_major_locator(ticker.LinearLocator(4))
			axs[2].yaxis.set_minor_locator(ticker.LinearLocator(8))
			for item in ([axs[2].xaxis.label, axs[2].yaxis.label] + axs[2].get_xticklabels() + axs[2].get_yticklabels()):
				item.set_fontsize(25)

		for item in ([axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
			item.set_fontsize(25)
		for item in ([axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
			item.set_fontsize(25)
	else:
		axs.set_xlabel('time')
		axs.legend(loc = "upper right",fontsize="25")

		axs.yaxis.set_major_locator(ticker.LinearLocator(4))
		axs.yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs.xaxis.label, axs.yaxis.label] + axs.get_xticklabels() + axs.get_yticklabels()):
			item.set_fontsize(25)
	
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig
	
def ODE_plot_data(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 3:
		fig, axs = plt.subplots(3, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	elif qoi_k_Dflag == 2:
		fig, axs = plt.subplots(2, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	else:
		fig, axs = plt.subplots(1, 1, figsize=(12, 8))
	
	caption = ""
	# fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot pred for both dimensions 
 
	# print("cond time: ",data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	# print("qoi time: ", data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	ind_cond = np.argsort(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	if qoi_k_Dflag >= 2:
		fine_u_0 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		fine_u_1 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		coarse_u_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		coarse_u_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_0, 'r+', markersize=15, label='fine ode')
		axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'g+', markersize=15, label='coarse ode')
		axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u_0, 'o', markersize=9, label='FMint ode', alpha = 0.5)

		axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_1, 'r+', markersize=15, label='fine ode')
		axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond], 'g+', markersize=15, label='coarse ode')
		axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u_1, 'o', markersize=9, label='FMint ode', alpha = 0.5)
		if qoi_k_Dflag == 3:
			fine_u_2 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]
			coarse_u_2 = (pred[data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]

			axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
								fine_u_2, 'r+', markersize=15, label='fine ode')
			axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
								(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond], 'g+', markersize=15, label='coarse ode')
			axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
								coarse_u_2, 'o', markersize=9, label='FMint ode', alpha = 0.5)

	else:
		fine_u = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]

		coarse_u = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]

		# print("qoi time: ", (data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]).shape)
		# print("quest cond v shape: ", (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0]).shape)
		# print("label shape: ", (label[0, data.quest_qoi_mask[0,:].astype(bool), 0]).shape)
		# print("fine u: ", fine_u.shape)


		axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u, 'r+', markersize=15, label='fine ode')
		axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'g+', markersize=15, label='coarse ode')
		axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u, 'o', markersize=9, label='FMint ode', alpha = 0.5)

	if qoi_k_Dflag >= 2:
		axs[0].set_xlabel('time')
		axs[0].set_ylabel('value')
		axs[0].legend(loc = "upper right", fontsize="18")
		axs[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		axs[1].set_xlabel('time')
		axs[1].set_ylabel('value')
		axs[1].legend(loc = "upper right",fontsize="18")
		axs[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[1].yaxis.set_minor_locator(ticker.LinearLocator(8))
		if qoi_k_Dflag == 3:
			axs[2].set_xlabel('time')
			axs[2].set_ylabel('value')
			axs[2].legend(loc = "upper right",fontsize="18")
			axs[2].yaxis.set_major_locator(ticker.LinearLocator(4))
			axs[2].yaxis.set_minor_locator(ticker.LinearLocator(8))
			for item in ([axs[2].xaxis.label, axs[2].yaxis.label] + axs[2].get_xticklabels() + axs[2].get_yticklabels()):
				item.set_fontsize(25)

		for item in ([axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
			item.set_fontsize(25)
		for item in ([axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
			item.set_fontsize(25)
	else:
		axs.set_xlabel('time'); axs.set_ylabel('value')
		axs.legend(loc = "upper right", fontsize="18")

		axs.yaxis.set_major_locator(ticker.LinearLocator(4))
		axs.yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs.xaxis.label, axs.yaxis.label] + axs.get_xticklabels() + axs.get_yticklabels()):
			item.set_fontsize(25)
	
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig
	
def ODE_plot_3D2D(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)

	# if qoi_k_Dflag == 3:
	# 	fig, axs = plt.subplots(3, 1, figsize=(16, 12))
	# 	fig.subplots_adjust(hspace=0.4, wspace=0.0)
	# elif qoi_k_Dflag == 2:
	# 	fig, axs = plt.subplots(2, 1, figsize=(16, 12))
	# 	fig.subplots_adjust(hspace=0.4, wspace=0.0)
	# else:
	# 	fig, axs = plt.subplots(1, 1, figsize=(12, 8))
	
	caption = ""

	# fine_u: ground truth
	# data.quest_cond_v: coarse solution
	# coarse_u: FMint result

	ind_cond = np.argsort(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	if qoi_k_Dflag >= 2:
		fine_u_0 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		fine_u_1 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		FMint_u_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		FMint_u_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		coarse_0 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		coarse_1 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		if qoi_k_Dflag == 3:
			fine_u_2 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]
			FMint_u_2 = (pred[data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]
			coarse_2 = (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 2])[ind_cond]

			gt_points =jnp.array([fine_u_0,fine_u_1,fine_u_2]).T.reshape(-1,1,3)
			segments = jnp.concatenate([gt_points[:-1],gt_points[1:]], axis=1)
			# 使用渐变颜色

			colors = np.linspace(0, 1, len(ind_cond))

			fig = plt.figure(figsize=(12,18))
			# fig = plt.subplots(figsize=(16, 12))
			for j in range(3):
				ax = fig.add_subplot(1, 3, j+1, projection='3d')
				# 添加渐变轨迹
				if j == 0:
					for i in range(len(ind_cond) - 1):
						ax.plot(fine_u_0[i:i+2], fine_u_1[i:i+2], fine_u_2[i:i+2], color=cm.viridis(colors[i]), lw=2)
				elif j == 1:
					for i in range(len(ind_cond) - 1):
						ax.plot(FMint_u_0[i:i+2], FMint_u_1[i:i+2], FMint_u_2[i:i+2], color=cm.viridis(colors[i]), lw=2)
				else:
					for i in range(len(ind_cond) - 1):
						ax.plot(coarse_0[i:i+2], coarse_1[i:i+2], coarse_2[i:i+2], color=cm.viridis(colors[i]), lw=2)

	# if qoi_k_Dflag >= 2:
	# 	axs[0].set_xlabel('time')
	# 	axs[0].set_ylabel('value')
	# 	axs[0].legend(loc = "upper right", fontsize="25")
	# 	axs[0].yaxis.set_major_locator(ticker.LinearLocator(4))
	# 	axs[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
	# 	axs[1].set_xlabel('time')
	# 	axs[1].set_ylabel('value')
	# 	axs[1].legend(loc = "upper right",fontsize="25")
	# 	axs[1].yaxis.set_major_locator(ticker.LinearLocator(4))
	# 	axs[1].yaxis.set_minor_locator(ticker.LinearLocator(8))
	# 	if qoi_k_Dflag == 3:
	# 		axs[2].set_xlabel('time')
	# 		axs[2].set_ylabel('value')
	# 		axs[2].legend(loc = "upper right",fontsize="25")
	# 		axs[2].yaxis.set_major_locator(ticker.LinearLocator(4))
	# 		axs[2].yaxis.set_minor_locator(ticker.LinearLocator(8))
	# 		for item in ([axs[2].xaxis.label, axs[2].yaxis.label] + axs[2].get_xticklabels() + axs[2].get_yticklabels()):
	# 			item.set_fontsize(25)

	# 	for item in ([axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
	# 		item.set_fontsize(25)
	# 	for item in ([axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
	# 		item.set_fontsize(25)
	# else:
	# 	axs.set_xlabel('time'); axs.set_ylabel('value')
	# 	axs.legend(loc = "upper right", fontsize="25")

	# 	axs.yaxis.set_major_locator(ticker.LinearLocator(4))
	# 	axs.yaxis.set_minor_locator(ticker.LinearLocator(8))

	# 	for item in ([axs.xaxis.label, axs.yaxis.label] + axs.get_xticklabels() + axs.get_yticklabels()):
	# 		item.set_fontsize(25)
	
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig
	
def icon_ODE_plot_data(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_Dflag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_Dflag == 3:
		fig, axs = plt.subplots(3, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	elif qoi_k_Dflag == 2:
		fig, axs = plt.subplots(2, 1, figsize=(16, 12))
		fig.subplots_adjust(hspace=0.4, wspace=0.0)
	else:
		fig, axs = plt.subplots(1, 1, figsize=(12, 8))
	
	caption = ""
	# fig.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot pred for both dimensions 
 
	# print("cond time: ",data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	# print("qoi time: ", data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	ind_cond = np.argsort(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	if qoi_k_Dflag >= 2:
		fine_u_0 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]
		fine_u_1 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi]

		coarse_u_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]
		coarse_u_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi]

		axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_0, 'r+', markersize=15, label='fine ode')
		# axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'g+', markersize=15, label='coarse-grained ode')
		axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u_0, '*', markersize=9, label='ICON', alpha = 0.5)

		axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_1, 'r+', markersize=15, label='fine ode')
		# axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond], 'g+', markersize=15, label='coarse-grained ode')
		axs[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u_1, '*', markersize=9, label='ICON', alpha = 0.5)
		if qoi_k_Dflag == 3:
			fine_u_2 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi]
			coarse_u_2 = (pred[data.quest_qoi_mask[0,:].astype(bool), 2])[ind_qoi]

			axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
								fine_u_2, 'r+', markersize=15, label='fine ode')
			# axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
			# 					(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'g+', markersize=15, label='coarse-grained ode')
			axs[2].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
								coarse_u_2, '*', markersize=9, label='ICON', alpha = 0.5)
	else:
		fine_u = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]

		coarse_u = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]

		# print("qoi time: ", (data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]).shape)
		# print("quest cond v shape: ", (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0]).shape)
		# print("label shape: ", (label[0, data.quest_qoi_mask[0,:].astype(bool), 0]).shape)
		# print("fine u: ", fine_u.shape)


		axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u, 'r+', markersize=15, label='fine-grained ode')
		# axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'g+', markersize=15, label='coarse-grained ode')
		axs.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							coarse_u, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)

	if qoi_k_Dflag >= 2:
		axs[0].set_xlabel('time')
		axs[0].set_ylabel('value')
		axs[0].legend(loc = "upper right", fontsize="16")
		axs[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		axs[1].set_xlabel('time')
		axs[1].set_ylabel('value')
		axs[1].legend(loc = "upper right",fontsize="16")
		axs[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		axs[1].yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs[0].xaxis.label, axs[0].yaxis.label] + axs[0].get_xticklabels() + axs[0].get_yticklabels()):
			item.set_fontsize(25)
		for item in ([axs[1].xaxis.label, axs[1].yaxis.label] + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
			item.set_fontsize(25)

		if qoi_k_Dflag == 3:
			axs[2].set_xlabel('time')
			axs[2].set_ylabel('value')
			axs[2].legend(loc = "upper right",fontsize="16")
			axs[2].yaxis.set_major_locator(ticker.LinearLocator(4))
			axs[2].yaxis.set_minor_locator(ticker.LinearLocator(8))

			for item in ([axs[2].xaxis.label, axs[2].yaxis.label] + axs[2].get_xticklabels() + axs[2].get_yticklabels()):
				item.set_fontsize(25)
	else:
		axs.set_xlabel('time'); axs.set_ylabel('value')
		axs.legend(loc = "upper right", fontsize="16")

		axs.yaxis.set_major_locator(ticker.LinearLocator(4))
		axs.yaxis.set_minor_locator(ticker.LinearLocator(8))

		for item in ([axs.xaxis.label, axs.yaxis.label] + axs.get_xticklabels() + axs.get_yticklabels()):
			item.set_fontsize(25)
	
	if to_tfboard:
		return utils.plot_to_image(fig)
	else:  # save to a file
		return fig
	

def plot_keynote(equation, caption, data, label, pred, config, to_tfboard = True):
	'''
	plot all figures in demo and prediction
	@param 
		equation: string
		caption: string
	@return
		the figure
	'''
	plt.close('all')
	k_index, qoi_k_2D_flag = get_plot_k_index(config['k_mode'], equation)
	if qoi_k_2D_flag:
		fig1, axs1 = plt.subplots(2, 1, figsize=(16, 12))
		fig2, axs2 = plt.subplots(2, 1, figsize=(16, 12))
		fig3, axs3 = plt.subplots(2, 1, figsize=(16, 12))
		fig1.patch.set_alpha(0.0)
		fig2.patch.set_alpha(0.0)
		fig3.patch.set_alpha(0.0)
	else:
		fig1, axs1 = plt.subplots(1, 1, figsize=(16, 12))
		fig2, axs2 = plt.subplots(1, 1, figsize=(16, 12))
		fig3, axs3 = plt.subplots(1, 1, figsize=(16, 12))
		fig1.patch.set_alpha(0.0)
		fig2.patch.set_alpha(0.0)
		fig3.patch.set_alpha(0.0)
	fig1.subplots_adjust(hspace=0.2, wspace=0.0)
	caption = ""
	# fig1.suptitle("eqn:{}\ncaption: {}".format(equation, caption))
	# plot cond quest
	
	# cond_quest = prompt[mask_cond_quest, :k_dim+v_dim]  # [cond_len_in_use, k_dim+v_dim]
	ind_cond = np.argsort(data.quest_cond_k[0, data.quest_cond_mask[0,:].astype(bool), 0])
	ind_qoi = np.argsort(data.quest_qoi_k[0, data.quest_cond_mask[0,:].astype(bool), 0])

	if qoi_k_2D_flag:
		fine_u_0 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		fine_u_1 = (label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		# coarse_u_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond]
		# coarse_u_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi] + (data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond]

		pred_0 = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]
		pred_1 = (pred[data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi]

		axs1[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_0, 'o-', color = 'orange', markersize=12, label='fine-grained ode')
		axs1[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'go-', markersize=12, label='coarse-grained ode')
		# axs[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					coarse_u_0, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)

		axs1[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u_1, 'o-', color = 'orange',markersize=12, label='fine-grained ode')
		axs1[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond], 'go-', markersize=12, label='coarse-grained ode')
		# axs1[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					coarse_u_1, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)
  
		axs2[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'go-', markersize=15, label='coarse-grained ode')
		axs2[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 1])[ind_cond], 'go-', markersize=15, label='coarse-grained ode')
		
		axs3[0].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi], 'yo-', markersize=15, label='FG-CG')
		axs3[1].plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(label[0, data.quest_qoi_mask[0,:].astype(bool), 1])[ind_qoi], 'yo-', markersize=15, label='FG-CG')
		
	else:
		fine_u = (label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]

		pred = (pred[data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi]

		axs1.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							fine_u, 'o-', color = 'orange', markersize=12, label='fine-grained ode')
		axs1.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'go-', markersize=12, label='coarse-grained ode')
		# axs1.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
		# 					coarse_u, '*', markersize=9, label='predicted ode -- icon', alpha = 0.5)
  
		axs2.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(data.quest_cond_v[0, data.quest_cond_mask[0,:].astype(bool), 0])[ind_cond], 'go-', markersize=15, label='coarse-grained ode')

		axs3.plot(np.sort(data.quest_qoi_k[0, data.quest_qoi_mask[0,:].astype(bool), k_index]),
							(label[0, data.quest_qoi_mask[0,:].astype(bool), 0])[ind_qoi], 'yo-', markersize=15, label='FG-CG')

	if qoi_k_2D_flag:
		# axs1[0].set_xlabel('time')
		# axs[0].set_ylabel('value')
		axs1[0].legend(loc = "upper right", fontsize="25")
		axs1[0].set_yticklabels([])
		axs1[0].set_xticklabels([])
		# axs1[0].yaxis.set_major_locator(ticker.LinearLocator(4))
		# axs1[0].yaxis.set_minor_locator(ticker.LinearLocator(8))
		# axs[1].set_xlabel('time')
		# axs[1].set_ylabel('value')
		axs1[1].legend(loc = "upper right",fontsize="25")
		axs1[1].set_yticklabels([])
		axs1[1].set_xticklabels([])
		# axs1[1].yaxis.set_major_locator(ticker.LinearLocator(4))
		# axs1[1].yaxis.set_minor_locator(ticker.LinearLocator(8))

		# for item in ([axs1[0].xaxis.label, axs1[0].yaxis.label] + axs1[0].get_xticklabels() + axs1[0].get_yticklabels()):
		# 	item.set_fontsize(25)
		# for item in ([axs1[1].xaxis.label, axs1[1].yaxis.label] + axs1[1].get_xticklabels() + axs1[1].get_yticklabels()):
		# 	item.set_fontsize(25)
	else:
		# axs.set_xlabel('time'); axs.set_ylabel('value')
		axs1.legend(loc = "upper right", fontsize="25")

		axs1.set_yticklabels([])
		axs1.set_xticklabels([])

		# axs1.yaxis.set_major_locator(ticker.LinearLocator(4))
		# axs1.yaxis.set_minor_locator(ticker.LinearLocator(8))

		# for item in ([axs1.xaxis.label, axs1.yaxis.label] + axs1.get_xticklabels() + axs1.get_yticklabels()):
		# 	item.set_fontsize(25)
	
	return fig1, fig2, fig3
	
	

if __name__ == "__main__":
	pass
