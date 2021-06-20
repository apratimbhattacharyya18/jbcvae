from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np

def tiny_value_of_dtype(dtype: torch.dtype):
	"""
	Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
	issues such as division by zero.
	This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
	Only supports floating point dtypes.
	"""
	if not dtype.is_floating_point:
		raise TypeError("Only supports floating point dtypes.")
	if dtype == torch.float or dtype == torch.double:
		return 1e-13
	elif dtype == torch.half:
		return 1e-4
	else:
		raise TypeError("Does not support dtype " + str(dtype))

def masked_softmax(
	vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1, memory_efficient: bool = False,
) -> torch.Tensor:
	if mask is None:
		result = torch.nn.functional.softmax(vector, dim=dim)
	else:
		while mask.dim() < vector.dim():
			mask = mask.unsqueeze(1)
		if not memory_efficient:
			# To limit numerical errors from large vector elements outside the mask, we zero these out.
			result = torch.nn.functional.softmax(vector * mask.float(), dim=dim)
			result = result * mask.float()
			result = result / (
				result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
			)
		else:
			masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
			result = torch.nn.functional.softmax(masked_vector, dim=dim)
	return result

def relative_to_abs(rel_traj, base_point):
	base_point = base_point[:,None,:]
	rel_traj = np.cumsum(rel_traj, axis=1)
	abs_traj = base_point + rel_traj
	return abs_traj

def bivariate_loss(V_pred,V_trgt,mean=True):
	normx = V_trgt[:,:,0]- V_pred[:,:,0]
	normy = V_trgt[:,:,1]- V_pred[:,:,1]

	sx = torch.exp(V_pred[:,:,2]) #sx
	sy = torch.exp(V_pred[:,:,3]) #sy
	corr = torch.tanh(V_pred[:,:,4]) #corr
	
	sxsy = sx * sy

	z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
	negRho = 1 - corr**2

	# Numerator
	result = torch.exp(-z/(2*negRho))
	# Normalization factor
	denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

	# Final PDF calculation
	result = result / denom

	# Numerical stability
	epsilon = 1e-20

	result = -torch.log(torch.clamp(result, min=epsilon))
	#print('bivariate_loss ',result.size())
	#sys.exit(0)
	if mean:
		result = torch.mean(result)#/float(np.log(2.))
	
	return result


def make_continuous_copy(alpha):
	alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
	continuous_x = np.zeros_like(alpha)
	continuous_x[0] = alpha[0]
	for i in range(1, len(alpha)):
		if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
			continuous_x[i] = continuous_x[i - 1] + (
					alpha[i] - alpha[i - 1]) - np.sign(
				(alpha[i] - alpha[i - 1])) * 2 * np.pi
		else:
			continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

	return continuous_x


def derivative_of(x, dt=1, radian=False):
	if radian:
		x = make_continuous_copy(x)

	if x[~np.isnan(x)].shape[-1] < 2:
		return np.zeros_like(x)

	dx = np.full_like(x, np.nan)
	dx[~np.isnan(x)] = np.gradient(x[~np.isnan(x)], dt)

	return dx
