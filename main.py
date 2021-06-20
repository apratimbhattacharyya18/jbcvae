from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.multivariate_normal as torchdist

import itertools
import os
import argparse
import sys
from tqdm import tqdm
import numpy as np

from modules.model import JointBetaCVAE
from modules.utils import relative_to_abs

from data.loader import data_loader_europvi as data_loader

torch.multiprocessing.set_sharing_strategy('file_system')

batch_size = 64
test_samples = 20

def evaluate_europvi( loader, generator, eval_steps=[9,19,29], num_samples=20):
	ade_outer, fde_outer = [], []
	total_traj = 0
	

	all_gts_ego = []
	all_preds_ego = []
	all_gts_ped = []
	all_preds_ped = []
	all_closest_approach = []
	all_loss_masks = []

	print('Evaluating ... ')
	with torch.no_grad():
		for batch in tqdm(loader):
			(obs_traj, pred_traj_gt, classes, loss_mask, seq_start_end) = batch
			total_traj += pred_traj_gt.size(0)
			curr_preds = []
			curr_preds_full = []

			for s_idx in range(num_samples):

				pred_traj_fake_rel =  generator(x=obs_traj[:,1:,:,:].cuda(), y=None, 
					x_last=obs_traj[:,0,-1,:].cuda(), loss_mask=loss_mask.cuda(), 
					classes=classes.cuda(), seq_start_end=seq_start_end.cuda(), 
					timesteps=pred_traj_gt.size(2), train=False)

				
				sx = torch.exp(pred_traj_fake_rel[:,:,2]) #sx
				sy = torch.exp(pred_traj_fake_rel[:,:,3]) #sy
				corr = torch.tanh(pred_traj_fake_rel[:,:,4]) #corr
				
				cov = torch.zeros(pred_traj_fake_rel.shape[0],pred_traj_fake_rel.shape[1],2,2).cuda()
				cov[:,:,0,0]= sx*sx
				cov[:,:,0,1]= corr*sx*sy
				cov[:,:,1,0]= corr*sx*sy
				cov[:,:,1,1]= sy*sy
				mean = pred_traj_fake_rel[:,:,0:2]
				
				mvnormal = torchdist.MultivariateNormal(mean,cov)

				pred_traj_fake_rel = mvnormal.sample()
				
				pred_traj_fake = relative_to_abs(
					pred_traj_fake_rel.cpu(),  obs_traj[:,0,-1,:]
				)

				_obs_traj = obs_traj[:,0,:,:].detach().cpu().numpy()
				pred_traj_fake = pred_traj_fake.detach().cpu().numpy()
				
				curr_preds.append(pred_traj_fake[:,None,:,:])
				curr_preds_full.append( np.concatenate([_obs_traj,pred_traj_fake], axis=1)[:,None,:,:] )
				

			curr_preds = np.concatenate(curr_preds, axis=1)
			curr_preds_full = np.concatenate(curr_preds_full, axis=1)
			pred_traj_gt = pred_traj_gt.detach().cpu().numpy()
			
			_gts_ped = []
			_preds_ped = []

			for st, ed in seq_start_end:
				_gts_ped.append(pred_traj_gt[st+1:ed,0])
				_preds_ped.append(curr_preds[st+1:ed])

			all_gts_ped.append(np.concatenate(_gts_ped, axis=0))
			all_preds_ped.append(np.concatenate(_preds_ped, axis=0))

		results_dict = {}
		all_gts_ped = np.concatenate(all_gts_ped, axis=0)
		all_preds_ped = np.concatenate(all_preds_ped, axis=0)

		performance_p = np.square(all_gts_ped[:,None,:,:] - all_preds_ped)
		

		results_dict['error_1_sec'] = np.mean(np.sort(np.sqrt(performance_p[:, :, eval_steps[0], :].sum(axis=2)), axis=1)[:,:1])
		results_dict['error_2_sec'] = np.mean(np.sort(np.sqrt(performance_p[:, :, eval_steps[1], :].sum(axis=2)), axis=1)[:,:1])
		results_dict['error_3_sec'] = np.mean(np.sort(np.sqrt(performance_p[:, :, eval_steps[2], :].sum(axis=2)), axis=1)[:,:1])


		return results_dict


def update_best_scores(results_dict, best_results):
	for k, v in results_dict.items():
		best_results[k] = min(best_results[k],results_dict[k])
	return best_results


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root', default='../europvi/traj_data/', type=str, help='Path to location of the dataset.')
	parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
	parser.add_argument('--epochs', default=200, type=int, help='Training epochs.')
	parser.add_argument('--test_epoch_interval', default=10, type=int, help='Test epoch interval.')
	parser.add_argument('--from_checkpoint', action='store_true', help='Evaluate Checkpoint.')
	parser.add_argument('--checkpoint_path', default='./ckpts/jbeta_cvae_europvi.pt', type=str, help='Path to location of the checkpoint.')
	args = parser.parse_args()
	data_root = args.data_root
	batch_size = args.batch_size
	epochs = args.epochs
	test_epoch_interval = args.test_epoch_interval
	from_checkpoint = args.from_checkpoint
	checkpoint_path = args.checkpoint_path

	print('Loading data ... ')
	dataset_train, loader_train = data_loader( data_root=data_root, batch_size=batch_size, train=True, loader_num_workers=4 )
	dataset_test, loader_test = data_loader( data_root=data_root, batch_size=batch_size, train=False, loader_num_workers=4 )
	print('Done.')

	jbeta_cvae = JointBetaCVAE( input_dim_x=4, output_dim_y=5, embedding_dim=32, hidden_dim=128, noise_dim=32,
		num_layers=1, classes=True ).cuda()

	optimizer_gen = optim.Adam( jbeta_cvae.parameters(),lr=0.003)
	sched = optim.lr_scheduler.ExponentialLR(optimizer_gen,gamma=0.9999)

	best_results = {}
	result_dict_keys = ['error_1_sec','error_2_sec','error_3_sec']

	for k in result_dict_keys:
		best_results[k] = sys.float_info.max

	if not from_checkpoint:

		for epoch in range(1,epochs+1):
			train_bar = tqdm(loader_train)
			for data in train_bar:
				optimizer_gen.zero_grad()
				obj_traj = data[0].float().cuda()
				target_traj = data[1].float().cuda()
				classes = data[2].float().cuda()
				loss_mark = data[3]
				seq_start_end = data[-1]

				
				nll_loss, kl_loss, _ = jbeta_cvae(x=obj_traj[:,1:,:,:], y=target_traj[:,1,:,:],
					x_last=obj_traj[:,0,-1,:], loss_mask=loss_mark, classes=classes,
					seq_start_end=seq_start_end, timesteps=target_traj.size(2), train=True)

				loss = nll_loss + 0.1*kl_loss# \beta = 0.1
				loss = torch.mean(loss)
				loss.backward()
				optimizer_gen.step()
				sched.step()
				
				train_bar.set_description('Train ELBO = %.2f | Epoch %d -- Iteration ' % (loss.item(),epoch))

			#torch.save(jbeta_cvae, checkpoint_path)
			if epoch % test_epoch_interval == 0 or epoch == 1:
				results_dict = evaluate_europvi( loader_test, jbeta_cvae, num_samples=20)
				best_results = update_best_scores( results_dict, best_results)
				print('Best results:  ', best_results)

	else:
		try:
			jbeta_cvae = torch.load( checkpoint_path )
			print('Checkpoint loaded!')
		except Exception:
			print('Error loading checkpoint!')
			sys.exit(0)	
		
		print('Evaluating model on checkpoint .... ')
		results_dict = evaluate_europvi( loader_test, jbeta_cvae, num_samples=20)
		print('Best results:  ', results_dict)
