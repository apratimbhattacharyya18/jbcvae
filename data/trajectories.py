#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import logging
import sys
import os
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from modules.utils import derivative_of

logger = logging.getLogger(__name__)

CLASS_STR_INT = {'ego':0,'ped':1,'bike':2}

def seq_collate(data):
	(obs_seq_list, pred_seq_list, classes_list, loss_mask_list) = zip(*data)
	#obs_seq_rel_list, pred_seq_rel_list,

	_len = [len(seq) for seq in obs_seq_list]
	cum_start_idx = [0] + np.cumsum(_len).tolist()
	seq_start_end = [[start, end]
					 for start, end in zip(cum_start_idx, cum_start_idx[1:])]

	obs_traj = torch.cat(obs_seq_list, dim=0)
	pred_traj = torch.cat(pred_seq_list, dim=0)
	loss_mask = torch.cat(loss_mask_list, dim=0)
	classes = torch.cat(classes_list, dim=0)[:,None]
	seq_start_end = torch.LongTensor(seq_start_end)
	out = [
		obs_traj, pred_traj, classes, loss_mask, seq_start_end
	]

	return tuple(out)

class TrajectoryDataset(Dataset):

	def __init__(
		self, data_root = None, obs_len=5, pred_len=30, min_hist=5, min_future=30, train=True, include_ego=True, skip=1
	):

		def diff_seq( _track ):
			diff_x = derivative_of( _track[:,0] )[:,None]
			diff_y = derivative_of( _track[:,1] )[:,None]
			return np.concatenate([diff_x,diff_y], axis=1)

		self.obs_len = obs_len
		self.pred_len = pred_len
		self.min_hist = min_hist
		self.min_future = min_future
		self.min_len = min_hist + min_future
		self.seq_len = self.obs_len + self.pred_len
		self.min_ped = 1 + include_ego
		self.skip = skip

		num_peds_in_seq = []
		seq_list = []# [[#peds,traj_len,2],[],[]]
		seq_list_rel = []
		loss_mask_list = []
		seq_ts_list = []
		class_list = []
		
		if train:
			with open(os.path.join(data_root, 'train.json'), 'r') as f:
				full_data = json.load(f)
		else:
			with open(os.path.join(data_root, 'valtest.json'), 'r') as f:
				full_data = json.load(f)

		ts_anno_dict = full_data['ts_anno_dict']
		seq_ts_dict = full_data['seq_ts_dict']

		for seq_idx, ts_list in (seq_ts_dict.items()):


			val_list = [150, 111, 22, 4, 248, 70, 157, 133, 93, 28, 113, 148, 135, 9, 73, 196, 178, 12, 229, 
			184, 192, 32, 206, 67, 119, 6, 78, 291, 227, 80, 281, 34, 172, 160, 168, 224, 117, 36, 239, 292, 
			54, 63, 107, 252, 276, 33, 100, 163, 176, 56, 127, 20, 290, 79, 156, 86, 277, 286, 165, 1, 237, 
			142, 260, 161, 243, 189, 42, 8, 105, 74, 121, 197, 21, 44, 16, 98, 116, 53, 95, 
			45, 149, 233, 94, 202, 273, 222, 3, 138, 147, 61, 288, 283, 99, 103, 241, 75, 250, 220, 124, 31]

			if not train and int(seq_idx) in val_list:
				continue


			for ts_idx in range(0,len(ts_list) - self.seq_len,1):
				common_insts = []
				for curr_ts_idx in range(ts_idx + (self.obs_len - self.min_hist),ts_idx+self.obs_len+self.min_future+1):
					curr_ts = ts_list[curr_ts_idx]
					#print('ts_anno_dict[curr_ts] ',type(ts_anno_dict[curr_ts]['annotations']))
					#print([anno for anno in ts_anno_dict[curr_ts]['annotations']])
					common_insts.append(set([anno['inst_id'] for anno in ts_anno_dict[curr_ts]['annotations']]))
				common_insts = set.intersection(*common_insts)
				#print(common_insts)
				if not include_ego:
					common_insts.remove(0)
				#print('before ', common_insts)
				if common_insts is not None:
					common_insts = list(common_insts)
				else:
					continue
				#print('common_insts ',common_insts)
				
				if len(common_insts) > (0 + include_ego):
					common_insts = np.sort(np.array(common_insts)).tolist()
					#print('sorted ', common_insts)
					all_tracks = []
					#all_tracks_rel = []
					all_loss_masks = []
					all_ts_list =[] 
					all_class_list = []
					for ped in common_insts:
						track = []
						curr_loss_mask = []
						curr_ts_list = []
						curr_class = None
						for curr_ts_idx in range(ts_idx,ts_idx+self.seq_len,self.skip):
							curr_ts = ts_list[curr_ts_idx]
							curr_ts_present = False
							for anno in ts_anno_dict[curr_ts]['annotations']:
								if anno['inst_id'] == ped:
									track.append(anno['pos'][:2])
									curr_ts_present = True
									curr_loss_mask.append(1.)
									curr_ts_list.append(int(curr_ts))
									if curr_class is None:
										curr_class = CLASS_STR_INT[anno['class']]
									break
							if not curr_ts_present:
								track.append(np.array([0,0]))
								curr_loss_mask.append(0.)
								curr_ts_list.append(0.)

						track = np.array(track)
						curr_loss_mask = np.array(curr_loss_mask)

						track_r = track.copy()
						track_r[:,:] = track_r[:,:] - track_r[0,:][None,:]

						track_v = diff_seq(track)
						track_a = diff_seq(track_v)
						track_full = np.concatenate([track[None,:,:],track_v[None,:,:],track_a[None,:,:]], axis=0)#track_r[None,:,:],


						all_tracks.append(track_full)
						all_loss_masks.append(curr_loss_mask)
						all_ts_list.append(np.array(curr_ts_list))
						all_class_list.append(curr_class)

					num_peds_in_seq.append(len(common_insts))
					seq_list.append(all_tracks)
					loss_mask_list.append(all_loss_masks)
					seq_ts_list.append(all_ts_list)
					class_list.append(all_class_list)

		
		self.num_seq = len(seq_list)
		seq_list = np.concatenate(seq_list, axis=0)
		loss_mask_list = np.concatenate(loss_mask_list, axis=0)
		seq_ts_list = np.concatenate(seq_ts_list, axis=0)
		class_list = np.concatenate(class_list, axis=0)

		
		self.obs_traj = torch.from_numpy(
			seq_list[:, :, :self.obs_len, :]).type(torch.float)
		#if tot_len > obs_len:
		self.pred_traj = torch.from_numpy(
			seq_list[:, :, self.obs_len:, :]).type(torch.float)

		self.seq_ts = seq_ts_list
		self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
		self.classes = torch.from_numpy(class_list).type(torch.float)

		cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
		self.seq_start_end = [
			(start, end)
			for start, end in zip(cum_start_idx, cum_start_idx[1:])
		]


	def __len__(self):
		return self.num_seq


	def __getitem__(self, index):
		start, end = self.seq_start_end[index]
		out = [
			self.obs_traj[start:end, :], self.pred_traj[start:end, :],
			self.classes[start:end], self.loss_mask[start:end, :]
		]
		return out

