import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from modules.utils import bivariate_loss, masked_softmax

class JointRecog(nn.Module):
	def __init__( self, input_dim_x=2, embedding_dim=64, hidden_dim=64, noise_dim=8, num_layers=1, dropout=0.0):

		super(JointRecog, self).__init__()

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.noise_dim = noise_dim

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()

		self.spatial_embedding_x = nn.Linear(input_dim_x, embedding_dim)

		self.encoder_x = nn.LSTM(
			embedding_dim, hidden_dim, num_layers, dropout=dropout,
			batch_first=True
		)

		self.fc_out_x = [nn.Linear(3*hidden_dim + noise_dim, 128),nn.ReLU(),nn.Linear(128, 2*noise_dim)]
		self.fc_out_x = nn.Sequential(*self.fc_out_x)

		self.attn_x_y = nn.ModuleDict({
			'fc_e':nn.Linear( hidden_dim, 64),
			'fc_l':nn.Linear( 2, 64),
			'fc_c':nn.Linear( hidden_dim, 64),
			'fc_final':nn.Linear( 64, 1)
			})

		
		self.attn_z = nn.ModuleDict({
			'fc_e':nn.Linear( hidden_dim, 64),
			'fc_l':nn.Linear( 2, 64),
			'fc_final':nn.Linear( 64, 1)
			})


	def init_hidden(self, batch):
		h = Variable(torch.zeros(self.num_layers, batch, self.hidden_dim), requires_grad=True).cuda()
		c = Variable(torch.zeros(self.num_layers, batch, self.hidden_dim), requires_grad=True).cuda()
		return (h,c)

	def attend_joint_feats(self, x_enc_feats, x_last_feats, curr_x_enc ):
		attn_x_enc = self.attn_x_y['fc_e'](x_enc_feats)
		attn_x_last = self.attn_x_y['fc_l'](x_last_feats)
		attn_x_curr = self.attn_x_y['fc_c'](curr_x_enc).unsqueeze(1).repeat(1,attn_x_enc.size(1),1)
		
		attn_out = self.attn_x_y['fc_final']( self.tanh(attn_x_enc + attn_x_last + attn_x_curr ))
		attn_out = attn_out.squeeze(-1)
		
		attn_mask = torch.sum(torch.abs(x_enc_feats), dim=2) > 0
		alpha = masked_softmax(attn_out, attn_mask)

		attn_feat = (x_enc_feats * alpha.unsqueeze(2)).sum(dim=1)

		return attn_feat

	def get_joint_feats(self, x_enc, x_last, seq_start_end):
		social_feats = []
		for i, (st, ed) in  enumerate(seq_start_end):
			x_enc_curr = x_enc[st:ed]
			x_last_curr = x_last[st:ed]
			num_ped = ed - st
			_x_enc = x_enc_curr.repeat(num_ped,1)
			_x_enc = _x_enc.view(num_ped,num_ped,-1)

			_x_last = x_last_curr.repeat(num_ped,1)
			_x_last = _x_last.view(num_ped,num_ped,-1)
			_x_last = _x_last - x_last_curr[:,None,:]

			_social_feat = self.attend_joint_feats(_x_enc,_x_last,x_enc_curr)
			social_feats.append(_social_feat)
		return torch.cat(social_feats,dim=0)

	def attend_prev_z(self, zs, x_enc_feats, x_last_feats ):
		attn_x_enc = self.attn_z['fc_e'](x_enc_feats)
		attn_x_last = self.attn_z['fc_l'](x_last_feats)

		attn_out = self.attn_z['fc_final']( self.tanh(attn_x_enc + attn_x_last))# + attn_zs 
		attn_out = attn_out.squeeze(-1)
		out_feats = torch.cat([x_enc_feats, zs], dim=2)

		attn_mask = torch.sum(torch.abs(out_feats), dim=2) > 0 
		alpha = masked_softmax(attn_out, attn_mask)

		attn_feat = (out_feats * alpha.unsqueeze(2)).sum(dim=1)
		return attn_feat

	def pad_seq(self, x_seq, seq_start_end ):
		#print('x_seq pad before -- ',x_seq.size())
		max_seq_len = 0
		for st, ed in seq_start_end:
			if int(ed - st) > max_seq_len:
				max_seq_len = int(ed - st)
		pad_seq_start_end = []
		amount_pad = []#0
		new_seq = []
		for st, ed in seq_start_end:
			if ed - st < max_seq_len:
				_pad = torch.zeros( max_seq_len - (ed - st), x_seq.size(1) ).cuda()
				new_seq.append( torch.cat([x_seq[st:ed], _pad], dim=0)[None,:,:] )
				amount_pad.append(ed-st)
			else:
				new_seq.append( x_seq[st:ed][None,:,:] )
				amount_pad.append(ed-st)

		new_seq = torch.cat(new_seq, dim=0)
		return new_seq, amount_pad

	def unpad_seq(self, x_seq, pad_seq_start_end):
		x_seq = torch.cat([x_seq[i,:_len] for i, _len in enumerate(pad_seq_start_end) ], dim=0)
		return x_seq



	def joint_sample_batched(self, x_y_full_enc, x_y_full_attended, x_last, seq_start_end):
		means, logs, zs = [], [], []

		x_y_full_enc_pad, pad_seq_start_end = self.pad_seq( x_y_full_enc, seq_start_end )
		x_y_full_attended_pad, _ = self.pad_seq( x_y_full_attended, seq_start_end)
		x_last_pad, _ = self.pad_seq( x_last, seq_start_end )
		

		z_prev_init = torch.zeros(x_y_full_enc_pad.size(0),self.hidden_dim+self.noise_dim).cuda()

		for j in range(x_y_full_enc_pad.size(1)):
			if j > 0:
				_zs_temp = torch.cat(zs, dim=1)
				_x_last_temp = x_last_pad[:,0:j] - x_last_pad[:,j:j+1]
				z_prev_attn = self.attend_prev_z( _zs_temp[:,0:], x_y_full_enc_pad[:,0:j], _x_last_temp )
			else:
				z_prev_attn = z_prev_init

			
			full_enc = torch.cat([x_y_full_enc_pad[:,j], x_y_full_attended_pad[:,j], z_prev_attn], dim=1)
			outs = self.fc_out_x(full_enc)
			mean, log = outs[:,:outs.size(1)//2], outs[:,outs.size(1)//2:]
			z = (torch.randn(mean.size()).cuda())*torch.exp(log*0.5) + mean
			means.append(mean[:,None,:])
			logs.append(log[:,None,:])
			zs.append(z[:,None,:])


		means = self.unpad_seq(torch.cat(means, dim=1), pad_seq_start_end)
		logs = self.unpad_seq(torch.cat(logs, dim=1), pad_seq_start_end)
		zs = self.unpad_seq(torch.cat(zs, dim=1), pad_seq_start_end)

		return means, logs, zs


	def forward(self, x_y_full, x_last, seq_start_end ):
		x_y_full_enc = self.spatial_embedding_x(x_y_full)
		hiddens = self.init_hidden(x_y_full.size(0))
		_, hiddens_x = self.encoder_x(x_y_full_enc, hiddens)
		x_y_full_enc = hiddens_x[0][0]

		x_y_full_attended = self.get_joint_feats( x_y_full_enc, x_last, seq_start_end)

		means_x, logs_x, z_x = self.joint_sample_batched( x_y_full_enc, x_y_full_attended, x_last, seq_start_end)#

		return means_x, logs_x, z_x



class JointPrior(nn.Module):
	def __init__( self, input_dim_x=2, embedding_dim=64, hidden_dim=64, noise_dim=8, num_layers=1, dropout=0.0):

		super(JointPrior, self).__init__()

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.noise_dim = noise_dim

		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()

		self.spatial_embedding_x = nn.Linear(input_dim_x, embedding_dim)

		self.encoder_x = nn.LSTM(
			embedding_dim, hidden_dim, num_layers, dropout=dropout,
			batch_first=True
		)

		#self.fc_out_x = nn.Linear(hidden_dim + noise_dim, 2*noise_dim)

		#print('3*hidden_dim + noise_dim ',3*hidden_dim + noise_dim)
		self.fc_out_x = [nn.Linear(3*hidden_dim + noise_dim, 128),nn.ReLU(),nn.Linear(128, 2*noise_dim)]
		self.fc_out_x = nn.Sequential(*self.fc_out_x)

		self.attn_x = nn.ModuleDict({
			'fc_e':nn.Linear( hidden_dim, 64),
			'fc_l':nn.Linear( 2, 64),
			'fc_c':nn.Linear( hidden_dim, 64),
			'fc_final':nn.Linear( 64, 1)
			})

		
		self.attn_z = nn.ModuleDict({
			'fc_e':nn.Linear( hidden_dim, 64),
			'fc_l':nn.Linear( 2, 64),
			'fc_final':nn.Linear( 64, 1)
			})

	def init_hidden(self, batch):
		h = Variable(torch.zeros(self.num_layers, batch, self.hidden_dim), requires_grad=True).cuda()
		c = Variable(torch.zeros(self.num_layers, batch, self.hidden_dim), requires_grad=True).cuda()
		return (h,c)

	def attend_joint_feats(self, x_enc_feats, x_last_feats, curr_x_enc ):
		attn_x_enc = self.attn_x['fc_e'](x_enc_feats)
		attn_x_last = self.attn_x['fc_l'](x_last_feats)
		attn_x_curr = self.attn_x['fc_c'](curr_x_enc).unsqueeze(1).repeat(1,attn_x_enc.size(1),1)
		
		attn_out = self.attn_x['fc_final']( self.tanh(attn_x_enc + attn_x_last + attn_x_curr ))
		attn_out = attn_out.squeeze(-1)
		
		attn_mask = torch.sum(torch.abs(x_enc_feats), dim=2) > 0
		alpha = masked_softmax(attn_out, attn_mask)

		attn_feat = (x_enc_feats * alpha.unsqueeze(2)).sum(dim=1)

		return attn_feat

	def get_joint_feats(self, x_enc, x_last, seq_start_end):
		social_feats = []
		for i, (st, ed) in  enumerate(seq_start_end):
			x_enc_curr = x_enc[st:ed]
			x_last_curr = x_last[st:ed]
			num_ped = ed - st
			_x_enc = x_enc_curr.repeat(num_ped,1)
			_x_enc = _x_enc.view(num_ped,num_ped,-1)

			_x_last = x_last_curr.repeat(num_ped,1)
			_x_last = _x_last.view(num_ped,num_ped,-1)
			_x_last = _x_last - x_last_curr[:,None,:]

			_social_feat = self.attend_joint_feats(_x_enc,_x_last,x_enc_curr)
			social_feats.append(_social_feat)
		return torch.cat(social_feats,dim=0)

	def attend_prev_z(self, zs, x_enc_feats, x_last_feats ):
		attn_x_enc = self.attn_z['fc_e'](x_enc_feats)
		attn_x_last = self.attn_z['fc_l'](x_last_feats)

		attn_out = self.attn_z['fc_final']( self.tanh(attn_x_enc + attn_x_last))# + attn_zs 
		attn_out = attn_out.squeeze(-1)
		out_feats = torch.cat([x_enc_feats, zs], dim=2)

		attn_mask = torch.sum(torch.abs(out_feats), dim=2) > 0 
		alpha = masked_softmax(attn_out, attn_mask)

		attn_feat = (out_feats * alpha.unsqueeze(2)).sum(dim=1)
		return attn_feat


	def pad_seq(self, x_seq, seq_start_end ):
		max_seq_len = 0
		for st, ed in seq_start_end:
			if int(ed - st) > max_seq_len:
				max_seq_len = int(ed - st)
		pad_seq_start_end = []
		amount_pad = []#0
		new_seq = []
		for st, ed in seq_start_end:
			if ed - st < max_seq_len:
				_pad = torch.zeros( max_seq_len - (ed - st), x_seq.size(1) ).cuda()
				new_seq.append( torch.cat([x_seq[st:ed], _pad], dim=0)[None,:,:] )
				amount_pad.append(ed-st)
			else:
				new_seq.append( x_seq[st:ed][None,:,:] )
				amount_pad.append(ed-st)

		new_seq = torch.cat(new_seq, dim=0)
		return new_seq, amount_pad

	def unpad_seq(self, x_seq, pad_seq_start_end):
		x_seq = torch.cat([x_seq[i,:_len] for i, _len in enumerate(pad_seq_start_end) ], dim=0)
		return x_seq



	def joint_sample_batched(self, x_enc, x_enc_social, x_last, seq_start_end):
		means, logs, zs = [], [], []

		x_enc_pad, pad_seq_start_end = self.pad_seq( x_enc, seq_start_end )
		x_enc_social_pad, _ = self.pad_seq( x_enc_social, seq_start_end)
		x_last_pad, _ = self.pad_seq( x_last, seq_start_end )
		

		z_prev_init = torch.zeros(x_enc_pad.size(0),self.hidden_dim+self.noise_dim).cuda()
		
		for j in range(x_enc_pad.size(1)):
			if j > 0:
				_zs_temp = torch.cat(zs, dim=1)
				_x_last_temp = x_last_pad[:,0:j] - x_last_pad[:,j:j+1]
				z_prev_attn = self.attend_prev_z( _zs_temp[:,0:], x_enc_pad[:,0:j], _x_last_temp )
			else:
				z_prev_attn = z_prev_init

			
			full_enc = torch.cat([x_enc_pad[:,j], x_enc_social_pad[:,j], z_prev_attn], dim=1)
			outs = self.fc_out_x(full_enc)
			mean, log = outs[:,:outs.size(1)//2], outs[:,outs.size(1)//2:]
			z = (torch.randn(mean.size()).cuda())*torch.exp(log*0.5) + mean
			means.append(mean[:,None,:])
			logs.append(log[:,None,:])
			zs.append(z[:,None,:])

		means = self.unpad_seq(torch.cat(means, dim=1), pad_seq_start_end)
		logs = self.unpad_seq(torch.cat(logs, dim=1), pad_seq_start_end)
		zs = self.unpad_seq(torch.cat(zs, dim=1), pad_seq_start_end)

		return means, logs, zs

		

	def sample_x(self, x_enc, seq_start_end):
		outs = self.fc_out_x(x_enc)
		means, logs = outs[:,:outs.size(1)//2], outs[:,outs.size(1)//2:]
		z = (torch.randn(means.size()).cuda())*torch.exp(logs*0.5) + means
		return means, logs, z

	def forward(self, x_enc, x_last, seq_start_end ):
		x_enc_social = self.get_joint_feats( x_enc, x_last, seq_start_end)
		means_x, logs_x, z_x = self.joint_sample_batched( x_enc, x_enc_social, x_last, seq_start_end)
		return means_x, logs_x, z_x

class JointBetaCVAE(nn.Module):
	def __init__( self, input_dim_x=2, output_dim_y=5, pool=False, embedding_dim=16, hidden_dim=32, 
		noise_dim=8, num_layers=1, classes=False, dropout=0.0):

		super(JointBetaCVAE, self).__init__()
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.noise_dim = noise_dim

		self.spatial_embedding_x_enc = nn.Linear(input_dim_x, embedding_dim)
		if classes:
			self.class_embedding = nn.Linear(1, embedding_dim)
			self.final_embedding = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
		
		self.encoder_x = nn.LSTM(
			embedding_dim, hidden_dim, num_layers, dropout=dropout,
			batch_first=True
		)


		self.decoder_y = nn.LSTM(
			hidden_dim + noise_dim, hidden_dim, num_layers, dropout=dropout,
			batch_first=True
		)


		self.recog = JointRecog(input_dim_x=2, embedding_dim=embedding_dim, hidden_dim=hidden_dim, noise_dim=noise_dim)

		self.prior = JointPrior(input_dim_x=input_dim_x, embedding_dim=embedding_dim, hidden_dim=hidden_dim, noise_dim=noise_dim)

		self.fc_out_y = nn.Linear(hidden_dim, output_dim_y)

	def init_hidden(self, batch):
		h = Variable(torch.zeros(self.num_layers, batch, self.hidden_dim), requires_grad=True).cuda()
		c = Variable(torch.zeros(self.num_layers, batch, self.hidden_dim), requires_grad=True).cuda()
		return (h,c)

	def encode_x(self, x, classes, seq_start_end):
		#print('x ',x.size())
		x_enc = self.spatial_embedding_x_enc(x)
		if classes is not None:
			x_cls = self.class_embedding(classes)

		hiddens = self.init_hidden(x.size(0))
		_, hiddens_x = self.encoder_x(x_enc, hiddens)
		x_enc = hiddens_x[0][0]
		if classes is not None:
			x_enc = self.final_embedding(torch.cat([x_enc,x_cls], dim=1))
		return x_enc

	

	def decode_y(self, x_enc, z_x, seq_start_end, timesteps):
		dec_in = torch.cat([x_enc,z_x], dim=1)
		dec_in = dec_in[:,None,:].repeat(1,timesteps,1)
		hiddens = self.init_hidden(dec_in.size(0))
		outputs, _ = self.decoder_y(dec_in, hiddens)
		y_out = self.fc_out_y(outputs)
		return y_out


	def forward(self, x, y, x_last, loss_mask, classes, seq_start_end, timesteps, train=True):
		if train:

			x_in = torch.cat([x[:,i,:,:] for i in range(0,x.size(1))], dim=2)

			x_enc = self.encode_x(x_in, classes, seq_start_end)


			p_means_x, p_logs_x, _ = self.prior(x_enc, x_last, seq_start_end )
			x_full = torch.cat([x[:,1,:,:], y], dim=1)
			q_means_x, q_logs_x, q_z_x = self.recog( x_full, x_last, seq_start_end)
			y_hat = self.decode_y(x_enc, q_z_x, seq_start_end, timesteps)

			kl_loss = 0.5*(p_logs_x - q_logs_x) + (q_logs_x.exp() +(q_means_x - p_means_x)**2 )/(2*p_logs_x.exp()) - 0.5
			kl_loss = torch.sum(kl_loss, axis=1)

			nll_loss = bivariate_loss(y_hat, y)

			return nll_loss, kl_loss, y_hat

		else:
			with torch.no_grad():
				x_in = torch.cat([x[:,i,:,:] for i in range(0,x.size(1))], dim=2)
				x_enc = self.encode_x(x_in, classes, seq_start_end)
				_, _, p_z_x = self.prior(x_enc, x_last, seq_start_end )
				y_hat = self.decode_y(x_enc, p_z_x, seq_start_end, timesteps)

				return y_hat



