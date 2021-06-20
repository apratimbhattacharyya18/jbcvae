from torch.utils.data import DataLoader

def data_loader_europvi( data_root, batch_size, train, loader_num_workers):#args, path, split_file_path, split, train
	from data.trajectories import TrajectoryDataset, seq_collate
	dset = TrajectoryDataset( data_root=data_root, train=train )
	loader = DataLoader(
		dset, drop_last=False,
		batch_size=batch_size,
		shuffle=True,
		num_workers=loader_num_workers,
		collate_fn=seq_collate)
	return dset, loader


