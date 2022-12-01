import copy
import collections
from tqdm import tqdm

import torch

from training import evaluate_model
from cka import CudaCKA

def weighted_averaging(w_list, num_sample_list):
	num_total_samples = sum(num_sample_list)
	keys = w_list[0].keys()
	w_avg = collections.OrderedDict()

	device = w_list[0][list(keys)[0]].device
	
	for k in keys:
		w_avg[k] = torch.zeros(w_list[0][k].size()).to(device)   # Reshape w_avg to match local weights.

	for k in keys:
		for i in range(len(w_list)):
			w_avg[k] += num_sample_list[i] * w_list[i][k]
		w_avg[k] = torch.div(w_avg[k], num_total_samples)
	return w_avg


# Local training.
def local_update_fedavg(glob_model, client_loader, num_local_epochs, optim, optim_args):

	# Global model.
	global_model = copy.deepcopy(glob_model)
	for param in global_model.parameters():
		param.requires_grad = False

	# Evaluating global model on local data.
	train_loss, train_acc = evaluate_model(global_model, client_loader, tqdm_desc='local_train_loss')

	# Local model.
	local_model = copy.deepcopy(glob_model)

	# Params.
	device = next(glob_model.parameters()).device
	if optim == 'sgd':
		optimizer = torch.optim.SGD(local_model.parameters(), **optim_args)
	elif optim == 'adam':
		optimizer = torch.optim.Adam(local_model.parameters(), **optim_args)
	grad_scaler = torch.cuda.amp.GradScaler()
	loss_ce = torch.nn.CrossEntropyLoss()

	# Training.
	local_model.train()
	for epoch in range(num_local_epochs):
		for (x, y) in tqdm(client_loader, desc='epoch {}/{}'.format(epoch + 1, num_local_epochs)):
			optimizer.zero_grad()
			x = x.to(device)
			y = y.to(device)

			# Calculate loss.
			y_pred, _ = local_model(x)
			loss = loss_ce(y_pred, y)
			
			grad_scaler.scale(loss).backward()
			grad_scaler.step(optimizer)
			grad_scaler.update()
	
	# Return update.
	local_update_dict ={
		'local_w': local_model.state_dict(),
		'num_samples': len(client_loader.dataset),
		'train_loss': train_loss,
		'train_acc': train_acc,
	}
	
	return local_update_dict


# Local update with based on model contrastive learning.
def local_update_moon(glob_model, prev_local_w, client_loader, num_local_epochs, optim, optim_args, moon_args):

	# Global model.
	global_model = copy.deepcopy(glob_model)
	for param in global_model.parameters():
		param.requires_grad = False

	# Evaluating global model on local data.
	train_loss, train_acc = evaluate_model(global_model, client_loader, tqdm_desc='local_train_loss')

	# Previous local model.
	prev_local_model = copy.deepcopy(glob_model)
	prev_local_model.load_state_dict(copy.deepcopy(prev_local_w))
	for param in prev_local_model.parameters():
		param.requires_grad = False

	# Current local model.
	local_model = copy.deepcopy(glob_model)

	# Params.
	device = next(glob_model.parameters()).device
	if optim == 'sgd':
		optimizer = torch.optim.SGD(local_model.parameters(), **optim_args)
	elif optim == 'adam':
		optimizer = torch.optim.Adam(local_model.parameters(), **optim_args)
	grad_scaler = torch.cuda.amp.GradScaler()
	
	loss_ce = torch.nn.CrossEntropyLoss()
	cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
	
	mu = moon_args['mu']
	temperature = moon_args['temperature']
	
	# Training.
	local_model.train()
	for epoch in range(num_local_epochs):
		for (x, y) in tqdm(client_loader, desc='epoch {}/{}'.format(epoch + 1, num_local_epochs)):
			optimizer.zero_grad() 
			x = x.to(device)
			y = y.to(device)

			# Extract outputs from
			y_pred, local_out = local_model(x)
			_, global_out = global_model(x)
			_, prev_local_out = prev_local_model(x)

			# Supervised loss.
			loss_sup = loss_ce(y_pred, y)

			# Contrastive loss.
			layer = moon_args['inter_layer']

			z = local_out[layer]
			z_glob = global_out[layer]
			z_prev = prev_local_out[layer]
	
			postive = torch.mean(cosine_similarity(z, z_glob) / temperature)
			negative = torch.mean(cosine_similarity(z, z_prev) / temperature)
			loss_con = - torch.log(torch.exp(postive) / (torch.exp(postive) + torch.exp(negative)))

			# print('loss_con:', loss_con)

			# Total loss.
			loss = loss_sup + mu * loss_con

			grad_scaler.scale(loss).backward()
			grad_scaler.step(optimizer)
			grad_scaler.update()

	# Return update.
	local_update_dict ={
		'local_w': local_model.state_dict(),
		'num_samples': len(client_loader.dataset),
		'train_loss': train_loss,
		'train_acc': train_acc,
	}

	return local_update_dict


def local_update_fedprox(glob_model, client_loader, num_local_epochs, optim, optim_args, fedprox_args):

	# Global model.
	global_model = copy.deepcopy(glob_model)
	for param in global_model.parameters():
		param.requires_grad = False

	# Evaluating global model on local data.
	train_loss, train_acc = evaluate_model(global_model, client_loader, tqdm_desc='local_train_loss')

	# Local model.
	local_model = copy.deepcopy(glob_model)

	# Params.
	device = next(glob_model.parameters()).device
	if optim == 'sgd':
		optimizer = torch.optim.SGD(local_model.parameters(), **optim_args)
	elif optim == 'adam':
		optimizer = torch.optim.Adam(local_model.parameters(), **optim_args)
	grad_scaler = torch.cuda.amp.GradScaler()
	
	loss_ce = torch.nn.CrossEntropyLoss()
	mu = fedprox_args['mu']
	
	# Training.
	local_model.train()
	for epoch in range(num_local_epochs):
		for (x, y) in tqdm(client_loader, desc='epoch {}/{}'.format(epoch + 1, num_local_epochs)):
			optimizer.zero_grad() 
			x = x.to(device)
			y = y.to(device)

			# Extract outputs from
			y_pred, local_out = local_model(x)

			# Supervised loss.
			loss_sup = loss_ce(y_pred, y)

			proximal_term = 0.0
			for w_local, w_global in zip(local_model.parameters(), global_model.parameters()):
				proximal_term += (w_local - w_global).norm(2)

			# Total loss.
			loss = loss_sup + (mu / 2) * proximal_term

			grad_scaler.scale(loss).backward()
			grad_scaler.step(optimizer)
			grad_scaler.update()

	# Return update.
	local_update_dict ={
		'local_w': local_model.state_dict(),
		'num_samples': len(client_loader.dataset),
		'train_loss': train_loss,
		'train_acc': train_acc,
	}

	return local_update_dict


# Local update of FedCKA.
def local_update_fedcka(glob_model, prev_local_w, client_loader, num_local_epochs, optim, optim_args, fedcka_args):

	# Global model.
	global_model = copy.deepcopy(glob_model)
	for param in global_model.parameters():
		param.requires_grad = False

	# Evaluating global model on local data. This is also used to calculate local mu.
	train_loss, train_acc = evaluate_model(global_model, client_loader, tqdm_desc='local_train_loss')

	# Previous local model.
	prev_local_model = copy.deepcopy(glob_model)
	prev_local_model.load_state_dict(copy.deepcopy(prev_local_w))
	for param in prev_local_model.parameters():
		param.requires_grad = False

	# Current local model.
	local_model = copy.deepcopy(glob_model)

	# Params.
	device = next(glob_model.parameters()).device
	if optim == 'sgd':
		optimizer = torch.optim.SGD(local_model.parameters(), **optim_args)
	elif optim == 'adam':
		optimizer = torch.optim.Adam(local_model.parameters(), **optim_args)
	grad_scaler = torch.cuda.amp.GradScaler()
	
	loss_ce = torch.nn.CrossEntropyLoss()
	cka = CudaCKA(device)
	
	mu = fedcka_args['mu']

	# Which intermediate representions to use.
	inter_layers = fedcka_args['inter_layers']

	# Training.
	local_model.train()
	for epoch in range(num_local_epochs):
		for (x, y) in tqdm(client_loader, desc='epoch {}/{}'.format(epoch + 1, num_local_epochs)):
			optimizer.zero_grad() 
			x = x.to(device)
			y = y.to(device)

			# Extract outputs from different models.
			y_pred, local_out = local_model(x)
			_, global_out = global_model(x)
			_, prev_local_out = prev_local_model(x)

			# Supervised loss.
			loss_sup = loss_ce(y_pred, y)

			# Intermediate representation loss.
			l_i = []
			for layer in inter_layers:

				z = local_out[layer]
				z_glob = global_out[layer]
				z_prev = prev_local_out[layer]

				postive = cka.linear_CKA(z, z_glob)
				negative = cka.linear_CKA(z, z_prev)

				loss = - torch.log(torch.exp(postive) / (torch.exp(postive) + torch.exp(negative)))
				l_i.append(loss)
			
			l_i = torch.stack(l_i)
			loss_cka = torch.mean(l_i)

			# Total loss.
			loss = loss_sup + mu * loss_cka

			grad_scaler.scale(loss).backward()
			grad_scaler.step(optimizer)
			grad_scaler.update()

	# Return update.
	local_update_dict ={
		'local_w': local_model.state_dict(),
		'num_samples': len(client_loader.dataset),
		'train_loss': train_loss,
		'train_acc': train_acc,
	}

	return local_update_dict



# Local update of FedIR.
# def local_update_fedir(glob_model, prev_local_w, client_loader, num_local_epochs, optim_args, fedir_args):

# 	# Global model.
# 	global_model = copy.deepcopy(glob_model)
# 	global_model.eval()
# 	for param in global_model.parameters():
# 		param.requires_grad = False

# 	# Evaluating global model on local data. This is also used to calculate local mu.
# 	train_loss, train_acc = evaluate_model(global_model, client_loader, tqdm_desc='local_train_loss')

# 	# Previous local model.
# 	prev_local_model = copy.deepcopy(glob_model)
# 	prev_local_model.load_state_dict(copy.deepcopy(prev_local_w))
# 	prev_local_model.eval()
# 	for param in prev_local_model.parameters():
# 		param.requires_grad = False

# 	# Current local model.
# 	local_model = copy.deepcopy(glob_model)

# 	# Params.
# 	device = next(glob_model.parameters()).device
# 	optimizer = torch.optim.SGD(local_model.parameters(), **optim_args)
# 	grad_scaler = torch.cuda.amp.GradScaler()
	
# 	loss_ce = torch.nn.CrossEntropyLoss()
# 	########### todo: handle different distance function.
# 	cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
	
# 	mu = fedir_args['mu']
# 	temperature = fedir_args['temperature']

# 	# Which intermediate representions to use.
# 	inter_layers = ['block1', 'block2', 'block3']

# 	# Training.
# 	local_model.train()
# 	for epoch in range(num_local_epochs):
# 		for (x, y) in tqdm(client_loader, desc='epoch {}/{}'.format(epoch + 1, num_local_epochs)):
# 			optimizer.zero_grad() 
# 			x = x.to(device)
# 			y = y.to(device)

# 			# Extract outputs from different models.
# 			y_pred, local_out = local_model(x)
# 			_, global_out = global_model(x)
# 			_, prev_local_out = prev_local_model(x)

# 			# Supervised loss.
# 			loss_sup = loss_ce(y_pred, y)

# 			# Intermediate representation loss.
# 			l_i = []
# 			alpha_i = []
# 			for layer in inter_layers:
# 				# Contrastive loss calculation.
# 				z = local_out[layer]
# 				z_glob = global_out[layer]
# 				z_prev = prev_local_out[layer]

# 				postive = torch.mean(cosine_similarity(z, z_glob) / temperature)
# 				negative = torch.mean(cosine_similarity(z, z_prev) / temperature)
# 				loss_con = - torch.log(torch.exp(postive) / (torch.exp(postive) + torch.exp(negative)))

# 				l_i.append(loss_con)
# 				alpha_i.append(postive)

# 			l_i = torch.stack(l_i)
# 			alpha_i = torch.nn.functional.softmax(torch.stack(alpha_i), 0)
# 			# Scale the loss of each intermediate representation with respective alpha and sum.
# 			loss_ir = torch.sum(l_i * alpha_i)

# 			# Total loss.
# 			loss = loss_sup + mu * loss_ir

# 			grad_scaler.scale(loss).backward()
# 			grad_scaler.step(optimizer)
# 			grad_scaler.update()

# 	# Return update.
# 	local_update_dict ={
# 		'local_w': local_model.state_dict(),
# 		'num_samples': len(client_loader.dataset),
# 		'train_loss': train_loss,
# 		'train_acc': train_acc,
# 	}

# 	return local_update_dict


# Local update of FedIntR.
def local_update_fedir(glob_model, prev_local_w, client_loader, num_local_epochs, optim, optim_args, fedir_args, ls_hw=False):

	# Global model.
	global_model = copy.deepcopy(glob_model)
	for param in global_model.parameters():
		param.requires_grad = False

	# Evaluating global model on local data. This is also used to calculate local mu.
	train_loss, train_acc = evaluate_model(global_model, client_loader, tqdm_desc='local_train_loss')

	# Previous local model.
	prev_local_model = copy.deepcopy(glob_model)
	prev_local_model.load_state_dict(copy.deepcopy(prev_local_w))
	for param in prev_local_model.parameters():
		param.requires_grad = False

	# Current local model.
	local_model = copy.deepcopy(glob_model)

	# Params.
	device = next(glob_model.parameters()).device
	if optim == 'sgd':
		optimizer = torch.optim.SGD(local_model.parameters(), **optim_args)
	elif optim == 'adam':
		optimizer = torch.optim.Adam(local_model.parameters(), **optim_args)
	grad_scaler = torch.cuda.amp.GradScaler()
	
	loss_ce = torch.nn.CrossEntropyLoss()
	loss_mse = torch.nn.MSELoss()
	cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
	cka = CudaCKA(device)
	
	mu = fedir_args['mu']
	temperature = fedir_args['temperature']

	# Which intermediate representions to use.
	inter_layers = fedir_args['inter_layers']

	# Training.
	local_model.train()
	for epoch in range(num_local_epochs):
		for (x, y) in tqdm(client_loader, desc='epoch {}/{}'.format(epoch + 1, num_local_epochs)):
			optimizer.zero_grad() 
			x = x.to(device)
			y = y.to(device)

			# Extract outputs from different models.
			y_pred, local_out = local_model(x)
			_, global_out = global_model(x)
			_, prev_local_out = prev_local_model(x)

			# Supervised loss.
			loss_sup = loss_ce(y_pred, y)

			# Intermediate representation loss.
			l_i = []
			alpha_i = []
			for layer in inter_layers:

				z = local_out[layer]
				z_glob = global_out[layer]
				z_prev = prev_local_out[layer]

				# Representation loss.
				if fedir_args['loss'] == 'contrastive':
					postive = torch.mean(cosine_similarity(z, z_glob) / temperature)
					negative = torch.mean(cosine_similarity(z, z_prev) / temperature)
					loss = - torch.log(torch.exp(postive) / (torch.exp(postive) + torch.exp(negative)))
					l_i.append(loss)
				elif fedir_args['loss'] == 'mse':
					loss = loss_mse(z, z_glob)
					l_i.append(loss)
				elif fedir_args['loss'] == 'neg_cosine':
					loss = - torch.mean(cosine_similarity(z, z_glob))
					l_i.append(loss)
				elif fedir_args['loss'] == 'cka':
					postive = cka.linear_CKA(z, z_glob)
					negative = cka.linear_CKA(z, z_prev)
					loss = - torch.log(torch.exp(postive) / (torch.exp(postive) + torch.exp(negative)))
					l_i.append(loss)

				# Scale factor.
				if fedir_args['scaling'] == 'avg':			# Average.
					alpha_i.append(torch.tensor(1.).to(device))
				elif fedir_args['scaling'] == 'cosine':
					sim = torch.mean(cosine_similarity(z, z_glob) / temperature)
					alpha_i.append(sim)
				elif fedir_args['scaling'] == 'cka':
					sim = cka.linear_CKA(z, z_glob)
					alpha_i.append(sim)

			l_i = torch.stack(l_i)
			alpha_i = torch.stack(alpha_i)

			if ls_hw:
				alpha_i = torch.neg(alpha_i)

			alpha_i = torch.nn.functional.softmax(alpha_i, 0)
			
			# Scale the loss of each intermediate representation with respective alpha and sum.
			loss_ir = torch.sum(l_i * alpha_i)

			# Total loss.
			loss = loss_sup + mu * loss_ir

			grad_scaler.scale(loss).backward()
			grad_scaler.step(optimizer)
			grad_scaler.update()

	# Return update.
	local_update_dict ={
		'local_w': local_model.state_dict(),
		'num_samples': len(client_loader.dataset),
		'train_loss': train_loss,
		'train_acc': train_acc,
	}

	return local_update_dict



# # Local update with intermediate layer representation regularization. 
# def local_update_rep_reg(glob_model, data_loader, optim_args, num_epochs):

# 	glob_w = glob_model.state_dict()

# 	global_model = copy.deepcopy(glob_model)
# 	global_model.load_state_dict(copy.deepcopy(glob_w))
# 	global_model.eval()
# 	for param in global_model.parameters():
# 		param.requires_grad = False

# 	local_model = copy.deepcopy(glob_model)
# 	local_model.load_state_dict(copy.deepcopy(glob_w))

# 	# Calculate local performance.
# 	local_loss, local_acc = evaluate_model(local_model, data_loader)

# 	optimizer = optim.SGD(local_model.parameters(), **optim_args)
# 	loss_ce = nn.CrossEntropyLoss()
# 	loss_mse = nn.MSELoss()
# 	mu = 1

# 	local_model.train()
# 	for epoch in range(num_epochs):
# 		for step, data in enumerate(data_loader):
# 			optimizer.zero_grad() 
# 			x_batch, y_batch = data[0].to(device), data[1].to(device)
			
# 			y_pred, (z1, z2, z3) = local_model(x_batch)
# 			_, (z1_glob, z2_glob, z3_glob) = global_model(x_batch)

# 			mse1 = loss_mse(z1, z1_glob)
# 			mse2 = loss_mse(z2, z2_glob)
# 			mse3 = loss_mse(z3, z3_glob)
# 			loss_rep = torch.mean(torch.stack((mse1, mse2, mse3)))

# 			loss_sup = loss_ce(y_pred, y_batch)
# 			loss = loss_sup + mu * loss_rep

# 			loss.backward()
# 			optimizer.step()

# 	# Return update.
# 	local_update_dict ={
# 		'local_w': local_model.state_dict(),
# 		'num_samples': len(data_loader.dataset),
# 		'loss': local_loss,
# 		'acc': local_acc
# 	}

# 	return local_update_dict