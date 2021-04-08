# Imports
import netket as nk
from qiskit.chemistry import FermionicOperator
import numpy as np
import json
import os
import time

import torch
import torch.backends.cudnn as cudnn

# Own imports
from ComplexRNNwavefunction import RNNwavefunction
import utility as ut

# Cuda settings
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


# num_units stands for the dimension d_h of the hidden states h_n (which get concatenated with one-hot visible units)
# num_layers is the 'height' (# RNN cells) at every visible unit
# See literature/Hibat_RecurrentNNWF.pdf Figure 1!
def run_RNN(N = 10, num_units = 50, num_layers = 2, learningrate = 2.5e-4, lrschedule='O', numsamples = 500, numsteps = 1000, seed = 123):

	# At the moment, 'C' (constant), 'H' (Hibat) and 'O' (Original) are available
	lrs = lrschedule

	# seed engines
	np.random.seed(seed)
	nk.legacy.random.seed(seed=seed)
	torch.manual_seed(seed)

	# path, filename & outfile for logging E_mean, E_var & wf.
	path = './../data/RNN_runs/rnn/'
	filename = 'testLiH_4'
	outfile = path + filename

	#LiH
	OB = np.load('../data/integrals/STO-3G/STO-3G_LiH_OB_d1-548_eq1.npy')
	TB = np.load('../data/integrals/STO-3G/STO-3G_LiH_TB_d1-548_eq1.npy')
	N=6
	n_electrons=4

	########
	FerOp = FermionicOperator(OB, TB)

	mapping = FerOp.mapping('jordan_wigner')
	weights = [w[0] for w in mapping.paulis]
	operators = [w[1].to_label() for w in mapping.paulis]

	ha = nk.operator.PauliStrings(operators, weights)
	hi = ha.hilbert

	# this was wrong, since wf was initiated with systemsize N instead of N-1
	wf = RNNwavefunction(N, inputdim=2, n_electrons=n_electrons, hidden_size=num_units, num_layers=num_layers, seed=seed)
	# wf = torch.load(outfile)
	numparam = sum(p.numel() for p in wf.rnn.parameters() if p.requires_grad)
	numparam += sum(p.numel() for p in wf.dense_ampl.parameters() if p.requires_grad)
	numparam += sum(p.numel() for p in wf.dense_phase.parameters() if p.requires_grad)
	print('number of parameters: ', numparam)
	wf.to_device(device)

	params= list(wf.rnn.parameters()) + list(wf.dense_ampl.parameters()) + list(wf.dense_phase.parameters())
	
	if lrs == 'C':
		adjust_lr = lambda epoch: 1  # Learning rate is constant
	elif lrs == 'O':
		adjust_lr = lambda epoch: 1./(1. + 0.1*(epoch*learningrate)) # Original lrs of RNN, which is equal to Hibat (lr = lr*adjust)
	else:
		NotImplementedError('Learning rate schedule not implemented')

	optimizer = torch.optim.Adam(params, lr=learningrate)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=adjust_lr)

	# CHECK WHICH HAVE TO BE COMPLEX
	# Why 5*N?
	sigmas = torch.zeros((5*N*numsamples,N), dtype=torch.int64) # Array to store all the diagonal and non diagonal sigmas for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	H = torch.zeros(5*N*numsamples, dtype=torch.float32) # Array to store all the diagonal and non diagonal matrix elements for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	sigmaH = torch.zeros((5*N,N), dtype=torch.int32) # Array to store all the diagonal and non diagonal sigmas for each sample sigma
	matrixelements = torch.zeros(5*N, dtype=torch.float32) # Array to store all the diagonal and non diagonal matrix elements for each sample sigma (the number of matrix elements is bounded by at most 2N)

	amplitudes = torch.zeros(5*N*numsamples, 2, dtype=torch.float32, device=device) # Array to store all the diagonal and non diagonal log_probabilities for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	local_energies = torch.zeros(numsamples, 2, dtype=torch.float32, device=device) # The type complex should be specified, otherwise the imaginary part will be discarded

	# meanEnergy=[]
	# varEnergy=[]

	# initialise dictionary for logging energie and varE
	dictionary = {"Output": dict()}
	energy_dictList = []

	# optimization
	start = time.time()
	sample_times = []

	for step in range(numsteps):

		print('optimization step ', step)

		optimizer.zero_grad()

		start_time_sampling = time.time()
		samples = wf.sample(numsamples)
		end_time_sampling = time.time()
		sample_times.append(end_time_sampling - start_time_sampling)

		with torch.no_grad():

			slices, len_sigmas = J1J2Slices(ha, samples.cpu().numpy(), sigmas, H, sigmaH, matrixelements, n_electrons)

			steps = len_sigmas//30000+1 # Process the sigmas in steps to avoid allocating too much memory

			for i in range(steps):
				if i < steps-1:
					cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
				else:
					cut = slice((i*len_sigmas)//steps,len_sigmas)


				amplitudes[cut] = wf.amplitude(sigmas[cut].to(device))

			#Generating the local energies
			for n in range(len(slices)):
				s=slices[n]
				local_energies[n,0] = torch.dot(H[s].to(device), (torch.mul(amplitudes[s][:,0]/amplitudes[s][0,0],torch.cos(amplitudes[s][:,1]-amplitudes[s][0,1])))) #real part
				local_energies[n,1] = torch.dot(H[s].to(device), (torch.mul(amplitudes[s][:,0]/amplitudes[s][0,0],torch.sin(amplitudes[s][:,1]-amplitudes[s][0,1])))) #complex part

		amplitudes_ = wf.amplitude(samples)
		cost = 2 *  torch.mean(torch.log(amplitudes_[:,0]) * local_energies[:,0] + amplitudes_[:,1] * local_energies[:,1])
		cost = cost - 2* torch.mean(torch.log(amplitudes_[:,0]))*torch.mean(local_energies[:,0]) - 2*torch.mean(amplitudes_[:,1])*torch.mean(local_energies[:,1])
		cost.backward()
		optimizer.step()
		scheduler.step()

		meanE = torch.mean(local_energies, dim=0)[0]
		varE = torch.var((local_energies[:,0]))

		# meanEnergy.append(meanE)
		# varEnergy.append(varE)

		step_dictionary = {
		"Energy": { "Mean": float(meanE) },
		"EnergyVariance": { "Mean": float(varE) },
		"Iteration": int(step) }

		energy_dictList.append(step_dictionary)

	end = time.time()

    # Write data to savefile. Energies & variances for all iterations and wf for final iteration. 
	dictionary["Output"] = energy_dictList
	with open(outfile+'.log', 'w') as f:
		json.dump(dictionary, f)

	torch.save(wf, outfile+'.pt')

	# ------------- Start evaluation run ---------------------------
	print('starting evaluation run')

	eval_dictionary = {"Output": dict()}

	start_eval = time.time()
	eval_samples = int(1e5)
	samples = wf.sample(eval_samples)

	sigmas = torch.zeros((5*N*eval_samples,N), dtype=torch.int64) 
	H = torch.zeros(5*N*eval_samples, dtype=torch.float32) 
	sigmaH = torch.zeros((5*N,N), dtype=torch.int32)
	matrixelements = torch.zeros(5*N, dtype=torch.float32) 
	amplitudes = torch.zeros(5*N*eval_samples, 2, dtype=torch.float32, device=device)
	local_energies = torch.zeros(eval_samples, 2, dtype=torch.float32, device=device)

	with torch.no_grad():

		slices, len_sigmas = J1J2Slices(ha, samples.cpu().numpy(), sigmas, H, sigmaH, matrixelements, n_electrons)

		steps = len_sigmas//30000+1 # Process the sigmas in steps to avoid allocating too much memory

		for i in range(steps):
			if i < steps-1:
				cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
			else:
				cut = slice((i*len_sigmas)//steps,len_sigmas)


			amplitudes[cut] = wf.amplitude(sigmas[cut].to(device))

		# Generating the local energies
		for n in range(len(slices)):
			s=slices[n]
			local_energies[n,0] = torch.dot(H[s].to(device), (torch.mul(amplitudes[s][:,0]/amplitudes[s][0,0],torch.cos(amplitudes[s][:,1]-amplitudes[s][0,1])))) #real part
			local_energies[n,1] = torch.dot(H[s].to(device), (torch.mul(amplitudes[s][:,0]/amplitudes[s][0,0],torch.sin(amplitudes[s][:,1]-amplitudes[s][0,1])))) #complex part	


		eval_meanE = torch.mean(local_energies, dim=0)[0]
		eval_varE = torch.var((local_energies[:,0]))

		# Error bar on energy and variance (https://math.stackexchange.com/questions/72975/variance-of-sample-variance)
		eval_sigmaE = torch.sqrt( torch.mean((local_energies[:,0]-eval_meanE)**2)/eval_samples )
		eval_sigmavarE = torch.sqrt(torch.abs( torch.mean((local_energies[:,0]-eval_meanE)**4)/eval_samples - eval_varE**4*(eval_samples-3)/(eval_samples*(eval_samples-1)) ))


	eval_step_dictionary = {
	"Energy": { "Mean": float(eval_meanE), "Sigma": float(eval_sigmaE) },
	"EnergyVariance": { "Mean": float(eval_varE), "Sigma": float(eval_sigmavarE) },
	"Iteration": int(0) }

	eval_dictionary["Output"] = [eval_step_dictionary]
	with open(outfile+'_eval.log', 'w') as f:
		json.dump(eval_dictionary, f)


	end_eval = time.time()

	print('ending evaluation run')
	# ------------- END evaluation run -----------------------


	# Also save the time of the optimization
	with open(outfile+'.META', 'w') as f:
		json.dump({	"Time_optimization": end-start, 
					"Time_sampling": {"Mean": np.mean(sample_times), "Variance": np.var(sample_times)},
					"LocalSize": 2,
					"Seed": seed,
					"Evaluation_samples": eval_samples,
					"Evaluation_time": end_eval-start_eval,
					"Parameters_total": numparam}, f)



def J1J2MatrixElements(ha, sigmap, sigmaH, matrixelements, n_electrons):
	"""
	-Computes the matrix element of the hamiltonian for a given configuration sigmap
	-----------------------------------------------------------------------------------
	Parameters:
	ha: 			netket operator representing the hamiltonian
	sigmap: 		np.ndarray of dtype=int and shape (N-1)
	sigmaH: 		an array to store the diagonal and the non-diagonal configurations after applying the Hamiltonian on sigmap.
	matrixelements: an array where to store the matrix elements after applying the Hamiltonian on sigmap.
	-----------------------------------------------------------------------------------
	Returns: num, float which indicate the number of diagonal and non-diagonal configurations after applying the Hamiltonian on sigmap
	"""

	# get all connected matrix elements from netket hamiltonian
	# print(sigmap)
	# mel, connectors, newconfs = ha.get_conn(sigmap)
	# print('mel: ', mel)
	# print('connectors: ', connectors)
	# print('newconfs: ', newconfs)

	conn_states, matrix_elements = ha.get_conn(sigmap)

	k=0
	for i in range(len(matrix_elements)):
		if list(conn_states[i]).count(1) == n_electrons/2:
			matrixelements[k] = matrix_elements[i].real
			sigmaH[k] = torch.from_numpy(conn_states[i])
			k += 1

	# print('sigmap: ', sigmap)
	# print('connected states: ', conn_states)
	# print('sigmaH: ', sigmaH)
	# print('k: ', k)
		# matrixelements[i] = matrix_elements[i].real
		# sigmaH[i] = torch.from_numpy(conn_states[i])


	# num = len(mel) # Number of basis elements
	# num = len(matrix_elements)
	num = k
	# construct connected states as full configuration from output of netket
	# for i in range(len(mel)):
	# 	sig = np.copy(sigmap)
	# 	for j in range(len(connectors[i])):
	# 		sig[connectors[i][j]] = newconfs[i][j]
	# 	matrixelements[i] = mel[i].real 	# Be careful with only taking the real component. For AFH with only nn interactions this is ok.
	# 	sigmaH[i] = torch.from_numpy(sig)

	return num


def J1J2Slices(ham, sigmasp, sigmas, H, sigmaH, matrixelements, n_electrons):
	"""
	Returns: A tuple 
					-The list of slices (that will help to slice the array sigmas)
					-Total number of configurations after applying the Hamiltonian on the list of samples sigmasp (This will be useful later during training, note that it is not constant)
	----------------------------------------------------------------------------
	Parameters:
	ham: 			netket operator representing the hamiltonian
	sigmasp:    	np.ndarrray of dtype=int and shape (numsamples,N) - angular momentum states, integer encoded
	sigmas: 		an array to store the diagonal and the non-diagonal configurations after applying the Hamiltonian on all the samples sigmasp.
	H: 				an array to store the diagonal and the non-diagonal matrix elements after applying the Hamiltonian on all the samples sigmasp.
	sigmaH: 		an array to store the diagonal and the non-diagonal configurations after applying the Hamiltonian on a single sample.
	matrixelements: an array where to store the matrix elements after applying the Hamiltonian on sigmap on a single sample.
	----------------------------------------------------------------------------
	"""

	slices=[]
	sigmas_length = 0

	for n in range(sigmasp.shape[0]):
		sigmap = sigmasp[n,:]
		num = J1J2MatrixElements(ham ,sigmap, sigmaH, matrixelements, n_electrons) #note that sigmas[0,:]==sigmap, matrixelements and sigmaH are updated
		slices.append(slice(sigmas_length,sigmas_length + num))
		s = slices[n]

		H[s] = matrixelements[:num]
		sigmas[s] = sigmaH[:num]

		sigmas_length += num #Increasing the length of matrix elements sigmas

	return slices, sigmas_length


if __name__ == "__main__":
	run_RNN(N = 10, num_units = 50, num_layers = 1, learningrate = 5e-3, lrschedule='C', numsamples = 2000, numsteps = 2000, seed = 123)