# Imports
import netket as nk
from qiskit.chemistry import FermionicOperator
import numpy as np
import json
import time
import os

try:
	import torch
	import torch.backends.cudnn as cudnn
except:
	print('torch not installed')

# Own imports
from ComplexRNNwavefunction import RNNwavefunction
import utility as ut
from JW_hamiltonian import JW_H

# Cuda settings
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# print(device)


# num_units stands for the dimension d_h of the hidden states h_n (which get concatenated with one-hot visible units)
# num_layers is the 'height' (# RNN cells) at every visible unit
# See literature/Hibat_RecurrentNNWF.pdf Figure 1!
def run_RNN(systemData, num_units = 50, num_layers = 2, learningrate = 2.5e-4, lrschedule='O', numsamples = 500, numsteps = 1000, save_dir=None, seed = 123):

	# make outfile
	path = './../data/RNN_runs/'
	if save_dir: 
		try: os.mkdir(path+save_dir)
		except: pass
		path += save_dir + '/'
	filename = 'rnn_'
	filename += systemData['basis'] + '_'
	filename += systemData['molecule'] + '_'
	filename += str(systemData['distance']).replace('.', '-')[:6] + '_'
	filename += 'eq' + str(int(systemData['eq'])) + '_'
	filename += 'nU' + str(num_units) + '_'
	filename += 'nL' + str(num_layers) + '_'
	filename += 'lr' + str(learningrate).split('.')[-1] + '_'
	filename += 'lrs' + str(lrschedule) + '_'
	filename += 'ns' + str(numsamples)
	outfile = path+filename

	print(' \n #### outfile is: ', outfile, ' #### \n')

	# extract information from systemData
	N = systemData['n_basisfuncs']
	n_electrons= systemData['n_electrons']

	# make hamiltonian operator
	ha = JW_H(systemData=systemData)

	# seed engines
	np.random.seed(seed)
	torch.manual_seed(seed)


	# make RNN
	wf = RNNwavefunction(N, inputdim=2, n_electrons=n_electrons, hidden_size=num_units, num_layers=num_layers, seed=seed)
	numparam = sum(p.numel() for p in wf.rnn.parameters() if p.requires_grad)
	numparam += sum(p.numel() for p in wf.dense_ampl.parameters() if p.requires_grad)
	numparam += sum(p.numel() for p in wf.dense_phase.parameters() if p.requires_grad)
	print('number of parameters: ', numparam)
	wf.to_device(device)
	# store params
	params= list(wf.rnn.parameters()) + list(wf.dense_ampl.parameters()) + list(wf.dense_phase.parameters())
	
	# learning rate schedule. 'C' = constant, 'O' = Original based on Hibat paper.
	if lrschedule == 'C':
		adjust_lr = lambda epoch: 1  # Learning rate is constant
	elif lrschedule == 'O':
		adjust_lr = lambda epoch: 1./(1. + 0.1*(epoch*learningrate))
	else:
		NotImplementedError('Learning rate schedule not implemented')

	# optimizer
	optimizer = torch.optim.Adam(params, lr=learningrate)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=adjust_lr)


	# initialise all matrices
	sigmas = torch.zeros((2*N**2*numsamples,N), dtype=torch.int64) # Array to store all the diagonal and non diagonal sigmas for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	H = torch.zeros(2*N**2*numsamples, dtype=torch.float32) # Array to store all the diagonal and non diagonal matrix elements for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	sigmaH = torch.zeros((2*N**2,N), dtype=torch.int32) # Array to store all the diagonal and non diagonal sigmas for each sample sigma
	matrixelements = torch.zeros(2*N**2, dtype=torch.float32) # Array to store all the diagonal and non diagonal matrix elements for each sample sigma (the number of matrix elements is bounded by at most 2N)

	amplitudes = torch.zeros(2*N**2*numsamples, 2, dtype=torch.float32, device=device) # Array to store all the diagonal and non diagonal log_probabilities for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	local_energies = torch.zeros(numsamples, 2, dtype=torch.float32, device=device) # The type complex should be specified, otherwise the imaginary part will be discarded

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
		# print('sampling of ', numsamples, ' samples took: ', sample_times[-1])

		# start_time_Ecalculation = time.time()
		with torch.no_grad():

			# start_time_slicing = time.time()
			slices, len_sigmas = J1J2Slices(ha, samples.cpu().numpy(), sigmas, H, sigmaH, matrixelements, n_electrons)
			# end_time_slicing = time.time()
			# print('slicing took: ', end_time_slicing-start_time_slicing)

			steps = len_sigmas//30000+1 # Process the sigmas in steps to avoid allocating too much memory
			# steps = 1 # Process the sigmas in steps to avoid allocating too much memory

			for i in range(steps):
				if i < steps-1:
					cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
				else:
					cut = slice((i*len_sigmas)//steps,len_sigmas)

				# start_time_amplitudeslice = time.time()
				amplitudes[cut] = wf.amplitude(sigmas[cut].to(device))
				# end_time_amplitudeslice = time.time()
				# print('calculating amplitudes[cut] took (all sigmas, i.e. 1 slice=all): ', end_time_amplitudeslice-start_time_amplitudeslice)

			# generating the local energies
			# start_time_localE = time.time()
			for n in range(len(slices)):
				s=slices[n]
				local_energies[n,0] = torch.dot(H[s].to(device), (torch.mul(amplitudes[s][:,0]/amplitudes[s][0,0],torch.cos(amplitudes[s][:,1]-amplitudes[s][0,1])))) #real part
				local_energies[n,1] = torch.dot(H[s].to(device), (torch.mul(amplitudes[s][:,0]/amplitudes[s][0,0],torch.sin(amplitudes[s][:,1]-amplitudes[s][0,1])))) #complex part
			# end_time_localE = time.time()
			# print('local energy calculation took: ', end_time_localE-start_time_localE)

		# end_time_Ecalculation = time.time()
		# print('calculation of local energies took: ', end_time_Ecalculation - start_time_Ecalculation)

		# start_time_amplitudes = time.time()
		amplitudes_ = wf.amplitude(samples)
		# end_time_amplitudes = time.time()
		# print('calculating amplitudes took: ', end_time_amplitudes - start_time_amplitudes)

		# start_time_cost = time.time()
		cost = 2 *  torch.mean(torch.log(amplitudes_[:,0]) * local_energies[:,0] + amplitudes_[:,1] * local_energies[:,1])
		cost = cost - 2* torch.mean(torch.log(amplitudes_[:,0]))*torch.mean(local_energies[:,0]) - 2*torch.mean(amplitudes_[:,1])*torch.mean(local_energies[:,1])
		# cost = 2 *  torch.mean(torch.log(amplitudes_[:,0]) * local_energies[:,0])
		# cost = cost - 2* torch.mean(torch.log(amplitudes_[:,0]))*torch.mean(local_energies[:,0])
		# end_time_cost = time.time()
		# print('calculating cost took: ', end_time_cost-start_time_cost)

		# start_time_backward = time.time()
		cost.backward()
		optimizer.step()
		scheduler.step()
		# end_time_backward = time.time()
		# print('backward pass took: ', end_time_backward-start_time_backward)

		meanE = torch.mean(local_energies, dim=0)[0]
		varE = torch.var((local_energies[:,0]))
		print('mean energy: ', meanE.item())
		print('variance: ', varE.item())
		
		step_dictionary = {
		"Energy": { "Mean": float(meanE) },
		"EnergyVariance": { "Mean": float(varE) },
		"Iteration": int(step) }

		energy_dictList.append(step_dictionary)

	# Error bar on last energy and variance (https://math.stackexchange.com/questions/72975/variance-of-sample-variance)
	sigmaE = torch.sqrt( torch.mean((local_energies[:,0]-meanE)**2)/numsamples )
	sigmavarE = torch.sqrt(torch.abs( torch.mean((local_energies[:,0]-meanE)**4)/numsamples - varE**4*(numsamples-3)/(numsamples*(numsamples-1)) ))

	end = time.time()

    # Write data to savefile. Energies & variances for all iterations and wf for final iteration. 
	dictionary["Output"] = energy_dictList
	with open(outfile+'.log', 'w') as f:
		json.dump(dictionary, f)

	torch.save(wf, outfile+'.pt')

	# Intermediate save --- will be overwritten after evaluation run!
	with open(outfile+'.META', 'w') as f:
		json.dump({	"SystemData": systemData,
					"Total_energy": {"Mean": systemData['nuc_rep_energy'] + float(meanE),
									 "Sigma": float(sigmaE)},
					"Energy_variance": {"Mean": float(varE), 
										"Sigma": float(sigmavarE)}, 
                    "Time_optimization": end-start, 
                    "Network": {"machine": "crnn", "nU": num_units, "nL": num_layers, "n_par": numparam},
                    "Training": {"optimiser": "adam", "lr": learningrate, "lrs": lrschedule, "numsamples": numsamples, "seed": seed, "steps": numsteps},
					"Evaluation_samples": 0,
                    "LocalSize": 2,
					"Time_sampling": {"Mean": np.mean(sample_times), "Variance": np.var(sample_times)},
		}, f)



	### --------------- Evaluation run --------------------- ######
	evalsamples = int(1e6)
	samples = wf.sample(evalsamples)
	print('start evaluation with: ', evalsamples, ' samples')

	# initialise all matrices
	sigmas = torch.zeros((2*N**2*evalsamples,N), dtype=torch.int64) # Array to store all the diagonal and non diagonal sigmas for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	H = torch.zeros(2*N**2*evalsamples, dtype=torch.float32) # Array to store all the diagonal and non diagonal matrix elements for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	sigmaH = torch.zeros((2*N**2,N), dtype=torch.int32) # Array to store all the diagonal and non diagonal sigmas for each sample sigma
	matrixelements = torch.zeros(2*N**2, dtype=torch.float32) # Array to store all the diagonal and non diagonal matrix elements for each sample sigma (the number of matrix elements is bounded by at most 2N)

	amplitudes = torch.zeros(2*N**2*evalsamples, 2, dtype=torch.float32, device=device) # Array to store all the diagonal and non diagonal log_probabilities for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
	local_energies = torch.zeros(evalsamples, 2, dtype=torch.float32, device=device) # The type complex should be specified, otherwise the imaginary part will be discarded


	with torch.no_grad():

		# start_time_slicing = time.time()
		slices, len_sigmas = J1J2Slices(ha, samples.cpu().numpy(), sigmas, H, sigmaH, matrixelements, n_electrons)
		# end_time_slicing = time.time()
		# print('slicing took: ', end_time_slicing-start_time_slicing)

		steps = len_sigmas//30000+1 # Process the sigmas in steps to avoid allocating too much memory
		# steps = 1 # Process the sigmas in steps to avoid allocating too much memory

		for i in range(steps):
			if i < steps-1:
				cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
			else:
				cut = slice((i*len_sigmas)//steps,len_sigmas)

			# start_time_amplitudeslice = time.time()
			amplitudes[cut] = wf.amplitude(sigmas[cut].to(device))
			# end_time_amplitudeslice = time.time()
			# print('calculating amplitudes[cut] took (all sigmas, i.e. 1 slice=all): ', end_time_amplitudeslice-start_time_amplitudeslice)

		# generating the local energies
		# start_time_localE = time.time()
		for n in range(len(slices)):
			s=slices[n]
			local_energies[n,0] = torch.dot(H[s].to(device), (torch.mul(amplitudes[s][:,0]/amplitudes[s][0,0],torch.cos(amplitudes[s][:,1]-amplitudes[s][0,1])))) #real part
			local_energies[n,1] = torch.dot(H[s].to(device), (torch.mul(amplitudes[s][:,0]/amplitudes[s][0,0],torch.sin(amplitudes[s][:,1]-amplitudes[s][0,1])))) #complex part
		# end_time_localE = time.time()
		# print('local energy calculation took: ', end_time_localE-start_time_localE)

	# end_time_Ecalculation = time.time()
	# print('calculation of local energies took: ', end_time_Ecalculation - start_time_Ecalculation)

	meanE = torch.mean(local_energies, dim=0)[0]
	varE = torch.var((local_energies[:,0]))
	print('mean energy: ', meanE.item())
	print('variance: ', varE.item())
	print('The final system energy is:', systemData['nuc_rep_energy'] + float(meanE))

	# Error bar on last energy and variance (https://math.stackexchange.com/questions/72975/variance-of-sample-variance)
	sigmaE = torch.sqrt( torch.mean((local_energies[:,0]-meanE)**2)/numsamples )
	sigmavarE = torch.sqrt(torch.abs( torch.mean((local_energies[:,0]-meanE)**4)/numsamples - varE**4*(numsamples-3)/(numsamples*(numsamples-1)) ))

	# Save all useful energy in one file
	with open(outfile+'.META', 'w') as f:
		json.dump({	"SystemData": systemData,
					"Total_energy": {"Mean": systemData['nuc_rep_energy'] + float(meanE),
									 "Sigma": float(sigmaE)},
					"Energy_variance": {"Mean": float(varE), 
										"Sigma": float(sigmavarE)}, 
                    "Time_optimization": end-start, 
                    "Network": {"machine": "crnn", "nU": num_units, "nL": num_layers, "n_par": numparam},
                    "Training": {"optimiser": "adam", "lr": learningrate, "lrs": lrschedule, "numsamples": numsamples, "seed": seed, "steps": numsteps},
					"Evaluation_samples": evalsamples,
                    "LocalSize": 2,
					"Time_sampling": {"Mean": np.mean(sample_times), "Variance": np.var(sample_times)},
		}, f)


	print('end evaluation')

	##### --------- End evaluation run ---------- ######


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

	conn_states, matrix_elements = ha.get_conn(sigmap)

	matrixelements[:len(matrix_elements)] = torch.from_numpy(matrix_elements.real)
	sigmaH[:len(conn_states)] = torch.from_numpy(conn_states)
	num = len(matrix_elements)

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