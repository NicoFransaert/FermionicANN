import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from collections import Counter


cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class sqsoftmax(torch.nn.Module):
    def __init__(self):
        super(sqsoftmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim = 1)
    def forward(self, input):
        return torch.sqrt(self.softmax(input))


class softsign_(torch.nn.Module):
    def __init__(self):
        super(softsign_, self).__init__()
        self.softsign = torch.nn.Softsign()
    def forward(self, input):
        return np.pi*(self.softsign(input))


class multiLayerRNN(torch.nn.Module):
    def __init__(self, inputdim, num_hidden, num_layers):
        super(multiLayerRNN, self).__init__()
        self.inputdim = inputdim
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        self.rnn = torch.nn.ModuleList([torch.nn.GRUCell(input_size=self.inputdim, hidden_size = self.num_hidden)] + [torch.nn.GRUCell(input_size=self.num_hidden, hidden_size = self.num_hidden) for n in range(self.num_layers-1)] )

    def forward(self, input, hidden_):
        hidden = torch.zeros_like(hidden_, device=device, dtype=torch.float32)
        current_hidden = input
        for n in range(self.num_layers):
            hidden[n] = self.rnn[n](current_hidden, hidden_[n])
            current_hidden = hidden[n].clone()

        return hidden[-1], hidden


class RNNwavefunction():
    def __init__(self, systemsize, inputdim = 2, n_electrons=2, hidden_size = 10, num_layers = 2 , seed=111):
        """
            systemsize:  int, size of the lattice
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
        """
        self.N = systemsize #Number of sites of the 1D chain
        self.n_electrons = n_electrons
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.inputdim = inputdim

        #Seeding
        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator
        torch.manual_seed(seed) # Torch seed

        #Defining the neural network
        self.rnn = multiLayerRNN(self.inputdim, self.hidden_size, self.num_layers )
        self.dense_ampl = torch.nn.Sequential(torch.nn.Linear(hidden_size, self.inputdim), sqsoftmax()) #square amplitude of (marginal) probability amplitudes
        self.dense_phase = torch.nn.Sequential(torch.nn.Linear(hidden_size, self.inputdim), softsign_()) #phase of (marginal) probability amplitudes

    def to_device(self,device):
        self.rnn.to(device)
        self.dense_ampl.to(device)
        self.dense_phase.to(device)

    def sample(self,numsamples):
        """
            Generate samples from a probability distribution parametrized by a recurrent network
            We also impose the total number of electrons to be n_electrons/2
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            ------------------------------------------------------------------------

        """

        self.outputdim = self.inputdim
        self.numsamples = numsamples

        rnn_state = torch.zeros(self.num_layers, self.numsamples, self.hidden_size, dtype = torch.float32, device = device) #store all hidden states

        # Initialise all samples as 0 -> 0 -> ... -> 0
        samples = torch.zeros(self.numsamples, self.N, device = device, dtype = torch.int64)
        
        # Make one hot encoded "0" for all samples, such that the very first sample gets calculated from this "0"
        # This means that either "0" or "1" can be obtained for the very first visible unit
        zero_row = torch.zeros(numsamples,1, device = device)
        one_row = torch.ones(numsamples,1, device = device)
        inputs = torch.cat((one_row, zero_row), dim=1)

        inputs_ampl = inputs

        # Keeps track of number of up spins
        counter = np.zeros((numsamples))

        # For all visible units, make them 0 or 1. Take care of total #ones == n_electrons
        for n in range(self.N):
  
            # if (n%5000==0 and n>0): print("sampler is at: {} samples".format(n))
            rnn_output, rnn_state = self.rnn(inputs_ampl, rnn_state)

            #Applying softmax layer
            output_ampl = self.dense_ampl(rnn_output)

            # We need to mask the probabilities which lead to samples where #spin up =/ #electrons
            output_mask = np.zeros((self.numsamples, self.inputdim))
            output_mask[:, 0] = self.N-self.n_electrons - (n-counter) > 0
            output_mask[:, 1] = self.n_electrons - counter > 0
            output_ampl = output_ampl * torch.from_numpy(output_mask)


            #sample from probabilities
            output_ampl = torch.nn.functional.normalize(output_ampl, eps = 1e-30)
            sample_temp = torch.multinomial(output_ampl**2, 1)[:,0]


            #store new samples
            samples[:,n] = sample_temp
            counter += np.array(sample_temp)
            #make the sampled degrees of freedom inputs for the next iteration
            inputs = torch.nn.functional.one_hot(sample_temp, num_classes = self.outputdim).float()
            inputs_ampl = inputs

        self.samples = samples
        # print([list(samples[i]).count(1) for i in range(len(samples))])
        return self.samples


    def amplitude(self,samples):
        """
            calculate the amplitudes of ```samples`` while imposing zero total angular momentum
            ------------------------------------------------------------------------
            Parameters:
            samples:         torch tensor
                             a tensor of shape (number of samples,systemsize)
                             containing the input samples in integer encoding
            ------------------------------------------------------------------------
            Returns:
            log-amps      torch tensor of shape (number of samples,2)
                             the amplitude and phase of each sample
            """

        self.outputdim = self.inputdim
        self.numsamples = samples.shape[0]

        zero_row = torch.zeros(self.numsamples,1, device = device)
        one_row = torch.ones(self.numsamples,1, device = device)
        inputs = torch.cat((one_row, zero_row), dim=1)

        rnn_state = torch.zeros(self.num_layers, self.numsamples, self.hidden_size, dtype = torch.float32, device = device)
        amplitudes = torch.zeros(self.numsamples, self.N, 2, device = device)  #2 dims for amplitude and phase
        tot_ampl = torch.zeros(self.numsamples, 2, device=device, dtype=torch.float32)

        inputs_ampl = inputs
        one_hot_samples = torch.nn.functional.one_hot(samples, num_classes = self.inputdim).float()

        # Initialise counter that keeps track of # up spins
        counter = np.zeros((self.numsamples))

		# This is the beauty of RNNs, namely that probability can efficiently be calculated from conditionals.
        for n in range(self.N):

            rnn_output, rnn_state = self.rnn(inputs_ampl, rnn_state)

            #Applying softmax layer
            output_ampl = self.dense_ampl(rnn_output)
            output_phase = self.dense_phase(rnn_output)

            output_mask = np.zeros((self.numsamples, self.inputdim))
            output_mask[:, 0] = self.N-self.n_electrons - (n-counter) > 0
            output_mask[:, 1] = self.n_electrons - counter > 0
            output_ampl = output_ampl * torch.from_numpy(output_mask)

            # store amplitude and phase of marginal probability amplitude
            output_ampl = torch.nn.functional.normalize(output_ampl, eps = 1e-30)
            amplitudes[:, n,  0] = (output_ampl * one_hot_samples[:,n]).sum(dim = 1)
            amplitudes[:, n,  1] = (output_phase * one_hot_samples[:,n]).sum(dim = 1)


            counter += np.array(samples[:,n])

            inputs = one_hot_samples[:,n]
            inputs_ampl = inputs


        tot_ampl = torch.zeros(self.numsamples, 2, device=device, dtype=torch.float32)
        tot_ampl[:, 0] = amplitudes[:,:,0].prod(dim = 1)
        tot_ampl[:, 1] = amplitudes[:,:,1].sum(dim = 1)

        self.amplitudes = tot_ampl
        return self.amplitudes
