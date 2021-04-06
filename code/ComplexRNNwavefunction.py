import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np


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
    def __init__(self, systemsize, inputdim = 2, hidden_size = 10, num_layers = 2 , seed=111):
        """
            systemsize:  int, size of the lattice
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
        """
        self.N = systemsize #Number of sites of the 1D chain
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
            We also impose zero total angular momentum (SU(2) symmetry)
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            ------------------------------------------------------------------------

        """

        self.outputdim = self.inputdim
        self.numsamples = numsamples

        rnn_state = torch.zeros(self.num_layers, self.numsamples, self.hidden_size, dtype = torch.float32, device = device) #store all hidden states
        # samples = torch.zeros(self.numsamples, self.N, device = device, dtype = torch.int64)  #store all samples
        # # Set all samples to 0 -> 1 
        # samples[:,1]=1
        
        # # For a fusion tree, all samples start with j_0=0 and j_1=1 (by construction).  Make appropriate input tensor:
        # zero_row = torch.zeros(numsamples,1, device = device)
        # one_row = torch.ones(numsamples,1, device = device)
        # rest_zeros = torch.zeros(numsamples,self.inputdim-2, device = device) 
        # inputs = torch.cat((zero_row, one_row, rest_zeros), dim=1)

        # inputs_ampl = inputs
        # # Now every sample has as input the one hot (0,1,...,d_v) meaning L=1 state. Based on this, the next visible (X) will be generated: 0 -> 1 -> X

        '''
        I have the feeling that the above reasoning leads to the wrong result.
        Namely, the samples need to not start with 0 -> 1 -> X, where X is the first 'real' dof.
        Rather, we need to start with X -> Y -> Z -> ... -> 0, where X can be either 0 or 2
        
        Otherwise, we miss the states 2 -> Y -> ... -> 0 completely, and therefore cannot reach the GS

        Thus, we need to initialise with value "1" (which is just the first spin coupled to itself) and then use the loop below.
        Afterwards, we discard the initial "1" (we really never put it into the samples).

        This looks as follows
         '''
         # Initialise all samples as 0 -> 0 -> ... -> 0
        samples = torch.zeros(self.numsamples, self.N, device = device, dtype = torch.int64)
        
        # Make one hot encoded "1" for all samples, such that the very first sample gets calculated from this "1"
        # This means that either "0" or "2" can be obtained for the very first visible unit
        zero_row = torch.zeros(numsamples,1, device = device)
        one_row = torch.ones(numsamples,1, device = device)
        rest_zeros = torch.zeros(numsamples, self.inputdim-2, device = device) 
        inputs = torch.cat((zero_row, one_row, rest_zeros), dim=1)

        inputs_ampl = inputs

        
        # Now, we need to START AT 0 instead of 2
        for n in range(self.N):

            if (n%5000==0 and n>0): print("sampler is at: {} samples".format(n))

            rnn_output, rnn_state = self.rnn(inputs_ampl, rnn_state)

            #Applying softmax layer
            output_ampl = self.dense_ampl(rnn_output)

            zero_row = torch.zeros(numsamples,1, device = device)
            expanded_inputs_ampl = torch.cat((zero_row,inputs_ampl,zero_row), dim=1) #pad inputs_ampl with zeros
            input_ampl_up = torch.roll(expanded_inputs_ampl,1,dims=1)
            input_ampl_down = torch.roll(expanded_inputs_ampl,-1,dims=1)

            #output_mask has zeros where value of next angular momentum is impossible, one otherwise
            output_mask = torch.zeros_like(expanded_inputs_ampl)
            output_mask = torch.max(input_ampl_down, input_ampl_up) # Place ones at the place left and right to last one-hot L

            #undo the padding with zeros
            output_mask = output_mask[:,1:-1]

            #in this case, next angular momentum can only go down when current angular momentum is equal to number of sites until end of chain is reached: adjust the mask appropriately
            if self.N - n < self.inputdim:
                output_mask_mask = torch.cat( (torch.ones(numsamples, self.N - n, device = device), torch.zeros(numsamples, self.inputdim - (self.N - n), device=device)), dim=1)
                output_mask = output_mask*output_mask_mask
            
            #use mask to only leave valid probabilities for next state
            output_ampl = output_ampl*output_mask
            output_ampl = torch.nn.functional.normalize(output_ampl, eps = 1e-30)

            #sample from probabilities
            #we only sample one instance, therefore replacement argument is irrelevant
            sample_temp = torch.multinomial(output_ampl**2, 1)[:,0]
            #store new samples
            samples[:,n] = sample_temp
            #make the sampled degrees of freedom inputs for the next iteration
            inputs = torch.nn.functional.one_hot(sample_temp, num_classes = self.outputdim).float()

            inputs_ampl = inputs

        self.samples = samples
        # print(samples[:,0])
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

        # For a fusion tree, all samples start with j_0=0 and j_1=1 (by construction).  Make appropriate input tensor:
        # The above is not true. Samples can also start with j_0=2
        # The thing is that the initial j_start=1 is still the case. From this j_start, j_0 is calculated as either 0 or 2.
        zero_row = torch.zeros(self.numsamples,1, device = device)
        one_row = torch.ones(self.numsamples,1, device = device)
        rest_zeros = torch.zeros(self.numsamples,self.inputdim-2, device = device)
        inputs = torch.cat((zero_row, one_row, rest_zeros), dim=1)

        rnn_state = torch.zeros(self.num_layers, self.numsamples, self.hidden_size, dtype = torch.float32, device = device)

        amplitudes = torch.zeros(self.numsamples, self.N, 2, device = device)  #2 dims for amplitude and phase

        inputs_ampl = inputs

        # The following is only available for torch versions 1.1.0 and greater (thus 1.0.X is not supported)
        one_hot_samples = torch.nn.functional.one_hot(samples, num_classes = self.inputdim).float()

		# This is the beauty of RNNs, namely that probability can efficiently be calculated from conditionals.
        for n in range(self.N):

            rnn_output, rnn_state = self.rnn(inputs_ampl, rnn_state)

            #Applying softmax layer
            output_ampl = self.dense_ampl(rnn_output)
            output_phase = self.dense_phase(rnn_output)


            zero_row = torch.zeros(self.numsamples,1, device = device)
            expanded_inputs_ampl = torch.cat((zero_row,inputs_ampl,zero_row), dim=1) #pad inputs_ampl with zeros
            input_ampl_up = torch.roll(expanded_inputs_ampl,1,dims=1)
            input_ampl_down = torch.roll(expanded_inputs_ampl,-1,dims=1)

            #output_mask has zeros where value of next angular momentum is impossible, one otherwise
            output_mask = torch.zeros_like(expanded_inputs_ampl)
            output_mask = torch.max(input_ampl_down, input_ampl_up)

            #undo the padding with zeros
            output_mask = output_mask[:,1:-1]

            #in this case, next angular momentum can only go down when current angular momentum is equal to number of sites until end of chain is reached: adjust the mask appropriately
            if self.N - n < self.inputdim:
                output_mask_mask = torch.cat( (torch.ones(self.numsamples, self.N - n, device = device), torch.zeros(self.numsamples, self.inputdim - (self.N - n), device=device)), dim=1)
                output_mask = output_mask*output_mask_mask

            #use mask to only leave valid probabilities for next state
            output_ampl = output_ampl*output_mask
            output_ampl = torch.nn.functional.normalize(output_ampl, eps = 1e-30)

            # store amplitude and phase of marginal probability amplitude
            amplitudes[:, n,  0] = (output_ampl * one_hot_samples[:,n]).sum(dim = 1)
            amplitudes[:, n,  1] = (output_phase * one_hot_samples[:,n]).sum(dim = 1)

            inputs= one_hot_samples[:,n]

            inputs_ampl = inputs

        tot_ampl = torch.zeros(self.numsamples, 2, device=device, dtype=torch.float32)
        tot_ampl[:, 0] = amplitudes[:,:,0].prod(dim = 1)
        tot_ampl[:, 1] = amplitudes[:,:,1].sum(dim = 1)
        # print('amplitude: ', tot_ampl[:,0])
        # print('phase: ', tot_ampl[:,1])
        self.amplitudes = tot_ampl
        return self.amplitudes
