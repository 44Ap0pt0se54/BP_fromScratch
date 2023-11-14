import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

class SOFM:

    def __init__(self, m, n, dim):

        self.n = n
        self.m = m
        self.dim = dim

    def build(self):
        # create random normal unit weights
        random_vector = np.random.uniform(low=0.0, high=1.0, size=self.dim)
        self.weights_mat = np.array([random_vector/np.linalg.norm(random_vector) for i in range(self.m*self.n)])

    def predict(self, input, batch_size):
        if batch_size == 1:
            s = np.matmul(self.weights_mat, np.transpose(input))
            s_win = np.zeros(self.n*self.m)
            s_win[np.argmax(s)] = 1
            return s_win
        else:
            s = np.matmul(self.weights_mat, np.transpose(input))
            s_win = np.zeros((self.n*self.m, batch_size))
            s_win[[np.argmax(s[:,j]) for j in range(batch_size)],:] = 1
            return s_win

    def fit(self, inputs, eta_0, num_epochs, batch_size, plot):
        inputs = self.normalize(inputs, batch_size)
        time = 0
        
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=inputs[:,0].size, position=0, leave=True, desc="Epoch "+str(epoch+1)+"/"+str(num_epochs))
            if batch_size == None:
                self.epoch(inputs, eta_0, time, inputs[:,0].size, progress_bar)
                time+=1   
                       
            else:
                shuffle_index = np.random.permutation(batch_size)
                batch = inputs[shuffle_index, :]
            
                self.epoch(batch, eta_0, time, batch_size, progress_bar)
                time+=1
                        
            progress_bar.close()
            print("Epoch ", epoch+1, " completed")
        
        if plot!=None:
            output_mat = np.zeros(self.n*self.m)
            for x in inputs:
                output_mat = output_mat + self.predict(x, 1)

            output_mat = output_mat/inputs[:,0].size
            plt.imshow(output_mat.reshape(self.n, self.m), cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title('Activation Map')
            plt.show()
            

    def epoch(self, batch, eta_0, time, batch_size, progress_bar):

        s_win = self.predict(batch, batch_size)
        DELTA = np.array([self.Lambda(i, np.argmax(s_win[:,j]), time) for i in range(self.n*self.m) for j in range(batch_size)]).reshape(self.n*self.m, batch_size)
        j = 0
        for x in batch:
            X = np.transpose(np.outer(x, np.ones(self.n*self.m)))
            delta = (np.outer(DELTA[:,j], np.ones(self.dim))).reshape(self.n*self.m, self.dim)
            self.weights_mat = self.weights_mat + self.eta(eta_0, time)*np.multiply(delta, X-self.weights_mat)
            j+=1
            progress_bar.update(1) 

    def Lambda(self, i, i_star, t):

        diff = np.array([i//self.n, i%self.n])-np.array([i_star//self.n, i_star%self.n])
        return np.exp(-np.matmul(diff, np.transpose(diff))/(2*self.sigma(t)*self.sigma(t)))

    def sigma(self, t):

        sigma_0 = (self.n+self.m)/4
        tau_n = 1000/np.log(sigma_0)
        return sigma_0*np.exp(-t/tau_n)

    def eta(self, eta_0, t):
        tau_l = 1000
        return eta_0*np.exp(-t/tau_l)

    def normalize(self, input, batch_size):
        if batch_size == 1:
            return input/np.linalg.norm(input)
        else:
            input_norm = []
            for x in input:
                input_norm.append(x/np.linalg.norm(x))
            return np.array(input_norm)
