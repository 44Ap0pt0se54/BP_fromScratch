import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SOFM:

    def __init__(self, m, n, dim):

        self.n = n
        self.m = m
        self.dim = dim

    def build(self, weights_init, data):
        # create random normal unit weights
        if weights_init == "uniform":
            self.generate_uniform()
        elif weights_init == "normal":
            self.generate_normal()
        elif weights_init == "PCA":
            self.generate_PCA(data)

    def predict(self, input, batch_size):
        if batch_size == 1:
            s = np.matmul(self.weights_mat, np.transpose(input))
            s_win = np.zeros(self.n*self.m)
            s_win[np.argmax(s)] = 1

        else:
            s = np.matmul(self.weights_mat, np.transpose(input))
            s_win = np.zeros((self.n*self.m, batch_size))
            index = [np.argmax(s[:,j]) for j in range(batch_size)]
            for j in range(len(index)):
                s_win[index[j], j] = 1

        return s_win

    def fit(self, inputs, eta_0, num_epochs, batch_size, plot):
        inputs = self.normalize(inputs, inputs.shape[0])
        time = 0
        
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=inputs[:,0].size, position=0, leave=True, desc="Epoch "+str(epoch+1)+"/"+str(num_epochs))
            if batch_size == None:
                self.epoch(inputs, eta_0, time, inputs[:,0].size, progress_bar, num_epochs)
                time+=1   
                if plot != None:
                    self.plot_activation(inputs, "Global Activity Map")
                       
            else:
                shuffle_index = np.random.permutation(batch_size)
                batch = inputs[shuffle_index, :]
            
                self.epoch(batch, eta_0, time, batch_size, progress_bar, num_epochs)
                time+=1
                if plot != None:
                    self.plot_activation(batch, "Global Activity Map")
                        
            progress_bar.close()
            print("Epoch ", epoch+1, " completed")
        
    def epoch(self, batch, eta_0, time, batch_size, progress_bar, num_epochs):

        s_win = self.predict(batch, batch_size)
        lr = self.eta(eta_0, time, num_epochs)
        DELTA = np.array([self.Lambda(i, np.argmax(s_win[:,j]), time, lr) for i in range(self.n*self.m) for j in range(batch_size)]).reshape(self.n*self.m, batch_size)
        j = 0
        
        for x in batch:
            X = np.transpose(np.outer(x, np.ones(self.n*self.m)))
            delta = (np.outer(DELTA[:,j], np.ones(self.dim))).reshape(self.n*self.m, self.dim)
            self.weights_mat = self.weights_mat + lr*np.multiply(delta, X-self.weights_mat)
            j+=1
            progress_bar.update(1) 

    def Lambda(self, i, i_star, t, lr):

        diff = np.array([i//self.n, i%self.n])-np.array([i_star//self.n, i_star%self.n])
        return np.exp(-np.matmul(diff, np.transpose(diff))/(2*lr**2))

    def sigma(self, t):

        sigma_0 = (self.n+self.m)/4
        tau_n = 1000/np.log(sigma_0)
        return sigma_0*np.exp(-t/tau_n)

    def eta(self, eta_0, t, num_epochs):
        tau_l = num_epochs
        return eta_0*np.exp(-t/tau_l)

    def normalize(self, input, batch_size):
        if batch_size == 1:
            return input / np.linalg.norm(input)
        else:
            input_norm = input / np.linalg.norm(input, axis=1, keepdims=True)
            return input_norm


    def generate_uniform(self):
        self.weights_mat = np.random.uniform(low=-1.0, high=1.0, size=(self.n*self.m, self.dim))
        self.weights_mat /= np.linalg.norm(self.weights_mat, axis=1, keepdims=True)

    def generate_normal(self):
        self.weights_mat = np.random.normal(size=(self.n*self.m, self.dim))
        self.weights_mat /= np.linalg.norm(self.weights_mat, axis=1, keepdims=True)


    def generate_PCA(self, data):
        U, D, Vt = np.linalg.svd(data)
        self.plot_inertia(D)
        #nb_components = input(int("Number of components :"))
        random_vector = Vt[:,0:self.n*self.m]
        self.weights_mat = np.transpose(random_vector)

    def plot_inertia(self, D):
        total_inertia = np.sum(D)
        svd_inertia = []
        index = []
        for i in range(len(D)):
            svd_inertia.append(D[i]/total_inertia)
            index.append(i+1)
        plt.scatter(index,svd_inertia)
        plt.show()

    def plot_activation(self, inputs, title):
        output_mat = np.zeros(self.n * self.m)
        for x in inputs:
            output_mat += self.predict(x, 1)
        output_mat /= inputs.shape[0]
        fig, ax = plt.subplots()
        im = ax.imshow(output_mat.reshape(self.n, self.m), cmap='hot', interpolation='nearest')
        plt.colorbar(im)
        plt.title(title)
        plt.show()

    def save(self, name):
        np.savetxt(f"SOM_saved_{name}", self.weights_mat.T)
