from BP_fromScratch import Dense, Input, Preprocessing, Metrics, Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Sequential:

    def __init__(self):
        self.sequence = []

    def add(self, layer):
        self.sequence.append(layer)

    def compile(self, set_adam, eta):
        self.small_delta = [0 for i in range(len(self.sequence)-1)]
        self.set_adam = set_adam

        for index in range(1,len(self.sequence)):
            self.sequence[index].build(self.sequence[index-1].size, set_adam, eta)

    def normalize_array(self, arr):
        arr = np.round(arr, 10)
        arr[arr > 1] = 1
        arr[arr < -1] = -1
        return arr

    def feed(self, inputs, layer_index):
        outputs = self.sequence[1].call_act(inputs)
        for index in range(2, layer_index):
            outputs = self.sequence[index].call_act(outputs)
        return outputs

    def feed_memory(self, inputs, layer_index):
        out_s = []
        out_h = []
        out_s.append(self.sequence[1].call(inputs))
        out_h.append(self.sequence[1].call_act(inputs))
        for index in range(2, layer_index):
            out_s.append(self.sequence[index].call(out_h[index-2]))
            out_h.append(self.sequence[index].call_act(out_h[index-2]))
        return (out_s, out_h)
        
    def predict(self, inputs):
        pred = []
        for input in inputs:
            pred.append(self.feed(input, len(self.sequence)))
        return pred

    def fit(self, inputs, train_labels, eta, num_epochs, batch_size, loss_type, plot, clip_grad, validation_set):
        if plot!=None:
            Loss=np.zeros(num_epochs)
            Acc=[]
            Pre=[]
            Rec=[]
            val_Acc=None
            if validation_set!=None:
                val_Acc=[]

        time = 1 # for Adam optimizer
        if loss_type == "CategoricalCrossEntropy":
            targets = Preprocessing.encode(train_labels, self.sequence[-1].units) # label encoding
        elif loss_type == "MSE":
            if inputs.size == train_labels.size:
                targets = train_labels
            else:
                targets = Preprocessing.encode(train_labels, self.sequence[-1].units) ##### replace by train_labels if not classification problem   
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=inputs[:,0].size, position=0, leave=True, desc="Epoch "+str(epoch+1)+"/"+str(num_epochs))
            if batch_size == None:
                for batch, target in zip(inputs, targets):
                    loss = self.epoch(batch, target, eta, loss_type, clip_grad, time)
                    time+=1
                    progress_bar.update(1)
                    if plot!=None:
                        Loss[epoch] = Loss[epoch] + loss/inputs[:,0].size

                progress_bar.close()
                print("Epoch ", epoch+1, " completed")
            
                if plot!=None:
                    if inputs.size != train_labels.size:
                        acc, pre, rec, f1 = Metrics.metrics(train_labels, [np.argmax(arr) for arr in self.predict(inputs)])
                        Acc.append(acc)
                        Pre.append(pre)
                        Rec.append(rec)

                    if validation_set != None:
                        if inputs.size != train_labels.size:
                            acc, pre, rec, f1 = Metrics.metrics(validation_set[1], [np.argmax(arr) for arr in self.predict(validation_set[0])])
                            val_Acc.append(acc)
            else:
                shuffle_index = np.random.permutation(batch_size)
                examples = inputs[shuffle_index, :]
                labels = targets[shuffle_index]
                for batch, target in zip(examples, labels):
                    loss = self.epoch(batch, target, eta, loss_type, clip_grad, time)
                    time+=1
                    progress_bar.update(1)
                    if plot!=None:
                        Loss[epoch] = Loss[epoch] + loss/examples[:,0].size

                progress_bar.close()
                print("Epoch ", epoch+1, " completed")
            
                if plot!=None:    
                    if inputs.size != train_labels.size:
                        acc, pre, rec, f1 = Metrics.metrics(train_labels, [np.argmax(arr) for arr in self.predict(inputs)])
                        Acc.append(acc)
                        Pre.append(pre)
                        Rec.append(rec)

                    if validation_set != None:
                        if inputs.size != train_labels.size:
                            acc, pre, rec, f1 = Metrics.metrics(validation_set[1], [np.argmax(arr) for arr in self.predict(validation_set[0])])
                            val_Acc.append(acc)
        
        Metrics.plot_metrics(Acc, Loss)
        Metrics.plot_error_frac(Acc, val_Acc)

    def epoch(self, inputs, targets, eta, loss_type, clip_grad, time): 
        
        assert(len(self.sequence)>1) # layer 0 refers to input layer
        layer = (self.sequence[-1])
        out = self.feed_memory(inputs, len(self.sequence)) # [out_s, out_h]
        out_s = out[0][len(self.sequence)-2]
        out_h = out[1][len(self.sequence)-2]
        if len(self.sequence) > 2:                                     # all 1 x units
            in_x = out[1][len(self.sequence)-3]
        elif len(self.sequence) == 2:
            in_x = inputs

        if loss_type == "MSE":
            errors = np.array(out_h-targets)
            if errors.size > 1:
                loss = np.matmul(errors, np.transpose(errors))/2
            else:
                loss = np.square(errors)/2

            self.small_delta[len(self.sequence)-2] = np.multiply(layer.activation_deriv(out_s),errors)
            self.small_delta[len(self.sequence)-2] = (self.small_delta[len(self.sequence)-2]).reshape((self.small_delta[len(self.sequence)-2]).size, 1) # transposed reshape
                                                            # f'(s_j)*e_j  element wise multiplication
            delta = self.small_delta[len(self.sequence)-2]

            if delta.size == 1:
                if self.set_adam:
                    layer.adam.update(t=time, layer=layer, dw=np.transpose(delta*in_x), db=delta*np.ones((layer.b).size))
                else:
                    layer.w = layer.w - (eta)*np.clip(np.transpose(delta*in_x), -clip_grad, clip_grad)
                    layer.b = layer.b - (eta)*np.clip(delta*np.ones((layer.b).size), -clip_grad, clip_grad)
            else:
                if self.set_adam:
                    layer.adam.update(t=time, layer=layer, dw=np.transpose(np.outer(delta, in_x)), db=np.transpose(delta))
                else:
                    layer.w = layer.w - (eta)*np.clip(np.transpose(np.outer(delta, in_x)), -clip_grad, clip_grad)
                    layer.b = layer.b - (eta)*np.clip(np.transpose(delta), -clip_grad, clip_grad)


        elif loss_type == "CategoricalCrossEntropy":

            p = np.matmul(out_h, np.transpose(targets))
            if p < 0.00001:
                p = 0.00001 # to avoid 0 division
            loss = -np.log(p) 

            self.small_delta[len(self.sequence)-2] = -(1 - p)*targets
            delta = self.small_delta[len(self.sequence)-2]
            
            layer.w = layer.w - np.clip((eta)*np.outer(np.transpose(in_x), delta), -clip_grad, clip_grad)
            layer.b = layer.b - np.clip((eta)*delta, -clip_grad, clip_grad)

        layer.w = np.clip(layer.w, -1, 1)
        layer.b = np.clip(layer.b, -1, 1)
                                                    
        for layer_index in reversed(range(1,len(self.sequence)-1)):

            layer = self.sequence[layer_index]
            layer_above = self.sequence[layer_index+1]
            out_s = out[0][layer_index-1]                 # out, in , small delta shifted of -1 / layer sequence
            if layer_index>1: 
                in_x = out[1][layer_index-2]
            else:
                in_x = inputs    
            
            if loss_type == "MSE":
                if self.small_delta[layer_index].size == 1:
                    self.small_delta[layer_index-1] = np.multiply(layer.activation_deriv(out_s).reshape(layer.activation_deriv(out_s).size, 1), layer_above.w*self.small_delta[layer_index])
                else:
                    self.small_delta[layer_index-1] = np.multiply(layer.activation_deriv(out_s).reshape(layer.activation_deriv(out_s).size, 1),np.matmul(layer_above.w,self.small_delta[layer_index]))
            
            elif loss_type == "CategoricalCrossEntropy":
                #print(np.max(layer.call(inputs)) == np.max(out_s))
                #print("max w ", np.max(layer.w))
                #print("max out_s", np.max(out_s))
                #print("max layer.activation_deriv(out_s/np.max(out_s)) ", np.max(layer.activation_deriv(out_s)))
                self.small_delta[layer_index-1] = np.multiply(layer.activation_deriv(out_s).reshape(layer.activation_deriv(out_s).size, 1),np.matmul(layer_above.w,self.small_delta[layer_index].reshape(self.small_delta[layer_index].size, 1)))

            self.small_delta[layer_index-1] = (self.small_delta[layer_index-1]).reshape((self.small_delta[layer_index-1]).size, 1) # transposed reshape

            delta = self.small_delta[layer_index-1]
            #print("max delta ", np.max(delta))
            #print("max - (eta)*np.transpose(delta*in_x), ", np.max(- (eta)*np.transpose(delta*in_x)))
            if delta.size == 1:
                if self.set_adam:
                    layer.adam.update(t=time, layer=layer, dw=np.transpose(delta*in_x), db=delta*np.ones((layer.b).size))
                else:
                    layer.w = layer.w - (eta)*np.clip(np.transpose(delta*in_x), -clip_grad, clip_grad)
                    layer.b = layer.b - (eta)*np.clip((delta*np.ones((layer.b).size)), -clip_grad, clip_grad)
            else:
                if self.set_adam:
                    layer.adam.update(t=time, layer=layer, dw=np.transpose(np.outer(delta, in_x)), db=np.transpose(delta))
                else:
                    layer.w = layer.w - (eta)*np.clip(np.transpose(np.outer(delta, in_x)), -clip_grad, clip_grad)
                    layer.b = layer.b - (eta)*np.clip(np.transpose(delta), -clip_grad, clip_grad)

            layer.w = np.clip(layer.w, -1, 1)
            layer.b = np.clip(layer.b, -1, 1)

        return loss

    def save(layer_index):
        mat = self.sequence[layer_index].w
        np.savetxt(f"layer{layer_index}", mat)

    def load(layer_index):
        mat = np.loadtxt(f"layer{layer_index}")
        return mat

    def LMSfit(self, inputs, train_labels, eta, num_epochs, batch_size, loss_type, plot, clip_grad, validation_set):
        if plot!=None:
            Loss=np.zeros(num_epochs)
            Acc=[]
            Pre=[]
            Rec=[]
            val_Acc=None
            if validation_set!=None:
                val_Acc=[]

        time = 1 # for Adam optimizer
        if loss_type == "CategoricalCrossEntropy":
            targets = Preprocessing.encode(train_labels, self.sequence[-1].units) # label encoding
        elif loss_type == "MSE":
            if inputs.size == train_labels.size:
                targets = train_labels
            else:
                targets = Preprocessing.encode(train_labels, self.sequence[-1].units) ##### replace by train_labels if not classification problem   
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=inputs[:,0].size, position=0, leave=True, desc="Epoch "+str(epoch+1)+"/"+str(num_epochs))
            if batch_size == None:
                for batch, target in zip(inputs, targets):
                    loss = self.LMSepoch(batch, target, eta, loss_type, clip_grad, time)
                    time+=1
                    progress_bar.update(1)
                    if plot!=None:
                        Loss[epoch] = Loss[epoch] + loss/inputs[:,0].size

                progress_bar.close()
                print("Epoch ", epoch+1, " completed")
            
                if plot!=None:
                    if inputs.size != train_labels.size:
                        acc, pre, rec, f1 = Metrics.metrics(train_labels, [np.argmax(arr) for arr in self.predict(inputs)])
                        Acc.append(acc)
                        Pre.append(pre)
                        Rec.append(rec)

                    if validation_set != None:
                        if inputs.size != train_labels.size:
                            acc, pre, rec, f1 = Metrics.metrics(validation_set[1], [np.argmax(arr) for arr in self.predict(validation_set[0])])
                            val_Acc.append(acc)
            else:
                shuffle_index = np.random.permutation(batch_size)
                examples = inputs[shuffle_index, :]
                labels = targets[shuffle_index]
                for batch, target in zip(examples, labels):
                    loss = self.LMSepoch(batch, target, eta, loss_type, clip_grad, time)
                    time+=1
                    progress_bar.update(1)
                    if plot!=None:
                        Loss[epoch] = Loss[epoch] + loss/examples[:,0].size

                progress_bar.close()
                print("Epoch ", epoch+1, " completed")
            
                if plot!=None:    
                    if inputs.size != train_labels.size:
                        acc, pre, rec, f1 = Metrics.metrics(train_labels, [np.argmax(arr) for arr in self.predict(inputs)])
                        Acc.append(acc)
                        Pre.append(pre)
                        Rec.append(rec)

                    if validation_set != None:
                        if inputs.size != train_labels.size:
                            acc, pre, rec, f1 = Metrics.metrics(validation_set[1], [np.argmax(arr) for arr in self.predict(validation_set[0])])
                            val_Acc.append(acc)
        
        Metrics.plot_metrics(Acc, Loss)
        Metrics.plot_error_frac(Acc, val_Acc)


    def LMSepoch(self, inputs, targets, eta, loss_type, clip_grad, time): 
        
        assert(len(self.sequence)>1) # layer 0 refers to input layer
        layer = (self.sequence[-1])
        out = self.feed_memory(inputs, len(self.sequence)) # [out_s, out_h]
        out_s = out[0][len(self.sequence)-2]
        out_h = out[1][len(self.sequence)-2]
        if len(self.sequence) > 2:                                     # all 1 x units
            in_x = out[1][len(self.sequence)-3]
        elif len(self.sequence) == 2:
            in_x = inputs

        if loss_type == "MSE":
            errors = np.array(out_h-targets)
            if errors.size > 1:
                loss = np.matmul(errors, np.transpose(errors))/2
            else:
                loss = np.square(errors)/2

            self.small_delta[len(self.sequence)-2] = np.multiply(layer.activation_deriv(out_s),errors)
            self.small_delta[len(self.sequence)-2] = (self.small_delta[len(self.sequence)-2]).reshape((self.small_delta[len(self.sequence)-2]).size, 1) # transposed reshape
                                                            # f'(s_j)*e_j  element wise multiplication
            delta = self.small_delta[len(self.sequence)-2]

            if delta.size == 1:
                if self.set_adam:
                    layer.adam.update(t=time, layer=layer, dw=np.transpose(delta*in_x), db=delta*np.ones((layer.b).size))
                else:
                    layer.w = layer.w - (eta)*np.clip(np.transpose(delta*in_x), -clip_grad, clip_grad)
                    layer.b = layer.b - (eta)*np.clip(delta*np.ones((layer.b).size), -clip_grad, clip_grad)
            else:
                if self.set_adam:
                    layer.adam.update(t=time, layer=layer, dw=np.transpose(np.outer(delta, in_x)), db=np.transpose(delta))
                else:
                    layer.w = layer.w - (eta)*np.clip(np.transpose(np.outer(delta, in_x)), -clip_grad, clip_grad)
                    layer.b = layer.b - (eta)*np.clip(np.transpose(delta), -clip_grad, clip_grad)


        elif loss_type == "CategoricalCrossEntropy":

            p = np.matmul(out_h, np.transpose(targets))
            if p < 0.00001:
                p = 0.00001 # to avoid 0 division
            loss = -np.log(p) 

            self.small_delta[len(self.sequence)-2] = -(1 - p)*targets
            delta = self.small_delta[len(self.sequence)-2]
            
            layer.w = layer.w - np.clip((eta)*np.outer(np.transpose(in_x), delta), -clip_grad, clip_grad)
            layer.b = layer.b - np.clip((eta)*delta, -clip_grad, clip_grad)

        layer.w = np.clip(layer.w, -1, 1)
        layer.b = np.clip(layer.b, -1, 1)

        return loss



                        
                    








