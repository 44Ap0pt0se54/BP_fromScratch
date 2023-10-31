from BP_fromScratch import Dense, Input, Preprocessing, Metrics
import numpy as np
import matplotlib.pyplot as plt

class Sequential:

    def __init__(self):
        self.sequence = []

    def add(self, layer):
        self.sequence.append(layer)

    def compile(self):
        self.small_delta = [0 for i in range(len(self.sequence)-1)]

        for index in range(1,len(self.sequence)):
            self.sequence[index].build(self.sequence[index-1].size)

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

    def fit(self, inputs, train_labels, eta, num_epochs, batch_size, loss_type, plot, clip_grad):
        if plot!=None:
            Loss=np.zeros(num_epochs)
            Acc=[]
            Pre=[]
            Rec=[]

        if loss_type == "CategoricalCrossEntropy":
            targets = encode(train_labels, self.sequence[-1].units) # label encoding
        elif loss_type == "MSE":
            targets = train_labels
        for epoch in range(num_epochs):
            if batch_size == None:
                for batch, target in zip(inputs, targets):
                    loss = self.epoch(batch, target, eta, loss_type, clip_grad)
                    if plot!=None:
                        Loss[epoch] = Loss[epoch] + loss
            
                if plot!=None:        
                    acc, pre, rec, f1 = metrics(train_labels, [np.argmax(arr) for arr in self.predict(inputs)])
                    Acc.append(acc)
                    Pre.append(pre)
                    Rec.append(rec)
                print("epoch ", epoch, " passed")

            else:
                shuffle_index = np.random.permutation(batch_size)
                examples = inputs[shuffle_index, :]
                labels = targets[shuffle_index]
                for batch, target in zip(examples, labels):
                    self.epoch(batch, target, eta, loss_type, epoch, clip_grad)
        
        plot_metrics(Acc, Loss)


    def epoch(self, inputs, targets, eta, loss_type, clip_grad): 
        
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
            loss = np.square(errors)/2

            self.small_delta[len(self.sequence)-2] = np.multiply(layer.activation_deriv(out_s),errors)
            self.small_delta[len(self.sequence)-2] = (self.small_delta[len(self.sequence)-2]).reshape((self.small_delta[len(self.sequence)-2]).size, 1) # transposed reshape
                                                            # f'(s_j)*e_j  element wise multiplication
            delta = self.small_delta[len(self.sequence)-2]

            if delta.size == 1:
                layer.w = layer.w - (eta)*np.clip(np.transpose(delta*in_x), -clip_grad, clip_grad)
                layer.b = layer.b - (eta)*np.clip(delta*np.ones((layer.b).size), -clip_grad, clip_grad)
            else:
                layer.w = layer.w - (eta)*np.clip(np.transpose(np.matmul(delta, in_x)), -clip_grad, clip_grad)
                layer.b = layer.b - (eta)*np.clip(np.transpose(delta), -clip_grad, clip_grad)


        elif loss_type == "CategoricalCrossEntropy":

            p = np.matmul(out_h, np.transpose(targets))
            if p < 0.00001:
                p = 0.00001 # to avoid 0 division
            loss = -np.log(p) 
            # print("out_h ", out_h)
            # print("targets ", targets)
            # print("proba ", p)
            self.small_delta[len(self.sequence)-2] = -(1 - p)*targets
            delta = self.small_delta[len(self.sequence)-2]
            
            layer.w = layer.w - np.clip((eta)*np.outer(np.transpose(in_x), delta), -clip_grad, clip_grad)
            layer.b = layer.b - np.clip((eta)*delta, -clip_grad, clip_grad)

        layer.w = np.clip(layer.w, -1e1, 1e1)
        layer.b = np.clip(layer.b, -1e1, 1e1)
                                                    
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
                # print("max w ", np.max(layer.w))
                # print("max out_s", np.max(out_s))
                #print("max layer.activation_deriv(out_s/np.max(out_s)) ", np.max(layer.activation_deriv(out_s)))
                self.small_delta[layer_index-1] = np.multiply(layer.activation_deriv(out_s).reshape(layer.activation_deriv(out_s).size, 1),np.matmul(layer_above.w,self.small_delta[layer_index].reshape(self.small_delta[layer_index].size, 1)))

            self.small_delta[layer_index-1] = (self.small_delta[layer_index-1]).reshape((self.small_delta[layer_index-1]).size, 1) # transposed reshape

            delta = self.small_delta[layer_index-1]
            #print("max delta ", np.max(delta))
            #print("max - (eta)*np.transpose(delta*in_x), ", np.max(- (eta)*np.transpose(delta*in_x)))
            if delta.size == 1:
                layer.w = layer.w - (eta)*np.clip(np.transpose(delta*in_x), -clip_grad, clip_grad)
                layer.b = layer.b - (eta)*np.clip((delta*np.ones((layer.b).size)), -clip_grad, clip_grad)
            else:
                layer.w = layer.w - (eta)*np.clip(np.transpose(np.outer(delta, in_x)), -clip_grad, clip_grad)
                layer.b = layer.b - (eta)*np.clip(np.transpose(delta), -clip_grad, clip_grad)

            layer.w = np.clip(layer.w, -1e1, 1e1)
            layer.b = np.clip(layer.b, -1e1, 1e1)

        return loss

                        
                    








