import numpy as np
class Adam():
    def __init__(self, input_size, unit_size, eta, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = np.zeros(input_size*unit_size).reshape(input_size, unit_size), np.zeros(input_size*unit_size).reshape(input_size, unit_size)
        self.m_db, self.v_db = np.zeros(unit_size), np.zeros(unit_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def update(self, t, layer, dw, db):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        # *** biases *** #
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*np.square(dw)
        # *** biases *** #
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*np.square(db)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)

        ## update weights and biases
        layer.w = layer.w - self.eta*(np.divide(m_dw_corr,np.sqrt(v_dw_corr)+self.epsilon))
        layer.b = layer.b - self.eta*(np.divide(m_db_corr,np.sqrt(v_db_corr)+self.epsilon))