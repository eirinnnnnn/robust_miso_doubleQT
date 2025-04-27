import numpy as np
import scipy.stats as stats

class Para  :
    def __init__(self, Nt, Nr, M, snr, Pt, B=None, rep=5000, snrest = 11, Pin=0.9):
        ## shared env param
        self.Nt = Nt
        self.Nr = Nr
        self.M = M
        self.sigma_w = 1
        # ref A.Pascual-Iserte
        self.r2 = stats.chi2.ppf(Pin, df=2*Nt*Nr)/2
        # print(self.r2)
        # self.signal_pwr = 1
        self.snr = snr
        self.B = np.eye(Nt*Nr)*(1+10**(snrest/10))/self.r2 if B==None else B 
        self.B_inv = np.linalg.pinv(self.B)
        self.Pt = Pt
        self.rep = rep
        self.lamb = 1

        self.snrest = snrest
        ## initialization
        self.h_hat = self._init_h_hat()
        
        self.h = self.delta_H + self.h_hat

    def _init_delta_H (self) :
        # r = np.random.normal(0, np.sqrt(0.5), (dim, 1)) 
        # c = np.random.normal(0, np.sqrt(0.5), (dim, 1)) 
        sigma_e = 10**(-self.snrest/10)
        r = np.random.normal(0, sigma_e/2, ((self.Nr, self.Nt))) 
        c = np.random.normal(0, sigma_e/2, ((self.Nr, self.Nt))) 
        m = r + 1j*c

        # norm2 = m.conj().T @ self.B @ m 
        # m = m / np.sqrt(norm2)
        return m  
    
    def _init_h_hat (self) :
        r = np.random.normal(0, np.sqrt(0.5), (self.Nr, self.Nt)) 
        c = np.random.normal(0, np.sqrt(0.5), (self.Nr, self.Nt)) 
        return r + 1j*c