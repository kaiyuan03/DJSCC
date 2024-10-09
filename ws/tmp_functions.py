import torch
from torch import nn
import warnings

def sinr_downlink(W_dict,mode,P,H,p_noise,disp=0):
    # for mode,W in W_dict.items():
    U = len(P)
    W = W_dict[mode]
    if disp:
        print("-------------",mode)
    # compute SINR
    sinr = []
    for u in range(U):
        signal_power = P[u]*(W[:,u]@H[:,u])**2
        interference_power = 0
        for u_ in range(U):
            if u_ == u:
                continue
            interference_power += P[u_]*(W[:,u_]@H[:,u])**2
        if disp:
            print("interference_power: ", interference_power)
        sinr.append(signal_power/(interference_power+p_noise[u]))
    if disp:
        print("sinr(linear): ", sinr)
    sinr_db = torch.log10(torch.Tensor(sinr))
    if disp:
        print("sinr(dB): ", sinr_db)
    return sinr,sinr_db

# Implement a MU-MISO channel layer
class CHN_MU_MISO(nn.Module):
    """
    U: Number of users.
    Nt: Number of transmitter antennas.
    W: Nt x U matrix. Each column is a beamforming vector.
    H: Nt x U matrix. Each column is a channel vector.
    P: N x 1 vector. Each element is a power allocated to a user.
    p_noise: N x 1 vector. Each element is a complex-noise power for a user.
    """
    def __init__(self, W, H, P, p_noise):
        super(CHN_MU_MISO, self).__init__()
        Nt,U = H.shape
        p_noise_ = torch.Tensor(p_noise)/2
        stddev = p_noise_**(0.5)
        self.U = U
        self.Nt = Nt
        self.W = W
        self.H = H
        self.P = P
        self.stddev = stddev
        v = sum(abs(torch.norm(self.W,dim=0)-1))
        if v>1e-5:
            warnings.warn(f"Non-unit beamforming vector: {v}")
    def forward(self, x):
        """
        x: U x Batch x CWH tensor. concatenated input of all users.
        """
        # beamforming vector weighted signals
        # for user 1:
        combined_signal = 0
        for u in range(self.U):
            combined_signal += torch.stack([(self.P[u])**(0.5)*x[u]*self.W[nt,u] for nt in range(self.Nt)])
        # combined_signal: Nt x Batch x CWH
        # print("combined_signal: ", combined_signal.shape)
        y = []
        for u in range(self.U):
            # transmit to u-th user
            received_signal = torch.stack([combined_signal[nt]*self.H[nt,u] for nt in range(self.Nt)])
            received_signal = torch.sum(received_signal,dim=0)
            # print("received_signal: ", received_signal.shape)
            noise = torch.normal(0, self.stddev[u], size=received_signal.shape).to(received_signal.device)
            received_signal += noise
            amp_gain = (self.P[u])**(0.5)*(self.H[:,u]@self.W[:,u]).item()
            decoded_signal = received_signal/amp_gain
            y.append(decoded_signal)
        y = torch.stack(y)
        return y

class RUN_STAT:
    def __init__(self,name=""):
        self.name = name
        self.num = 0
        self.EX = 0
        self.EX2 = 0
    def __call__(self,x):
        self.EX = (self.EX*self.num+x)/(self.num+1)
        self.EX2= (self.EX2*self.num+x**2)/(self.num+1)
        self.num += 1
    def __str__(self):
        var = self.EX2-(self.EX)**2
        if self.name:
            return f"[{self.name:5}]: [N={self.num:5.2E}] [EX: {self.EX:10.3E}] [VarX: {var:10.3E}]"
        else:
            return f"[N={self.num:5.2E}] [EX: {self.EX:10.3E}] [VarX: {var:10.3E}]"