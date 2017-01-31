import argparse

import numpy as np
import matplotlib.pyplot as plt

# from ep3 import Genet

# producing network / generator / autonomous
class Genet(object):
    def __init__(self, modelsize = 2, state_dim = 2, M = None):
        self.modelsize = modelsize
        self.state_dim = state_dim + 1 # bias

        self.networks = {
            "fast": {},
            }

        netw = self.networks["fast"]
        
        # network state update
        if M is not None:
            netw["M"] = M.copy()
        else:
            netw["M"] = np.random.uniform(-1e-1, 1e-1, (self.modelsize, self.modelsize)) * 5.0
            
        # network state
        netw["x"] = np.random.uniform(-1e-2, 1e-2, (netw["M"].shape[0], 1))
        
        # transfer func
        self.nonlin = np.tanh

        # leak factor
        self.leak = 0.5

    def step(self):
        netw = self.networks["fast"]
        netw["x"] = self.leak * netw["x"] + (1 - self.leak) * (np.dot(netw["M"], netw["x"]))
        netw["x"] = self.nonlin(netw["x"]) + np.random.normal(0, 1e-3, netw["x"].shape)
        
class GenetPlast(Genet):
    def __init__(self, modelsize = 2, state_dim = 2, M = None):
        Genet.__init__(self, modelsize = modelsize, state_dim = state_dim, M = None)

        self.state_dim = state_dim + 1
        
        # define two networks, one coding the fast activation dynamics,
        # the other coding the slow modulatory dynamics
        self.networks = {
            "fast": {},
            "slow": {},
            }

        # fast network timescale
        self.networks["fast"]["tau"] = 0.8
        # fast network state
        self.networks["fast"]["x"] = np.random.uniform(-1e-2, 1e-2, (self.state_dim, 1))
        # fast network transition matrix
        self.networks["fast"]["M"] = np.random.uniform(-1e-2, 1e-2, (self.state_dim, self.state_dim))

        # fast network timescale
        self.networks["slow"]["tau"] = 0.96
        # slow network state dim
        self.networks["slow"]["s_dim"] = np.prod(self.networks["fast"]["M"].shape)
        # slow network input dim
        self.networks["slow"]["i_dim"] = self.networks["slow"]["s_dim"] + self.state_dim
        # slow network transition matrix
        if M is None:
            self.networks["slow"]["M"] = np.random.uniform(-1, 1, (self.networks["slow"]["s_dim"], self.networks["slow"]["i_dim"])) * 0.5
        else:
            self.networks["slow"]["M"] = M
            
    def step(self):
        netw = self.networks["fast"]
        netw["x"] = netw["tau"] * netw["x"] + (1 - netw["tau"]) * np.dot(netw["M"], netw["x"])

        # bias
        netw["x"][0,0] = 10.0

        # bound
        netw["x"] = np.tanh(netw["x"])

        netw["x"] += np.random.normal(0.0, 1e-2, netw["x"].shape)
        
        netw_s = self.networks["slow"]
        # print "netw M .shape", netw_s["M"].shape, netw_s["M"]
        # netw["M"] =
        Xx = np.vstack((netw["x"], netw["M"].reshape((-1, 1))))
        # print "Xx,shape", Xx.shape
        upd = np.dot(netw_s["M"], Xx).reshape(netw["M"].shape)
        # print "upd.shape", upd.shape, upd
        fullupd = (netw_s["tau"] * netw["M"] + (1 - netw_s["tau"]) * upd)
        netw["M"] = np.clip(fullupd, -3, 3) # np.tanh(fullupd) * 3.0
        # print "A", A.shape

def get_log(g, numiterations):
    # prepare logging
    log = {}
    for k1, v1 in g.networks.items():
        log[k1] = {}
        for k2, v2 in v1.items():
            # print "type(v2)", type(v2)
            if type(v2) is float:
                # print "is float"
                log[k1][k2] = v2
            elif type(v2) is np.ndarray:
                log[k1][k2] = np.zeros((numiterations, ) + v2.shape)
    return log
                
        
def main(args):

    # init evo / opt process
    numgenerations = 1
    numindividuals = 1
    numiterations = args.numsteps

    generations = []
    newgen = []
    for i in range(numindividuals):
        # g = GenetPlast()
        # print "g", g
        newgen.append([])

    for g in range(numgenerations):
        for i in range(numindividuals):
            g = GenetPlast()
            log = get_log(g, numiterations)
            for t in range(numiterations):
                g.step()
                print "g.x", g.networks["fast"]["x"]
                log["fast"]["x"][t] = g.networks["fast"]["x"]
                log["fast"]["M"][t] = g.networks["fast"]["M"]

            plt.subplot(211)
            plt.plot(log["fast"]["x"][:,:,0])
            plt.subplot(212)
            plt.plot(log["fast"]["M"].reshape((log["fast"]["M"].shape[0], log["fast"]["M"].shape[1] * log["fast"]["M"].shape[2])))
            plt.show()
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numsteps", type=int, default=1000,
                        help="number of timesteps for individual evaluation [1000]")

    args = parser.parse_args()
     
    main(args)
