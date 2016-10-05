# trying to
# create simple (few neurons) recurrent neural networks and evolve / optimize
# their weights towards a "complexity" objective: ES, CMA-ES, hyperopt


# TODO
#  - different esstimators: kernel, kraskov, ...
#  - different measures: TE / AIS / literature
#  - ES / HyperNeat / CMA-ES / hyperopt
#  - 

# FIXME:
#  1 - check base network dynamics
#  2 - average over multiple runs per individual
#  found error: newgen wasn't properly used but overwritten by random configuration

import cPickle, time, argparse
from functools import partial
import numpy as np
import pylab as pl

from jpype import startJVM, isJVMStarted, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble, attachThreadToJVM

from smp.infth import init_jpype, ComplexityMeas

class ComplexityMeasure(object):
    def __init__(self):
        init_jpype()
        self.piCalcClass = JPackage("infodynamics.measures.continuous.kraskov").PredictiveInfoCalculatorKraskov
        self.piCalc = self.piCalcClass()
        self.piCalc.setProperty("NORMALISE", "true"); # Normalise the individual var
        self.tau = 1
    # loss measure complexity
    def compute_pi(self, X):
        k = 10
        self.piCalc.initialise(k, self.tau)        
        # self.piCalc.setObservations(X.reshape((X.shape[0],)))
        pi_avg = 0.0
        for d in [0]: # range(X.shape[1]):
            self.piCalc.setObservations(X[:,d])
            pi_avg += self.piCalc.computeAverageLocalOfObservations();
        return pi_avg

# producing network / generator / autonomous
class Genet(object):
    def __init__(self, modelsize = 2, state_dim = 2, M = None):
        self.modelsize = modelsize
        self.state_dim = state_dim
        # network state update
        if M is not None:
            self.M = M.copy()
        else:
            self.M = np.random.uniform(-1e-1, 1e-1, (modelsize, modelsize)) * 5.0
        # network state
        self.x = np.random.uniform(-1e-2, 1e-2, (self.M.shape[0], 1))
        # transfer func
        self.nonlin = np.tanh

        # leak factor
        self.leak = 0.5

    def step(self):
        self.x = self.leak * self.x + (1 - self.leak) * (np.dot(self.M, self.x))
        self.x = self.nonlin(self.x) + np.random.normal(0, 1e-3, self.x.shape)
        # print self.x.shape

def test_ind(M = None):
    if M is not None:
        M = M.copy()
    else:
        M = np.array([[ 0.69656214,  0.33208246],
                  [ 0.41712419,  0.56571223]])
        M = np.array([[ 0.69656214,  0.33208246],
                [ 0.41712419,  0.56571223]])
        M = np.array([[ 0.69656214,  0.33208246],
                [ 0.41712419,  0.56571223]])
        # M = np.array([[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]])
        # M = np.array([[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]])
        # M = np.array([[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]])
        # M = np.array([[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]])
        # M = np.array([[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]])
        
    n = Genet(M = M)

    numsteps = 2000
        
    Xs = np.zeros((numsteps, n.state_dim))
    # loop over timesteps
    for i in range(numsteps):
        # x = np.dot(M, x)
        n.step()
        Xs[i] = n.x.reshape((n.state_dim,))
        
    pl.subplot(311)
    pl.plot(Xs[:,0], Xs[:,1], "k-o", alpha=0.1)
    pl.gca().set_aspect(1)
    # pl.yscale("log")
    # pl.xscale("log")
    pl.subplot(312)
    pl.plot(Xs[:,0], "k-,", alpha=0.33)
    pl.subplot(313)
    pl.plot(Xs[:,1], "k-,", alpha=0.33)
    pl.show()

def objective(params):
    """evaluate an individual (parameter set) with respect to given objective"""
    numsteps = params["numsteps"]
    n = Genet(M = params["M"])
    cm = params["measure"]
    # a dict containg network config, timeseries, loss
    experiment = dict()
    #  create network
    # n = Genet(2, 2)
    # state trajectory
    Xs = np.zeros((numsteps, n.state_dim))
    # loop over timesteps
    for i in range(numsteps):
        # x = np.dot(M, x)
        n.step()
        Xs[i] = n.x.reshape((n.state_dim,))
    # return structure: params, timeseries, scalar loss
    experiment = {
        "M": n.M,
        "timeseries": Xs.copy(),
        # "loss": np.var(Xs),
        "loss": cm.compute_pi(Xs) # compute_complexity(Xs)
    }
    return experiment

def main(args):
    """main, dispatch mode"""
    if args.mode == "es_vanilla":
        main_es_vanilla(args)
    elif args.mode == "cma_es":
        main_cma_es()
    elif args.mode == "hp_tpe":
        main_hp_tpe()
    elif args.mode == "hp_random_search":
        main_hp_random_search()
    elif args.mode == "hp_gp_ucb":
        main_hp_gp_ucb()
    elif args.mode == "hp_gp_ei":
        main_hp_gp_ei()

def main_cma_es(args):
    import cma
    print "args", args
    options = {'CMA_diagonal':100, 'seed':1234, 'verb_time':0, "maxfevals": 2000}
    func = objective # args["func"]
    # arguments: function, initial params, initial var, options
    # res = cma.fmin(cma.fcts.griewank, [0.1] * 10, 0.5, options)
    res = cma.fmin(func, [0.5] * args["dim"], 0.3, options)

    print res[0], res[1]
    return res[0]
    
    
def main_es_vanilla(args):
    # experiment signature
    expsig = time.strftime("%Y%m%d-%H%M%S")
    # evolution / opt params
    numgenerations = 200
    numpopulation = 20
    numsteps = 1000

    # generations array containing
    generations = []
    
    cm = ComplexityMeasure()

    # array of parameter ndarray for the current generation
    newgen = []
    for j in range(numpopulation):
        n = Genet(2, 2)
        newgen.append(n.M)
    
    pl.ion()
    # loop over generations
    for k in range(numgenerations):
        # arrays of individuals each element of which is
        population = dict()
        # loop over population
        for j in range(numpopulation):
            params = {
                "M": newgen[j],
                "numsteps": numsteps,
                "measure": cm
                }
            experiment = objective(params)
            # population.append(experiment)
            population["%d" % j] = experiment

            # print "M", n.M
            
            # pl.subplot(411)
            # pl.plot(Xs[:,0], Xs[:,1], "k-o", alpha=0.1)
            # pl.gca().set_aspect(1)
            # # pl.yscale("log")
            # # pl.xscale("log")
            # pl.subplot(412)
            # pl.plot(Xs[:,0], "k-.", alpha=0.33)
            # pl.subplot(413)
            # pl.plot(Xs[:,1], "k-.", alpha=0.33)
            # pl.subplot(414)
            # pl.gca().clear()
            # pl.plot([ind["loss"] for ind in population.values()], "k-x", alpha=0.1)
            # pl.draw()
            # pl.pause(0.001)

        # pl.cla()

        ind_loss = np.array([ind["loss"] for ind in population.values()])
        avg_fit = np.mean(ind_loss)
        std_fit = np.std(ind_loss)
        max_fit = np.max(ind_loss)
        min_fit = np.min(ind_loss)
        print "gen %04d: max fit = %f, avg/std fit = %f/%f, min fit = %f, " % (k, max_fit, avg_fit, std_fit, min_fit)

        generations.append(population)

        # save experiment progress
        cPickle.dump(generations, open("ep3/ep3_generations_%s.bin" % (expsig), "wb"))

        # generate new generation from loss sorted current generation
        # get best n individuals (FIXME: use a dataframe)
        # import operator
        # print generations[-1].items()
        # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
        # sorted_x = sorted(generations[-1].items(), key=operator.itemgetter(1))
        sorted_x = sorted(generations[-1].items(), key=lambda x: x[1]["loss"], reverse=True)

        # print sorted_x[0][1]
        newgen[0] = sorted_x[0][1]["M"]
        for i in range(1, numpopulation):
            c = np.random.choice(10)# make tournament or something
            newgen[i] = sorted_x[c][1]["M"]
            # mutate
            if np.random.uniform() < 0.05:
                mut_idx = np.random.choice(np.prod(newgen[i].shape))
                # print "mut_idx", mut_idx
                tmp_s = newgen[i].shape
                tmp = newgen[i].flatten()
                tmp[mut_idx] += np.random.normal(0, 0.1)
                newgen[i] = tmp.copy().reshape(tmp_s)
                
            # # crossover
            # if np.random.uniform() < 0.01:
            #     mut_idx = np.random.choice(np.prod(newgen[i].shape))
            #     print "mut_idx", mut_idx
            #     tmp_s = newgen[i].shape
            #     tmp = newgen[i].flatten()
            #     tmp[mut_idx] += np.random.normal(0, 0.05)
            #     newgen[i] = tmp.reshape(tmp_s)
                
        # print "sorted_x", sorted_x
        
    # print "generations", generations[-1]
    sorted_x = sorted(generations[-1].items(), key=lambda x: x[1]["loss"], reverse=True)
    # for ind in generations[-1].values():
    for i,ind in enumerate(sorted_x):
        if i < 5:
            test_ind(ind[1]["M"])
        print "last generation fit/M", ind[1]["loss"], ind[1]["M"]
    pl.ioff()
    pl.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="es_vanilla",
                        help="optimization / search mode [es_vanilla], (es_vanilla, cma_es, hp_tpe, hp_random_search, hp_gp_ucb, hp_gp_ei)")
    parser.add_argument("-n", "--numsteps", type=int, default=1000,
                        help="number of timesteps for individual evaluation [1000]")
    args = parser.parse_args()
    main(args)
    # test_ind(None)
