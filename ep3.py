# trying to
# create simple (few neurons) recurrent neural networks and evolve / optimize
# their weights towards a "complexity" objective: ES, CMA-ES, hyperopt

# TODO
#  - sample and plot catalogue of phylogenetic history
#  - different esstimators: kernel, kraskov, ...
#  - different measures: TE / AIS / lyapunov / recurrence plot / literature
#  - ES / CMA-ES / hyperopt
#  - CPPN, HyperNeat and Map syn/mod composite network
#  - ES: compare fitness/generation curves for gaussian and pareto noise
#  - more precise modulation

# FIXME:
#  1 - check base network dynamics: moved to ep4
#  2 - average over multiple runs per individual
#  3 - ensure / encourage diversity: novelty, genotype distance, behaviour distance (overlap of 2d space covered, same without phase information)
#  4 - proper selection, xover, mut operators
# params: which selection, xover points, 
# evolve: tau, tau_s, state_dim
# add additional states to fast/slow combi

# DONE
#  - x complete logging
#   - x experiment config
#   - x genotype
#   - x phenotype (timeseries)
#  found error: newgen wasn't properly used but overwritten by random configuration

from __future__ import print_function

import pickle, time, argparse, os
from functools import partial, reduce
import numpy as np
import pylab as pl
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hyperopt import hp
from hyperopt import STATUS_OK, STATUS_FAIL
from hyperopt import fmin, tpe, Trials, rand, anneal
# import hp_gpsmbo.hpsuggest
try:
    from hp_gpsmbo import suggest_algos
except ImportError:
    print("Couldn't import hp_gpsmbo")

from jpype import startJVM, isJVMStarted, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble, attachThreadToJVM

from smp.infth import init_jpype # , ComplexityMeas

from ep4 import Genet, GenetPlast

# note to self: make easy wrapper for robotics / ML applications
# variants: k, tau, global/local

# from lmjohns3:kohonen/kohonen/kohonen.py
def argsample(pdf, n=1):
    '''Return n indices drawn proportionally from a discrete mass vector.'''
    assert (pdf >= 0).all(), 'cannot sample from %r!' % pdf
    cdf = pdf.cumsum()
    return np.searchsorted(cdf, np.random.uniform(0, cdf[-1], n))

class ComplexityMeasure(object):
    def __init__(self, measure="PI", measure_k = 100, measure_tau = 1, estimator = "kraskov1"):

        self.measure = measure
        self.k   = measure_k
        self.tau = measure_tau
        self.estimator = estimator
        # calc_init_args = None
        
        # Predictive Information (PI) and Active Information Storage (AIS) Calculator
        if self.estimator == "kraskov1":
            alg = 1
            self.piCalcClass = JPackage("infodynamics.measures.continuous.kraskov").PredictiveInfoCalculatorKraskov
            self.aisCalcClass = JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
            self.piCalc = self.piCalcClass(alg)
            self.aisCalc = self.aisCalcClass(alg)
            self.picalc_init  =  self.piCalc.initialise
            self.aiscalc_init  =  self.aisCalc.initialise
        elif self.estimator == "kraskov2":
            alg = 2
            self.piCalcClass = JPackage("infodynamics.measures.continuous.kraskov").PredictiveInfoCalculatorKraskov
            self.aisCalcClass = JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
            self.piCalc = self.piCalcClass(alg)
            self.aisCalc = self.aisCalcClass(alg)
            self.picalc_init  =  self.piCalc.initialise
            self.aiscalc_init  =  self.aisCalc.initialise
        elif self.estimator == "gaussian":
            self.piCalcClass = JPackage("infodynamics.measures.continuous.gaussian").PredictiveInfoCalculatorGaussian
            self.aisCalcClass = JPackage("infodynamics.measures.continuous.gaussian").ActiveInfoStorageCalculatorGaussian
            self.piCalc = self.piCalcClass()
            self.aisCalc = self.aisCalcClass()
            self.picalc_init  =  self.piCalc.initialise
            self.aiscalc_init  =  self.aisCalc.initialise
        elif self.estimator == "kernel":
            calc_init_args = {"epsilon": 0.001}
            self.piCalcClass = JPackage("infodynamics.measures.continuous.kernel").PredictiveInfoCalculatorKernel
            self.aisCalcClass = JPackage("infodynamics.measures.continuous.kernel").ActiveInfoStorageCalculatorKernel
            self.piCalc = self.piCalcClass()
            self.aisCalc = self.aisCalcClass()
            self.picalc_init  =  partial(self.piCalc.initialise, epsilon = 0.001)
            self.aiscalc_init  =  partial(self.aisCalc.initialise, epsilon = 0.001)
            
        self.piCalc.setProperty("NORMALISE", "false"); # Normalise the individual var

        self.aisCalc.setProperty("NORMALISE", "false"); # Normalise the individual variables

        if self.measure == "PI":
            self.compute = self.compute_pi
        elif self.measure == "lPI":
            self.compute = self.compute_pi_local
        elif self.measure == "AIS":
            self.compute = self.compute_ais
                
    # loss measure complexity
    def compute_pi(self, X):
        # self.piCalc.setObservations(X.reshape((X.shape[0],)))
        pi_avg = 0.0
        # FIXME: make that a joint PI
        for d in range(X.shape[1]):
            # self.piCalc.initialise(self.k, self.tau)
            # self.piCalc.initialise(self.k, self.tau, 0.001)
            self.picalc_init(self.k, self.tau)
            self.piCalc.setObservations(X[:,d])
            pi_avg += self.piCalc.computeAverageLocalOfObservations();
        return pi_avg
    
    def compute_pi_local(self, X):
        winsize = 100
        # self.piCalc.setObservations(X.reshape((X.shape[0],)))
        pi_avg = 0.0
        for d in range(X.shape[1]):
            for i in range(winsize, X.shape[0], 10):
                self.picalc_init(self.k, self.tau)
                # self.piCalc.initialise(self.k, self.tau)
                # self.piCalc.initialise(self.k, self.tau, 0.1)
                # print("X[i:i+winsize,d]", X[i-winsize:i,d].shape)
                self.piCalc.setObservations(X[i-winsize:i,d])
                # pi_local = self.piCalc.computeLocalOfPreviousObservations()
                pi_avg += self.piCalc.computeAverageLocalOfObservations();
                # print("pi_local", np.sum(pi_local))
                # pi_avg += np.sum(pi_local)
        return pi_avg

    def compute_ais(self, X):
        ais_avg = 0.0
        for d in range(X.shape[1]):
            # self.aisCalc.initialise(self.k, self.tau) # init for kraskov
            # self.aisCalc.initialise(self.k, self.tau, 0.001)
            self.aiscalc_init(self.k, self.tau)
            src = X[:,d].ravel()
            # print(src)
            self.aisCalc.setObservations(src)
            ais_avg_ = self.aisCalc.computeAverageLocalOfObservations()
            # ais_local = self.aisCalc.computeLocalOfPreviousObservations()
            # ais_id = self.aisCalc.computeLocalInfoDistanceOfPreviousObservations()
            # k_auto = self.aisCalc.getProperty(self.aisCalcClass.K_PROP_NAME)
            # print("active infostorage (AIS, k = %d/%s) = %f nats" % (k, k_auto, ais_avg))
            # ais_avg_.append(ais_avg)
            ais_avg += ais_avg_
        return ais_avg

def test_ind(args, M = None, fig = None, axes = None):
    assert fig is not None
    assert axes is not None
    
    if M is not None:
        M = M.copy()
    else:
        # M = [[ 0.69656214,  0.33208246],
        #           [ 0.41712419,  0.56571223]]
        # M = [[ 0.69656214,  0.33208246],
        #         [ 0.41712419,  0.56571223]]
        # M = [[ 0.69656214,  0.33208246],
        #         [ 0.41712419,  0.56571223]]
        # M = [[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]]
        # M = [[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]]
        # M = [[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]]
        # M = [[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]]
        # M = [[ 0.69656214,  0.33208246],
        #             [ 0.41712419,  0.56571223]]
        # # es vanilla PI
        # M = [[ 1.65149195, -2.1663804 ], [ 1.09012667,  0.47483937]]
        # # es vanilla PI k = 1
        # M = [[ 0.35410052,  0.36185423],  [ 1.29750725,  0.29797154]]
        # # es vanilla AIS k = 10
        # M = [[ 0.15237389, -7.50620661],  [ 0.47059135,  0.72873167]]
        # cma 1234
        M = [[ 0.48566475,  0.71121745],  [0.35990499,  0.53510189]]
        # # cma something else
        # M = [[1.01299593, -0.01980717],  [0.0265115, 0.23173182]]
        # M = [[ 2.13513734,  3.73228563], [-2.00081304,  0.40366748]]
        # # hp tpe 1
        # M = [[0.35360716354223576, 3.190109171204226], [0.011541858306227516, 0.95750455820040858]]
        # # hp anneal
        # M = [[0.10161603853231675, 1.823057896074153], [0.95762558826082311, 0.027411306108273446]]
        # # hp gp ucb
        # M = [[0.75581178213155242, 1.827718038969383], [0.0097885563114531102, 0.9436968783153078]]
        # M = [[0.022632773868189446, 0.6668438467670601], [0.84183422133138563, 0.43948905630845503]]
        # # hp gp ucb PI k = 1
        # M_ = {'m1': 0.87536668282921848, 'm0': 0.055056894205574171, 'm3': 0.035338686184602119, 'm2': 1.0728958340419548}
        # hp_tpe with pi_local winsize 100 step 10 k 10
        # M_ = {'m1': 1.6493649280464324, 'm0': 0.71606638231926245, 'm3': 0.95933124164464356, 'm2': 0.013195138595586221}
        # # hp_tpe with PI?/AIS?
        # M_ = {'m1': 3.2898452455662013, 'm0': 0.14581152780579029, 'm3': 0.89520067809917869, 'm2': 0.031551465743829638}
        # M = [[M_["m0"], M_["m1"]], [M_["m2"], M_["m3"]]]

        # cma_es 20170131 slow M
 #        M = [-0.7514154 , -2.7365132 ,  2.61213931 ,-1.84910997 ,-0.66426485 , 2.08347174 -1.09825155 , 0.93310625 ,-1.65316607 , 1.16035822 , 0.24026319 , 0.59995612 
 # -0.41730344 , 7.79230665 ,-1.49233779 ,-1.58248257 , 3.6782291 , -2.89003224
 #  4.328539 ,   2.43253161 , 0.4810217 , -4.14685984 ,-4.90913503 , 0.93255625
 # -1.30142747 , 0.55256147 ,-0.55657707 , 0.57766382 , 1.81523245 , 6.89785479
 # -2.6975135 ,  6.14064481 , 1.93846706 ,-3.99651144 , 7.95055961 , 2.4293328
 #  1.52976672 , 1.57255522 ,-4.76556823 ,-2.45029775 , 4.4360247 ,  3.30405072
 #  0.42359173 ,-3.98507505 , 3.95753928 ,-5.56014638 , 5.54858053 , 3.48650422
 #  1.31697852 ,-3.85565617 , 3.83840935 , 3.18590168 , 0.18725208 ,-1.56320902
 # -0.69730546 ,-3.79129495 , 3.15212612 ,-0.75024253 , 4.92589775 ,-1.85079826
 #  7.69352833 , 4.53011577 ,-4.34926963 ,-0.17240608 ,-0.57012001 , 3.09799469
 #  0.36284337 ,-3.19599932 , 4.29777014 ,-3.0896056 ,  4.28426481 ,-2.12271555
 #  2.28169805 , 5.81709017 , 2.46634369 ,-3.83153241 ,-1.57012585 ,-1.72009192
 # -0.36671533 , 0.88318839 , 4.36377448 , 2.1980125 , -4.08149107 , 0.39740616
 # -5.3072653 ,  5.37462117 , 6.25147611 , 5.48232004 ,-9.12562418 , 1.45251352
 # -1.67627735 ,-5.21085258 ,-0.99722649 ,-0.42665154 , 2.80383282 ,-0.87153398
 # -4.27882815 ,-3.78795119 , 7.04832173 ,-3.15230275 , 2.80892391 ,-2.37740598
 #  5.55134174 ,-2.4525543 ,  4.17792407 , 0.24796831 , 4.93469566 , 5.60239347]
        
        # es vanilla
        M = [[ 0.56363375, -0.19903545],  [-0.3386425,   0.86326516]]
        
    M = np.array(M)

    conf = {
        "numsteps": args.numsteps,
        "generator": args.generator,
        "params": M
    }
        
    Xs = evaluate_individual(conf)

    # # n = Genet(M = M)
    # n = GenetPlast(M = M)

    # numsteps = 2000

    # # log_dim = n.state_dim
    # log_dim = n.state_dim + n.networks["slow"]["s_dim"]
        
    # Xs = np.zeros((numsteps, log_dim))
    # # loop over timesteps
    # for i in range(numsteps):
    #     # x = np.dot(M, x)
    #     n.step()
    #     # Xs[i] = n.x.reshape((n.state_dim,))
    #     Xs[i,:n.state_dim] = n.networks["fast"]["x"].reshape((n.state_dim,))
    #     Xs[i,n.state_dim:] = n.networks["fast"]["M"].reshape((n.networks["slow"]["s_dim"],))
    # pi = cm.compute_pi(Xs)
    
    Xs_meas = Xs[:,[1,2]]

    cm = ComplexityMeasure(args.measure, args.measure_k, args.measure_tau, args.estimator)
    # cm  = ComplexityMeasure()
    pi  = cm.compute_pi(Xs_meas)
    ais  = cm.compute_ais(Xs_meas)
    # pi_l  = cm.compute_pi_local(Xs_meas)

    # ax1, ax2, ax3, ax4, ax4cb = axes
    ax1, ax4, ax4cb, ax2, ax3 = axes
        
    ax1.clear()
    ax1.plot(Xs[:,1], Xs[:,2], "k.", alpha=0.75)
    # h2dimg,_,_,_ = ax1.hist2d(Xs[:,1], Xs[:,2], bins = 20, range = [[-1, 1], [-1, 1]], cmap = pl.get_cmap("Greys"))
    ax1.set_xlim((-1, 1))
    ax1.set_ylim((-1, 1))
    ax1.text(0, -0.5, "pi = %f nats" % pi)
    ax1.text(0, -0.75, "ais = %f nats" % ais)
    ax1.set_aspect(1)

    ax4.clear()
    ax4cb.clear()
    mappable = ax4.imshow(M, interpolation="none", cmap = pl.get_cmap("seismic"))#, vmin=-3.0, vmax=3.0)
    cbar = fig.colorbar(mappable = mappable, cax = ax4cb, orientation = "vertical")
    ax4.set_aspect(1)
    # pl.yscale("log")
    # pl.xscale("log")
    ax2.clear()
    ax2.plot(Xs[:,1], "k-,", alpha=0.33)
    ax3.clear()
    ax3.plot(Xs[:,2], "k-,", alpha=0.33)
    pl.draw()
    pl.pause(1e-3)
    fig.show()

    # h2dimg
    return Xs, pi, ais

def evaluate_individual(conf):
    """evaluate an individual for one episode with the given configuration"""

    if conf["generator"] == "basic":
        ind_cls = Genet
    elif conf["generator"] == "double":
        ind_cls = GenetPlast
        #        ind_logdims = 
    n = ind_cls(M = conf["params"])

    ind_logdims = [n.state_dim]
    if conf["generator"] == "double":
        ind_logdims += [n.networks["slow"]["s_dim"]]
    ind_logdim = reduce(lambda x,y: x+y, ind_logdims)
    
    numsteps = conf["numsteps"]

    # initialize storage
    Xs = np.zeros((numsteps * 1, ind_logdim))
    
    # loop over timesteps
    for j in range(1):
        n = ind_cls(M = conf["params"])
        for i in range(numsteps):
            n.step()
            idx = (j * numsteps) + i
            # Xs[i] = n.x.reshape((n.state_dim,))
            Xs[idx,:n.state_dim] = n.networks["fast"]["x"].reshape((n.state_dim,))
            if ind_logdim > n.state_dim:
                Xs[idx,n.state_dim:] = n.networks["fast"]["M"].reshape((n.networks["slow"]["s_dim"],))

    return Xs
        
def objective(params, hparams):
    """evaluate an individual (parameter set) with respect to given objective"""
    # print("params", params)
    # print("hparams", hparams)
    # return np.random.uniform(0.0, 1.0)

    # high-level params
    numsteps = hparams["numsteps"]
    cm = hparams["measure"]
    # core params
    # n = Genet(M = params["M"])
    # print("len(params)", len(params), params)

    rows = len(params)
    if rows > 0:
        cols = len(params[0])
    else:
        status = STATUS_FAIL
        
    M = np.array(params[0:(rows*cols)]).reshape((rows,cols))
    
    # a dict containg network config, timeseries, loss
    rundata = dict()
    
    # #  create network
    # # n = Genet(2, 2)
    # n = Genet(M = M)
    
    # # state trajectory
    # Xs = np.zeros((numsteps, n.state_dim))
    # # loop over timesteps
    # for i in range(numsteps):
    #     # x = np.dot(M, x)
    #     n.step()
    #     # Xs[i] = n.x.reshape((n.state_dim,))
    #     # print("n.networks[\"fast\"][\"x\"].shape", n.networks["fast"]["x"].shape)
    #     Xs[i,:n.state_dim] = n.networks["fast"]["x"].reshape((n.state_dim,))

    conf = {
        "numsteps": numsteps,
        "generator": args.generator,
        "params": M
    }
        
    pi = 0
    for i in range(5):
        Xs = evaluate_individual(conf)
        
        Xs_meas = Xs[:,[1,2]]
    
        # pi = cm.compute_pi(Xs)
        # pi = cm.compute_ais(Xs)
        # pi = cm.compute_pi_local(Xs)
        pi += cm.compute(Xs_meas)
    pi /= 5.0
    pi = max(0, pi) + 1e-9
    # print("pi = %f nats" % pi)
    # loss = -np.log(pi)
    loss = -pi

    status = STATUS_OK
    # return structure: params, timeseries, scalar loss
    rundata = {
        "loss": loss, # compute_complexity(Xs)
        "status": status, # compute_complexity(Xs)
        "M": M, # n.networks["fast"]["M"],
        "timeseries": Xs.copy(),
        # "loss": np.var(Xs),
    }
    if hparams["continuous"]:
        return rundata["loss"]
    else:
        return rundata

def objective_double(params, hparams):
    """evaluate an individual (parameter set) with respect to given objective"""
    # print("params", params)
    # print("hparams", hparams)

    # high-level params
    numsteps = hparams["numsteps"]
    cm = hparams["measure"]
    
    # core params
    # n = Genet(M = params["M"])
    # print("params[0:24]", params[0:24])
    # print("len(params)", len(params), params)
    M = np.array(params).reshape((9,12))
    
    # a dict containg network config, timeseries, loss
    rundata = dict()
    
    # #  create network
    # # n = Genet(2, 2)
    # n = GenetPlast(M = M)

    # # log_dim = n.state_dim
    # log_dim = n.state_dim + n.networks["slow"]["s_dim"]
        
    # # state trajectory
    # Xs = np.zeros((numsteps, log_dim))
    
    # # loop over timesteps
    # for i in range(numsteps):
    #     # x = np.dot(M, x)
    #     n.step()
    #     # Xs[i] = n.x.reshape((log_dim,))
    #     Xs[i,:n.state_dim] = n.networks["fast"]["x"].reshape((n.state_dim,))
    #     Xs[i,n.state_dim:] = n.networks["fast"]["M"].reshape((n.networks["slow"]["s_dim"],))

    conf = {
        "numsteps": numsteps,
        "generator": args.generator,
        "params": M
    }
        
    pi = 0
    for i in range(5):
        Xs = evaluate_individual(conf)
        
        Xs_meas = Xs[:,[1,2]]
    
        # pi = cm.compute_pi(Xs_meas)
        # pi = cm.compute_ais(Xs_meas)
        # pi = cm.compute_pi_local(Xs)
        pi += cm.compute(Xs_meas)
    pi /= 5.0
    pi = max(0, pi) + 1e-9
    # print("pi = %f nats" % pi)
    # loss = -np.log(pi)
    loss = -pi
    # return structure: params, timeseries, scalar loss
    rundata = {
        "loss": loss, # compute_complexity(Xs)
        "status": STATUS_OK, # compute_complexity(Xs)
        "M": M, # n.networks["slow"]["M"], # n.M
        "timeseries": Xs.copy(),
        # "loss": np.var(Xs),
    }
    if hparams["continuous"]:
        return rundata["loss"]
    else:
        return rundata

def main(args):
    """main, just dispatch to mode's main"""

    # init jpype
    init_jpype(args.jarloc_jidt)
    
    # experiment signature
    setattr(args, "expsig", time.strftime("%Y%m%d-%H%M%S"))
    
    if args.mode == "es_vanilla":
        main_es_vanilla(args)
    elif args.mode == "cma_es":
        main_cma_es(args)
    elif args.mode == "hp_tpe":
        setattr(args, "suggest", tpe.suggest)
        main_hp(args)
    elif args.mode == "hp_random_search":
        setattr(args, "suggest", rand.suggest)
        main_hp(args)
    elif args.mode == "hp_anneal":
        setattr(args, "suggest", anneal.suggest)
        main_hp(args)
    elif args.mode == "hp_gp_ucb":
        setattr(args, "suggest", partial(suggest_algos.ucb, stop_at=1e-0))
        main_hp(args)
    elif args.mode == "hp_gp_ei":
        setattr(args, "suggest", partial(suggest_algos.ei, stop_at=1e-0))
        main_hp(args)
    elif args.mode == "test_ind":
        test_ind(args)

def main_cma_es(args):
    import cma
    print("args", args)
    # options = {'CMA_diagonal':100, 'seed':1234, 'verb_time':0, "maxfevals": 2000}
    # options = {'CMA_diagonal': 0, 'seed':32984, 'verb_time':0, "maxfevals": 2000}
    options = {'CMA_diagonal': 0, 'seed':4534985, 'verb_time':0, "maxfevals": 4000, "termination_callback": None}

    hparams = {
        "numsteps": args.numsteps,
        "measure": ComplexityMeasure(),
        "continuous": True,
    }
        
    obj = get_obj(args)
    pobjective = partial(obj, hparams=hparams)
    # pobjective = partial(objective, hparams=hparams)

    # func = objective # args["func"]
    # arguments: function, initial params, initial var, options
    # res = cma.fmin(cma.fcts.griewank, [0.1] * 10, 0.5, options)
    # res = cma.fmin(pobjective, [0.5] * 4, 0.3, options)
    res = cma.fmin(pobjective, [0.5] * (9 * 12), 3.0, options)

    print("result cma_es", res[0], res[1])
    return res[0]

def main_hp(args):
      
    hparams = {
        "numsteps": args.numsteps,
        "measure": ComplexityMeasure(),
        "continuous": False,
    }
    pobjective = partial(objective, hparams=hparams)

    print(pobjective(params = np.random.uniform(-1.0, 1.0, (2,2))))
    
    def objective_hp(params):
        # print("params", params)
        targ = np.array(params) # .astype(np.float32)
        # print(targ.dtype)
        now = time.time()
        func = pobjective # args["func"]
        ret = func(targ) # cma.fcts.griewank(targ)
        took = time.time() - now
        print("feval took %f s with ret = %s" % (took, ret["loss"]))
        return ret

    space = [hp.loguniform("m%d" % i, -5, 2.0) for i in range(4)]

    trials = Trials()
    suggest = tpe.suggest # something
    bests = []
    initevals = 0
    maxevals = 500
    lrstate = np.random.RandomState(123)
    
    for i in range(initevals, maxevals):
        print("fmin iter %d" % i,)
        bests.append(fmin(objective_hp,
                    space,
                    algo=suggest,
                    max_evals=i+1,
                    rstate=lrstate, # 1, 10, 123, 
                    trials=trials,
                    verbose=1))
        lrstate = np.random.RandomState()

    best = bests[-1]
    for i in range(5):
        print("best[%d]" % (-1-i), bests[-1-i])
    
    # for k,v in best:
    pkeys = best.keys()
    pkeys.sort()
    ret = np.zeros((len(pkeys)))
    for i,k in enumerate(pkeys):
        # print(k)
        ret[i] = best[k]
        
    return ret

def get_generator(args):
    if args.generator == "basic":
        n = Genet(2,2)
    elif args.generator == "double":
        n = GenetPlast(2, 2)
    return n

def get_generator_params(args):
    if args.generator == "basic":
        n = Genet(2,2)
        p = n.networks["fast"]["M"]
        t = n.networks["fast"]["tau"]
    elif args.generator == "double":
        n = GenetPlast(2, 2)
        p = n.networks["slow"]["M"]
        t1 = np.ones_like(n.networks["fast"]["x"]) * n.networks["fast"]["tau"]
        t2 = np.ones_like(n.networks["slow"]["M"]) * n.networks["slow"]["tau"]
        # t = np.vstack((t1, t2))
        t = t1
    return n, p, t

def get_obj(args):
    if args.generator == "basic":
        obj = objective
    elif args.generator == "double":
        obj = objective_double
    return obj

def save_topinds(topinds, args, generation_cnt = 0):
    Mmax = 0
    Mmin = 0
    for topind in topinds:
        Mmax = max(Mmax, np.max(topind["M"]))
        Mmin = min(Mmin, np.min(topind["M"]))
        
    fig1 = pl.figure(figsize = (5*200.0/100.0, 300.0/100.0))
    fig2 = pl.figure(figsize = (5*200.0/100.0, 300.0/100.0))
    # fig.show()
    gs1 = gridspec.GridSpec(1, len(topinds) * 1)
    gs2 = gridspec.GridSpec(2, len(topinds) * 1, height_ratios = [0.95, 0.05]) # + 1, width_ratios = [1] * len(topinds) + [0.1])

    for i, topind in enumerate(topinds):
        ax1 = fig1.add_subplot(gs1[0,i])
        ax1.axis("off")
        # print("type(H2D)", type(topind["H2D"]))
        # ax1.imshow(topind["H2D"], interpolation = "none", cmap = pl.get_cmap("Greys"))
        ax1.plot(topind["H2D"][:,1], topind["H2D"][:,2], "k.", alpha=0.75)
        ax1.set_xlim([-1, 1])
        ax1.set_ylim((-1, 1))
        ax1.set_aspect(1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig2.add_subplot(gs2[0,i])
        mappable = ax2.imshow(topind["M"], interpolation = "none", cmap = pl.get_cmap("seismic"), vmin = Mmin, vmax = Mmax)
        ax2.set_aspect(1)
        ax2.set_xticks([])
        ax2.set_yticks([])

    cbarax = fig2.add_subplot(gs2[1,:])
    # divider = make_axes_locatable(cbarax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    # cbar = fig2.colorbar(mappable = mappable, cax = cbarax) # , fraction = 0.1)
    cbar = fig2.colorbar(mappable = mappable, cax = cbarax, orientation = "horizontal") # , fraction = 0.1)

    fig1.subplots_adjust(wspace=0.0, hspace = 0.0)
    fig1.savefig("%s/gen%04d_top%02d_pheno_%s.jpg" % (args.datadir, generation_cnt, len(topinds), args.expsig), dpi = 300, bbox_inches="tight")
    pl.close(fig1)
    
    fig2.subplots_adjust(wspace=0.0, hspace = 0.05)
    fig2.savefig("%s/gen%04d_top%02d_geno__%s.jpg" % (args.datadir, generation_cnt, len(topinds), args.expsig), dpi = 300, bbox_inches="tight")
    pl.close(fig2)
    # pl.draw()
    # pl.pause(1e-4)

def main_es_vanilla(args):
    # evolution / opt params
    numgenerations = args.numgenerations
    numpopulation = args.numpopulation
    numsteps = args.numsteps
    numelite = args.numelite

    setattr(args, "datadir", "ep3/ep3_es_vanilla_gen%s_meas%s_est%s_k%d_t%d_%s" % (args.generator, args.measure, args.estimator, args.measure_k, args.measure_tau, args.expsig))
    os.mkdir(args.datadir)
    
    # global logging structure: experiment configuration, generation data for all individuals and statistics
    experiment = {
        "conf": args,
        "generations": [],
        "generations_stats": [],
    }
    
    # # generations array containing
    # generations = []

    # complexity measure
    cm = ComplexityMeasure(args.measure, args.measure_k, args.measure_tau, args.estimator)

    # array of parameter ndarray for the current generation
    newgen = []
    for j in range(numpopulation):
        n, p, tau = get_generator_params(args)
        # print("p.shape", p.shape)
        # n = Genet(2, 2)
        # newgen.append(n.M)
        # n = GenetPlast(2, 2)
        # newgen.append(n.networks["slow"]["M"])
        newgen.append(p)

    numindplot = 5
        
    pl.ion()
    fig = pl.figure(figsize = (20, 13))
    
    # gs = gridspec.GridSpec(4, 7 * numindplot)
    # gs = gridspec.GridSpec(5, 6 * numindplot)
    gs = gridspec.GridSpec(4, 4 * numindplot + 1)

    allindaxes = []
    for i in range(numindplot):
        thisindaxes = []
        # thisindaxes.append(fig.add_subplot(gs[0:2, (i*7)       : (i * 7 + 3)]))
        # thisindaxes.append(fig.add_subplot(gs[0:2, (i * 7 + 3) : (i * 7 + 6)]))
        # thisindaxes.append(fig.add_subplot(gs[0:2, (i*7 + 6)]))
        # thisindaxes.append(fig.add_subplot(gs[2,   (i*7)       : ((i+1)*7)]))
        # thisindaxes.append(fig.add_subplot(gs[3,   (i*7)       : ((i+1)*7)]))
        
        # thisindaxes.append(fig.add_subplot(gs[0:2, (i*6)       : (i * 6 + 3)]))
        # thisindaxes.append(fig.add_subplot(gs[0:2, (i * 6 + 3) : (i * 6 + 6)]))
        # thisindaxes.append(fig.add_subplot(gs[2,   (i * 6 + 3) : (i * 6 + 6)]))
        # thisindaxes.append(fig.add_subplot(gs[3,   (i*6)       : ((i+1)*6)]))
        # thisindaxes.append(fig.add_subplot(gs[4,   (i*6)       : ((i+1)*6)]))

        thisindaxes.append(fig.add_subplot(gs[0:2, (i*4)       : (i * 4 + 2)]))
        thisindaxes.append(fig.add_subplot(gs[0:2, (i * 4 + 2) : (i * 4 + 4)]))
        thisindaxes.append(fig.add_subplot(gs[0:2, -1]))
        thisindaxes.append(fig.add_subplot(gs[2,   (i*4)       : ((i+1)*4)]))
        thisindaxes.append(fig.add_subplot(gs[3,   (i*4)       : ((i+1)*4)]))
        allindaxes.append(thisindaxes)
    
    fig.show()
    # pl.draw()

    fig2 = pl.figure(figsize = (10, 6))
    f2ax1 = fig2.add_subplot(111)
    fig2.show()

    fig3 = pl.figure(figsize = (10, 6))
    f3ax1 = fig3.add_subplot(111)
    fig3.show()
        
    # loop over generations
    for k in range(numgenerations):
        # arrays of individuals each element of which is
        population = dict()

        hparams = {
            "numsteps": numsteps,
            "measure": cm,
            "continuous": False,
        }
        # pobjective = partial(objective, hparams=hparams)
        # pobjective = partial(objective_double, hparams=hparams)
        
        obj = get_obj(args)
        pobjective = partial(obj, hparams=hparams)
        
        # loop over population
        for j in range(numpopulation):
            # params = {
            #     "M": newgen[j],
            # }

            # get parameters
            # print("newgen[j]", newgen[j].shape)
            params = newgen[j].tolist()

            # evaluate individual
            # rundata = objective(params, hparams)
            rundata = pobjective(params)

            # store results for indidividual
            # population.append(rundata)
            population["%d" % j] = rundata

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
        generation_stats = {
            "avg_fit": np.mean(ind_loss),
            "std_fit": np.std(ind_loss),
            "max_fit": np.max(ind_loss),
            "min_fit": np.min(ind_loss),
        }
        print("gen %04d: max fit = %f, avg/std fit = %f/%f, min fit = %f, " % (k, generation_stats["max_fit"], generation_stats["avg_fit"],
                                                                               generation_stats["std_fit"], generation_stats["min_fit"]))
        experiment["generations_stats"].append(generation_stats)

        avgf = []
        stdf = []
        maxf = []
        minf = []
        for gs in experiment["generations_stats"]:
            avgf.append(gs["avg_fit"])
            stdf.append(gs["std_fit"])
            maxf.append(gs["max_fit"])
            minf.append(gs["min_fit"])
        avgf = np.array(avgf)
        stdf = np.array(stdf)
        maxf = np.array(maxf)
        minf = np.array(minf)
        f3ax1.clear()
        f3ax1.plot(minf, "yo", alpha=0.5, label="min")
        f3ax1.plot(maxf, "ko", alpha=0.5, label="max")
        f3ax1.plot(avgf, "ro", alpha=0.5, label="avg")
        f3ax1.plot(avgf + stdf, "go", alpha=0.5, label="+sigma")
        f3ax1.plot(avgf - stdf, "go", alpha=0.5, label="-sigma")
        f3ax1.legend()
        pl.draw()
        pl.pause(1e-3)

        # generate new generation from loss sorted current generation
        # get best n individuals (FIXME: use a dataframe)
        # import operator
        # print(generations[-1].items())
        # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
        # sorted_x = sorted(generations[-1].items(), key=operator.itemgetter(1))
        # for maximization
        # sorted_x = sorted(generations[-1].items(), key=lambda x: x[1]["loss"], reverse=True)
        # for minimization of neg loss
        # sorted_x = sorted(generations[-1].items(), key=lambda x: x[1]["loss"], reverse=False)
        sorted_x = sorted(population.items(), key=lambda x: x[1]["loss"], reverse=False)

        # store current generation
        # generations.append(population)
        experiment["generations"].append(population)
        # generations.append(sorted_x)

        # save experiment progress
        # pickle.dump(generations, open("ep3/ep3_generations_%s.bin" % (args.expsig), "wb"))
        if k % 10 == 0:
            pickle.dump(experiment, open("%s/ep3_experiment_%s.bin" % (args.datadir, args.expsig), "wb"))

        fitprobs      = []
        fitprobs_rank = []
        for i, (idx, ind) in enumerate(sorted_x):
            # print("i", i, "idx", idx) # , "ind", ind)
            fitprobs.append(ind["loss"])
            fitprobs_rank.append((2 * numpopulation) - i)
        fitprobs = np.sqrt(np.abs(np.array(fitprobs)))
        fitprobs_rank = np.sqrt(np.array(fitprobs_rank, dtype=float))
        fitprobs /= np.sum(fitprobs)
        fitprobs_rank /= np.sum(fitprobs_rank)
        # print("fitprobs_rank", fitprobs_rank)
        # print("fitprobs", np.abs(fitprobs))

        f2ax1.clear()
        f2ax1.plot(fitprobs, "k-o", alpha=0.2)
        f2ax1.plot(fitprobs_rank, "r-o", alpha=0.2)
        pl.draw()
        pl.pause(1e-3)

        # print("sample fitprobs", argsample(fitprobs))
                    
        # print("sorted_x", sorted_x[:]["loss"])
        # print(sorted_x[0][1])
        
        # do elite backup
        for i in range(numelite):
            newgen[i] = sorted_x[i][1]["M"]

        # do non-elite new population
        for i in range(numelite, numpopulation):
            # select
            # 1: sample weighted by fitness
            c = argsample(fitprobs)[0]
            # 2: sample weighted by fitness rank
            # 3: naive uniform sampling
            # c = np.random.choice(min(numpopulation, 15))# make tournament or something
            newgen[i] = sorted_x[c][1]["M"]

            c1 = argsample(fitprobs)[0]
            c2 = argsample(fitprobs)[0]
            # c1 = argsample(fitprobs_rank)[0]
            # c2 = argsample(fitprobs_rank)[0]
            sh_ = sorted_x[c1][1]["M"].shape
            m1 = sorted_x[c1][1]["M"].flatten()
            m2 = sorted_x[c2][1]["M"].flatten()
            xover_at = np.random.randint(m1.shape[0])
            newgen[i] = np.hstack((m1[:xover_at], m2[xover_at:])).reshape(sh_)

        # mutate all
        for i in range(numpopulation * 2): # mutate more
            i = i % numpopulation
            # mutate
            if np.random.uniform() < 0.25: # 05:
                mut_idx = np.random.choice(np.prod(newgen[i].shape))
                ind_idx = np.random.choice(numpopulation)
                # print("mut_idx", mut_idx)
                tmp_s = newgen[ind_idx].shape
                tmp = newgen[ind_idx].flatten()
                if args.op_mutation == "normal":
                    n = np.random.normal(0, 1.0)
                elif args.op_mutation == "pareto":
                    n = ((np.random.binomial(1, 0.5) - 0.5) * 2) * np.random.pareto(1.5) * 0.5
                tmp[mut_idx] += n
                # print("n", n, tmp[mut_idx])
                newgen[ind_idx] = tmp.copy().reshape(tmp_s)

        # do some online plotting
        if k % args.plotinterval == 0:
            # print("generations", generations[-1])
            # sorted_x = sorted(generations[-1].items(), key=lambda x: x[1]["loss"], reverse=True)
            sorted_x = sorted(experiment["generations"][-1].items(), key=lambda x: x[1]["loss"], reverse=True)
            # for ind in generations[-1].values():
            numsteps_ = getattr(args, "numsteps")
            setattr(args, "numsteps", 1000)
            topinds = []
            for i,ind in enumerate(sorted_x):
                if i >= (numpopulation - 5):
                    # test_ind(args = args, M = ind[1]["M"], fig = fig, axes = [ax1, ax2, ax3, ax4, ax4cb])
                    H2D, pi, ais = test_ind(args = args, M = ind[1]["M"], fig = fig, axes = allindaxes[i - (numpopulation - numindplot)])
                    # save_ind(args, )
                    topinds.append({"M": ind[1]["M"], "H2D": H2D, "pi": pi, "ais": ais})
                # print("last generation fit/M", ind[1]["loss"], ind[1]["M"])
            save_topinds(topinds, args, k)
            setattr(args, "numsteps", numsteps_)
            
    pl.ioff()
    pl.show()
    if args.plotsave:
        fig.savefig("ep3_es_vanilla_top5_%s.pdf" % args.expsig, dpi=300, bbox_inches="tight")
        fig2.savefig("ep3_es_vanilla_stats_%s.pdf" % args.expsig, dpi=300, bbox_inches="tight")
        fig3.savefig("ep3_es_vanilla_prob_%s.pdf" % args.expsig, dpi=300, bbox_inches="tight")
    # pl.pause(100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-om", "--op_mutation", type=str, default="pareto",
                        help="Mutation operator noise type [pareto]")
    parser.add_argument("-e", "--estimator", type=str, default="kraskov1",
                        help="Type of estimator to use with JIDT [kraskov1].")
    parser.add_argument("-g", "--generator", type=str, default="basic",
                        help="Type of generator [basic]. This is the structure whose parameters we want to evolve.")
    parser.add_argument("-j", "--jarloc_jidt", type=str, default="/home/src/QK/infodynamics-dist/infodynamics.jar",
                        help="Location of information dynamics toolkit .jar file [/home/src/QK/infodynamics-dist/infodynamics.jar]")
    parser.add_argument("-m", "--mode", type=str, default="es_vanilla",
                        help="optimization / search mode [es_vanilla], (es_vanilla, cma_es, hp_tpe, hp_random_search, hp_gp_ucb, hp_gp_ei)")
    parser.add_argument("-ms", "--measure", type=str, default="PI",
                        help="Type of complexity measure to use as fitness [PI], one of PI, lPI (local PI), AIS")
    parser.add_argument("-msk", "--measure_k", type=int, default=100,
                        help="Complexity measure embedding length [100]")
    parser.add_argument("-mstau", "--measure_tau", type=int, default=1,
                        help="Complexity measure embedding delay [1]")
    parser.add_argument("-n", "--numsteps", type=int, default=1000,
                        help="number of timesteps for individual evaluation [1000]")
    parser.add_argument("-ne", "--numelite", type=int, default=1,
                        help="Extent of elitism, how many best individuals to transfer unmodified [1]")
    parser.add_argument("-ng", "--numgenerations", type=int, default=100,
                        help="number of generations to evolve for [100]")
    parser.add_argument("-np", "--numpopulation", type=int, default=20,
                        help="number of individuals in population [20]")
    parser.add_argument('-ps', "--plotsave", action='store_true', help='Save plot to pdf?')
    parser.add_argument('-pi', "--plotinterval", type=int, default=10, help='Interval for intermediate plotting [10], in number of generations')
    args = parser.parse_args()
    main(args)
