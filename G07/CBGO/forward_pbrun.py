import os
import pandas as pd
import numpy as np

def g07(x):
    """
    G07 function from Runarsson & Yao (2000)
    10-dimensional problem with 8 constraints
    """
    x = np.array(x)
    
    # Objective function
    f = x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1] + (x[2] - 10)**2 + \
        4*(x[3] - 5)**2 + (x[4] - 3)**2 + 2*(x[5] - 1)**2 + 5*x[6]**2 + \
        7*(x[7] - 11)**2 + 2*(x[8] - 10)**2 + (x[9] - 7)**2 + 45
    
    # Constraints (all in form g(x) <= 0)
    g1 = -105 + 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7]
    g2 = 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]
    g3 = -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12
    g4 = 3*(x[0] - 2)**2 + 4*(x[1] - 3)**2 + 2*x[2]**2 - 7*x[3] - 120
    g5 = 5*x[0]**2 + 8*x[1] + (x[2] - 6)**2 - 2*x[3] - 40
    g6 = x[0]**2 + 2*(x[1] - 2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5]
    g7 = 0.5*(x[0] - 8)**2 + 2*(x[1] - 4)**2 + 3*x[4]**2 - x[5] - 30
    g8 = -3*x[0] + 6*x[1] + 12*(x[8] - 8)**2 - 7*x[9]
    
    return {
        "func": f,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "g5": g5,
        "g6": g6,
        "g7": g7,
        "g8": g8
    }

# Replace rosenbrock with levy in the helper function
def helper(pvals=None):
    if pvals is None:
        x = pd.read_csv("dv.dat").values.reshape(-1).tolist()
    else:
        pvals_ordered = {pval: pvals[pval] for pval in sorted(pvals.index, key=lambda x: int(x[1:]))}
        x = np.array(list(pvals_ordered.values()))
    sim = {"func": g07(x)['func'], "func_var": 0, "ei": 0,
           "g1": g07(x)['g1'], "g2": g07(x)['g2'], "g3": g07(x)['g3'], "g4": g07(x)['g4'],
           "g5": g07(x)['g5'], "g6": g07(x)['g6'], "g7": g07(x)['g7'], "g8": g07(x)['g8'], 
           'pf_g1': 0, 'pf_g2': 0, 'pf_g3': 0, 'pf_g4': 0, 'pf_g5': 0, 'pf_g6': 0, 'pf_g7': 0, 'pf_g8': 0}

    with open('output.dat','w') as f:
        f.write('obsnme,obsval\n')
        f.write('func,'+str(sim["func"])+'\n')
        f.write('func_var,'+str(sim["func_var"])+'\n')
        f.write('ei,'+str(sim["ei"])+'\n')
        f.write('g1,'+str(sim["g1"])+'\n')
        f.write('g2,'+str(sim["g2"])+'\n')
        f.write('g3,'+str(sim["g3"])+'\n')
        f.write('g4,'+str(sim["g4"])+'\n')
        f.write('g5,'+str(sim["g5"])+'\n')
        f.write('g6,'+str(sim["g6"])+'\n')
        f.write('g7,'+str(sim["g7"])+'\n')
        f.write('g8,'+str(sim["g8"])+'\n')
        f.write('pf_g1,'+str(sim["pf_g1"])+'\n')
        f.write('pf_g2,'+str(sim["pf_g2"])+'\n')
        f.write('pf_g3,'+str(sim["pf_g3"])+'\n')
        f.write('pf_g4,'+str(sim["pf_g4"])+'\n')
        f.write('pf_g5,'+str(sim["pf_g5"])+'\n')
        f.write('pf_g6,'+str(sim["pf_g6"])+'\n')
        f.write('pf_g7,'+str(sim["pf_g7"])+'\n')
        f.write('pf_g8,'+str(sim["pf_g8"])+'\n')
    return sim

def ppw_worker(pst_name,host,port):
    import pyemu
    ppw = pyemu.os_utils.PyPestWorker(pst_name,host,port,verbose=False)
    pvals = ppw.get_parameters()
    if pvals is None:
        return

    obs = ppw._pst.observation_data.copy()
    obs = obs.loc[ppw.obs_names,"obsval"]

    while True:

        sim = helper(pvals=pvals)

        obs.update(sim)
        
        ppw.send_observations(obs.values)
        pvals = ppw.get_parameters()
        if pvals is None:
            break


if __name__ == "__main__":
    helper()
