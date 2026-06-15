import pandas as pd
import numpy as np
import glob
import os
import laGPy as gpr
from scipy.stats import norm
import pyemu

temp_dir = "template_inner"
outer_dirs = sorted(glob.glob("outer_*"), key=lambda x: int(os.path.basename(x).split("_")[1]))
pst = pyemu.Pst(glob.glob(os.path.join(temp_dir, "*.pst"))[0])

def emulate(pvals = None):
    if pvals is None:
        decvar = pd.read_csv(os.path.join(temp_dir, "dv.dat")).values.transpose()
    else:
        pvals_ordered = {pval: pvals[pval] for pval in sorted(pvals.index, key=lambda x: int(x[1:]))}
        decvar = np.array(list(pvals_ordered.values())).reshape(1, -1)

    pred_obj = gpr.loadGP(fname=os.path.join(temp_dir, "gp_obj.gp")).predict_lite(decvar)
    pred_g1 = gpr.loadGP(fname=os.path.join(temp_dir, "gp_g1.gp")).predict_lite(decvar)
    pred_g2 = gpr.loadGP(fname=os.path.join(temp_dir, "gp_g2.gp")).predict_lite(decvar)
    pred_g3 = gpr.loadGP(fname=os.path.join(temp_dir, "gp_g3.gp")).predict_lite(decvar)
    pred_g4 = gpr.loadGP(fname=os.path.join(temp_dir, "gp_g4.gp")).predict_lite(decvar)
    pred_g5 = gpr.loadGP(fname=os.path.join(temp_dir, "gp_g5.gp")).predict_lite(decvar)
    pred_g6 = gpr.loadGP(fname=os.path.join(temp_dir, "gp_g6.gp")).predict_lite(decvar)
    pred_g7 = gpr.loadGP(fname=os.path.join(temp_dir, "gp_g7.gp")).predict_lite(decvar)
    pred_g8 = gpr.loadGP(fname=os.path.join(temp_dir, "gp_g8.gp")).predict_lite(decvar)

    pf_g1 = norm.cdf(-pred_g1["mean"].item()/np.sqrt(pred_g1["s2"].item()))
    pf_g2 = norm.cdf(-pred_g2["mean"].item()/np.sqrt(pred_g2["s2"].item()))
    pf_g3 = norm.cdf(-pred_g3["mean"].item()/np.sqrt(pred_g3["s2"].item()))
    pf_g4 = norm.cdf(-pred_g4["mean"].item()/np.sqrt(pred_g4["s2"].item()))
    pf_g5 = norm.cdf(-pred_g5["mean"].item()/np.sqrt(pred_g5["s2"].item()))
    pf_g6 = norm.cdf(-pred_g6["mean"].item()/np.sqrt(pred_g6["s2"].item()))
    pf_g7 = norm.cdf(-pred_g7["mean"].item()/np.sqrt(pred_g7["s2"].item()))
    pf_g8 = norm.cdf(-pred_g8["mean"].item()/np.sqrt(pred_g8["s2"].item()))

    sim = {
        'func': pred_obj["mean"].item(),
        'func_var': pred_obj["s2"].item(),
        'ei': 0,
        'g1': pred_g1["mean"].item(),
        'g2': pred_g2["mean"].item(),
        'g3': pred_g3["mean"].item(),
        'g4': pred_g4["mean"].item(),
        'g5': pred_g5["mean"].item(),
        'g6': pred_g6["mean"].item(),
        'g7': pred_g7["mean"].item(),
        'g8': pred_g8["mean"].item(),
        'pf_g1': pf_g1,
        'pf_g2': pf_g2,
        'pf_g3': pf_g3,
        'pf_g4': pf_g4,
        'pf_g5': pf_g5,
        'pf_g6': pf_g6,
        'pf_g7': pf_g7,
        'pf_g8': pf_g8,
    }

    #compute EI
    curr_opt = pd.read_csv(os.path.join(temp_dir, "curr_opt.csv"))['func'].values
    s = np.sqrt(sim['func_var'])
    I = curr_opt - sim['func']
    if np.isclose(s, 0):
        first_term = I if I > 0 else 0
        second_term = 0
    else:
        z = (curr_opt - sim['func']) / s
        first_term = I * norm.cdf(z)
        second_term = s * norm.pdf(z)
    sim['ei'] = (first_term + second_term).item()

    # #cluster analysis
    # training_dv = pd.read_csv(os.path.join(temp_dir, "gp_0.dv_training.csv"))
    # cluster_info = pd.read_csv(os.path.join(temp_dir, "cluster_info.csv"))
    # cluster_summary = pd.read_csv(os.path.join(temp_dir, "cluster_summary.csv"))
    # n_clusters = cluster_summary.shape[0]

    # from sklearn.metrics import pairwise_distances
    # training_dv_values = training_dv.drop(columns=['real_name']).values
    # distances = pairwise_distances(decvar.reshape(1, -1), training_dv_values)
    # nearest_index = np.argmin(distances)

    # # Get the nearest member's real name and corresponding values
    # nearest_member = training_dv.iloc[nearest_index]
    # nearest_real_name = nearest_member['real_name']
    # nearest_cluster_size = cluster_info.loc[cluster_info['real_name'] == nearest_real_name, 'avg_cluster_size'].values[0]
  
    # cluster_diffct = (1.5 * training_dv.shape[0] / n_clusters) - nearest_cluster_size
    # sim['cluster_diffct'] = cluster_diffct

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
        sim = emulate(pvals=pvals)
        obs.update(sim)
        ppw.send_observations(obs.values)
        pvals = ppw.get_parameters()
        if pvals is None:
            break

if __name__ == "__main__":
    emulate()
