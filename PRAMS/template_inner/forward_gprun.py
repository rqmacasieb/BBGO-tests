import pandas as pd
import numpy as np
import glob
import os
import laGPy as gpr
from scipy.stats import norm
import subprocess
import shutil

temp_dir = "."

def emulate(pvals = None):
    X = pd.read_csv(os.path.join(temp_dir, "gp_model", "gp_0.dv_training.csv"))
    y = pd.read_csv(os.path.join(temp_dir, "gp_model", "gp_0.obs_training.csv"))
    
    pump_data = pd.read_csv(os.path.join(temp_dir, "gp_model", "pump.dat"), 
                           sep=r'\s+', header=None, names=['well_id', 'rate']) 

    if pvals is None:
        col_order = [col for col in X.columns if col != 'real_name']
        pvals_ordered = {col: pump_data.loc[pump_data['well_id'] == col.upper(), 'rate'].item() for col in col_order}
        decvar = np.array(list(pvals_ordered.values())).reshape(1,-1)

    else:
        for i, row in pump_data.iterrows():
            well_id = row['well_id']
            if well_id in pvals:
                pump_data.loc[i, 'rate'] = pvals[well_id]
            elif well_id.upper() in pvals: 
                pump_data.loc[i, 'rate'] = pvals[well_id.upper()]
            elif well_id.lower() in pvals:  
                pump_data.loc[i, 'rate'] = pvals[well_id.lower()]
            else:
                print(f"Warning: Well ID {well_id} not found in parameters")
        
        pump_data.to_csv(os.path.join(temp_dir, "gp_model", "pump.dat"), sep=' ', index=False, header=False)
    
        col_order = [col for col in X.columns if col != 'real_name']
        pvals_ordered = {col: pvals[col] for col in col_order}
        decvar = np.array(list(pvals_ordered.values())).transpose()
    
    os.chdir(os.path.join(temp_dir, "gp_model"))
    subprocess.run(["./pumptds"], check=True)
    os.chdir("../")
    
    pump_tds = pd.read_csv(os.path.join(temp_dir, "gp_model", "pump_tds.txt"), header = None).values
    shutil.copy(os.path.join(temp_dir, "gp_model", "pump_tds.txt"), os.path.join(temp_dir, "pest", "pump_tds.txt"))

    pred_totpenalties = gpr.loadGP(fname=os.path.join(temp_dir, "gp_model", "gp_obj.gp")).predict_lite(decvar)
    pred_sw_flux = gpr.loadGP(fname=os.path.join(temp_dir, "gp_model", "gp_sw.gp")).predict_lite(decvar)
    pf_sw_flux = norm.cdf((10000-pred_sw_flux["mean"].item())/np.sqrt(pred_sw_flux["s2"].item()))

    sim = {
        'pumptotal': pump_tds[0].item(),
        'tds': pump_tds[1].item(),
        'pf_tds': 0,
        'total_penalties': pred_totpenalties['mean'].item(),
        'total_penalties_var': pred_totpenalties['s2'].item(),
        'sw_flux': pred_sw_flux["mean"].item(),
        'pf_sw_flux': pf_sw_flux.item(),
        'ei': 0
    }

    #compute EI
    curr_opt = pd.read_csv(os.path.join(temp_dir, 'gp_model',"curr_opt.csv"))['total_penalties'].values
    s = np.sqrt(sim['total_penalties_var'])
    I = curr_opt - sim['total_penalties']
    if np.isclose(s, 0):
        first_term = I if I > 0 else 0
        second_term = 0
    else:
        z = (curr_opt - sim['total_penalties']) / s
        first_term = I * norm.cdf(z)
        second_term = s * norm.pdf(z)
    sim['ei'] = (first_term + second_term).item()

    with open(os.path.join(temp_dir, "pest", "penalties.dat"), 'w') as f:
        f.write('name value\n')
        f.write('TOTAL '+str(sim['total_penalties'])+'\n')
        f.write('VAR '+str(sim['total_penalties_var'])+'\n')
        f.write('EI '+str(sim['ei'])+'\n')

    with open(os.path.join(temp_dir, "pest", "sw_flux.dat"), 'w') as f:
        f.write('SW_FLUX '+str(sim['sw_flux'])+'\n')
        f.write('PF_SW_FLUX '+str(sim['pf_sw_flux'])+'\n')
    
    return sim

def ppw_worker(pst_name,host,port):
    import pyemu
    ppw = pyemu.os_utils.PyPestWorker(pst_name,host,port,verbose=False)
    
    original_dir = os.getcwd()
    # Debug prints
    # print("Initial directory:", os.getcwd())
    # print("Directory contents:", os.listdir())
    
    try:
        # Get process ID to identify worker
        import multiprocessing
        worker_id = multiprocessing.current_process()._identity[0]
        # print(f"Worker ID: {worker_id}")
        
        # Look for worker directory in MEMDIR if available
        memdir = os.environ.get('MEMDIR', '/dev/shm')
        worker_base = os.path.join(memdir, str(os.getppid()), "prams_bbgo")
        worker_dir = os.path.join(worker_base, f"worker_{worker_id}")
        # print(f"Expected worker directory: {worker_dir}")
        
        # If worker directory doesn't exist in MEMDIR, look in current directory
        if not os.path.exists(worker_dir):
            worker_dir = os.path.join(os.getcwd(), f"worker_{worker_id}")
            # print(f"Falling back to local worker directory: {worker_dir}")
        
        # Make sure worker directory exists and has necessary files
        os.makedirs(worker_dir, exist_ok=True)
        if not os.path.exists(os.path.join(worker_dir, "gp_model")):
            template_dir = "template_inner"
            # print(f"Copying files from {template_dir} to {worker_dir}")
            shutil.copytree(os.path.join(template_dir, "gp_model"), 
                        os.path.join(worker_dir, "gp_model"))
        
        # Change to worker directory
        os.chdir(worker_dir)
        # print("Final working directory:", os.getcwd())
        # print("Worker directory contents:", os.listdir())
        
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

    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    emulate()
