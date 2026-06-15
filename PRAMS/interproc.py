
import pandas as pd
import os
import laGPy as gpr
import pyemu
import glob
import sys
import shutil
import datetime
from sklearn.cluster import KMeans
import numpy as np
import argparse
import re
import warnings

nmax_outer = 20
nmax_inner = 100

restart = False #if continuing from last outer iter
num_workers = 100
max_infill = 100
min_infill_pool = 100
pop_size = 100
tmpl_in = "template_inner"

# parser = argparse.ArgumentParser()
# parser.add_argument('--output-dir', type=str)
# parser.add_argument('--seed', type=int)
# #parser.add_argument('--master-host', type=str)
# args = parser.parse_args()
# port = 4200 + args.seed
# output_dir = args.output_dir
# #master_host = args.master_host

output_dir = '.'
port = 4020

def inner_opt(iitidx):
    sys.path.insert(0, tmpl_in)
    from forward_gprun import ppw_worker as ppw_function 
    pyemu.os_utils.start_workers(tmpl_in, "pestpp-mou", 
                                 "./pest/prams_bbgo.pst", num_workers=num_workers, 
                                 worker_root=".", master_dir="./inner_"+str(iitidx), port=port,
                                 ppw_function=ppw_function)
    sys.path.remove(tmpl_in)

    #delete some files to save space
    file_formats_to_delete = ["*.gp", "*.trimmed.archive.summary.csv"]
    for file_format in file_formats_to_delete:
        files_to_delete = glob.glob(os.path.join(f"./inner_{iitidx}", file_format))
        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)

    return sorted([d for d in os.listdir() if d.startswith("inner_") and os.path.isdir(d)], key=lambda x: int(x.split("_")[1]))
    
def outer_sweep(oitidx):   
    if oitidx == 0:
        shutil.copy(os.path.join("template_outer", "pest", "gp.lhs.dv_pop.csv"), 
                    os.path.join("template_outer", "pest", "infill.dv_pop.csv"))

    memdir = os.environ.get('MEMDIR', '/dev/shm')
    worker_base = os.path.join(memdir, f"prams_bbgo_outer_{oitidx}")
    
    pyemu.os_utils.start_workers("template_outer", "pestpp-mou", 
                               "./pest/prams_bbgo.pst", num_workers=num_workers, 
                               worker_root=worker_base, 
                               master_dir="./outer_"+str(oitidx), port=port)

    # if os.path.exists(worker_base):
    #     shutil.rmtree(worker_base)
    
    # sys.path.remove("template_outer")

    return sorted([d for d in os.listdir() if d.startswith("outer_") and os.path.isdir(d)], key=lambda x: int(x.split("_")[1]))

def get_dirlist():
    inner_dirs = sorted([d for d in os.listdir() if d.startswith("inner_") and os.path.isdir(d)], key=lambda x: int(x.split("_")[1]))
    outer_dirs = sorted([d for d in os.listdir() if d.startswith("outer_") and os.path.isdir(d)], key=lambda x: int(x.split("_")[1]))

    return inner_dirs, outer_dirs

def inner_prep(inner_dirs, outer_dirs):
    curr_dv = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/pest/*0.dv_pop.csv", recursive=True)[0])
    curr_obs = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/pest/*0.obs_pop.csv", recursive=True)[0])

    #copy curr opt for ei calcs
    if outer_dirs[-1].endswith("_0"):
        curr_opt_obs = pd.read_csv(glob.glob(os.path.join(outer_dirs[-1], "pest", "*.archive.obs_pop.csv"), recursive=True)[0])
        curr_opt_dv = pd.read_csv(glob.glob(os.path.join(outer_dirs[-1], "pest", "*.archive.dv_pop.csv"), recursive=True)[0])
    else:
        curr_opt_obs = pd.read_csv(glob.glob(os.path.join(outer_dirs[-1], "outer_repo", "*.archive.obs_pop.csv"), recursive=True)[0])
        curr_opt_dv = pd.read_csv(glob.glob(os.path.join(outer_dirs[-1], "outer_repo", "*.archive.dv_pop.csv"), recursive=True)[0])
    
    curr_opt_obs.to_csv(os.path.join(tmpl_in, "gp_model", "curr_opt.csv"), index=False)

    #copy previous outer repo update dv and obs files
    if outer_dirs[-1].endswith("_0"):      
        restart_dv = pd.read_csv(os.path.join(tmpl_in, "pest", "starter.dv_pop.csv"))
        # shutil.copy(glob.glob(os.path.join(outer_dirs[-1], "*0.dv_pop.csv"), recursive=True)[0], os.path.join(tmpl_in, "initial.dv_pop.csv"))
        # shutil.copy(glob.glob(os.path.join(outer_dirs[-1], "*0.obs_pop.csv"), recursive=True)[0], os.path.join(tmpl_in, "initial.obs_pop.csv"))
    else:
        restart_dv = pd.read_csv(os.path.join(tmpl_in, "pest", "initial.dv_pop.csv"))

    # restart_dv = pd.concat([curr_opt_dv, restart_dv], ignore_index=True)
    restart_dv = restart_dv.loc[~restart_dv.duplicated(subset=restart_dv.columns.difference(['real_name']), keep='first')]
    restart_dv.to_csv(os.path.join(tmpl_in, "pest", "initial.dv_pop.csv"), index=False)
        # shutil.copy(os.path.join("template_repo_update", "merged.dv_pop.csv"), os.path.join(tmpl_in, "initial.dv_pop.csv"))
        # shutil.copy(os.path.join("template_repo_update", "merged.obs_pop.csv"), os.path.join(tmpl_in, "initial.obs_pop.csv"))

    #remove existing gp files and training data from template dir
    for file in glob.glob(os.path.join(tmpl_in, "gp_model", "*.gp")) + \
                glob.glob(os.path.join(tmpl_in, "gp_model", "*dv_training.csv")) + \
                glob.glob(os.path.join(tmpl_in, "gp_model", "*obs_training.csv")):
        os.remove(file)

    #update training data
    if outer_dirs[-1].endswith("_0"):
        training_dv, training_obs = curr_dv, curr_obs
    else:
        training_dv = pd.concat([pd.read_csv(glob.glob(os.path.join(inner_dirs[-1], "gp_model", "*0.dv_training.csv"), recursive=True)[0]), 
                                 curr_dv], ignore_index=True)
        training_obs = pd.concat([pd.read_csv(glob.glob(os.path.join(inner_dirs[-1], "gp_model", "*0.obs_training.csv"), recursive=True)[0]), 
                                  curr_obs], ignore_index=True)
        training_dv = training_dv[curr_dv.columns]
        training_obs = training_obs[curr_obs.columns]

    #create training data and GP model
    esc1_2 = pd.read_csv(os.path.join("template", "model", "preproc", "wel-decvar", "wel", "pumping-bore-data.csv"))
    esc1_2 = esc1_2.loc[esc1_2['esc']!=3]['name'].str.lower().values

    training_dv = training_dv.drop(columns=esc1_2)
    training_dv.to_csv(os.path.join(tmpl_in, "gp_model", f"gp_0.dv_training.csv"), index=False)
    training_obs.to_csv(os.path.join(tmpl_in, "gp_model", f"gp_0.obs_training.csv"), index=False)

    X = training_dv.drop(columns=['real_name']).values
    totpenalties = training_obs.drop(columns=['real_name'])['total_penalties'].values
    gpr.buildGP(X, totpenalties, fname=os.path.join(tmpl_in, "gp_model", f"gp_obj.gp"), kernel='matern32')

    sw_flux = training_obs.drop(columns=['real_name'])['sw_flux'].values
    gpr.buildGP(X, sw_flux, fname=os.path.join(tmpl_in, "gp_model", f"gp_sw.gp"), kernel='matern32')

    print(f"\n{datetime.datetime.now()}: GP training dataset saved. \n")

def update_outer_repo(outer_dirs):
    base_path = os.path.join(".", outer_dirs[-2], "pest")
    if len(outer_dirs) > 2:
        base_path = os.path.join(".", outer_dirs[-2], "outer_repo")
    
    prev_dv_file = glob.glob(os.path.join(base_path, "*.archive.dv_pop.csv"), recursive=True)
    prev_dv = pd.read_csv(prev_dv_file[0])
    prev_obs_file = glob.glob(os.path.join(base_path, "*.archive.obs_pop.csv"), recursive=True)
    prev_obs = pd.read_csv(prev_obs_file[0])

    curr_dv = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/pest/*0.dv_pop.csv", recursive=True)[0])
    curr_obs = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/pest/*0.obs_pop.csv", recursive=True)[0])

    for file_type in ["dv_pop", "obs_pop"]:
        merged_file = pd.concat([prev_dv, curr_dv] if file_type == "dv_pop" else [prev_obs, curr_obs], ignore_index=True)
        merged_file.drop_duplicates(subset='real_name', inplace=True)
        merged_file.to_csv(os.path.join(".", "template_repo_update", "pest", f"merged.{file_type}.csv"), index=False)

    #run pestpp mou in pareto sorting mode
    pyemu.os_utils.start_workers("template_repo_update", "pestpp-mou", 
                                 "./pest/outer_repo.pst", num_workers=1, port=port,
                                 worker_root=".", master_dir="temp")

    #copy outer repo update files to outer dir
    subdir = os.path.join(outer_dirs[-1], "outer_repo")
    if os.path.exists(subdir):
        os.remove(subdir)
    os.mkdir(subdir)

    outer_repo_sumlist = glob.glob(os.path.join("temp", "pest", "outer_repo.pareto*"), recursive=True)
    for name in outer_repo_sumlist:
        shutil.copy(name, subdir)
        
    outer_repo_sumlist = glob.glob(os.path.join("temp", "pest", "outer_repo.archive*"), recursive=True)
    for name in outer_repo_sumlist:
        shutil.copy(name, subdir)
    
    #clean up temp directory
    shutil.rmtree("temp")

def prep_templates():

    def write_external_files(pst_file):
        with open(pst_file, 'r') as f:
            lines = f.readlines()
            
        # Add ./pest/ prefix to external files
        for i, line in enumerate(lines):
            if any(section in line for section in ['* parameter groups external', 
                                                 '* parameter data external',
                                                 '* observation data external',
                                                 '* model input external',
                                                 '* model output external']):
                # Next line contains the filename
                if i + 1 < len(lines):
                    filename = lines[i + 1].strip()
                    if not filename.startswith('./pest/'):
                        lines[i + 1] = f'./pest/{filename}\n'
        
        with open(pst_file, 'w') as f:
            f.writelines(lines)


    print(f"\n{datetime.datetime.now()}: prepping templates \n")
          
    pst_files = glob.glob(os.path.join('template', 'pest', '*.pst'))
    if len(pst_files) != 1:
        raise ValueError("There should be exactly one .pst file in the template directory.")
    
    #prep outer iter pst file
    print(f"\n{datetime.datetime.now()}: prepping outer template \n")
    pst = pyemu.Pst(pst_files[0])
    if os.path.exists('template_outer'):
        shutil.rmtree('template_outer')
    shutil.copytree('template', 'template_outer', ignore=shutil.ignore_patterns('gp_model'))

    pst.model_command = 'python forward_pbrun.py'
    pst.write(os.path.join('template_outer', 'pest', os.path.basename(pst_files[0])), version=2)
    write_external_files(os.path.join('template_outer', 'pest', os.path.basename(pst_files[0])))
    print(f"\n{datetime.datetime.now()}: outer template prepped \n")

    #prep outer repo update template
    print(f"\n{datetime.datetime.now()}: prepping repo update template \n")
    pst = pyemu.Pst(pst_files[0])
    if os.path.exists('template_repo_update'):
        shutil.rmtree('template_repo_update')
    shutil.copytree('template', 'template_repo_update', ignore=shutil.ignore_patterns('gp_model'))

    pst.model_command = 'python forward_pbrun.py'
    pst.pestpp_options['mou_dv_population_file'] = './pest/merged.dv_pop.csv'
    pst.pestpp_options['mou_obs_population_restart_file'] = './pest/merged.obs_pop.csv'
    pst.write(os.path.join('template_repo_update', 'pest', "outer_repo.pst"), version=2)
    write_external_files(os.path.join('template_repo_update', 'pest', os.path.basename(pst_files[0])))
    print(f"\n{datetime.datetime.now()}: repo update template prepped \n")

    #prep inner iter pst file
    print(f"\n{datetime.datetime.now()}: prepping inner template \n")
    pst = pyemu.Pst(pst_files[0])
    if os.path.exists('template_inner'):
        shutil.rmtree('template_inner')
    shutil.copytree('template', 'template_inner', ignore=shutil.ignore_patterns('model'))

    pst.control_data.noptmax = nmax_inner
    obs = pst.observation_data
    obs.loc['total_penalties_var', 'obgnme'] = 'l_obj'
    pst.model_command = 'python forward_gprun.py'
    pst.pestpp_options['mou_objectives'] = 'total_penalties, total_penalties_var'
    pst.pestpp_options['opt_constraint_groups'] = 'less_than'
    pst.pestpp_options['mou_save_population_every'] = 1
    pst.pestpp_options['mou_population_size'] = pop_size
    pst.pestpp_options['mou_max_archive_size'] = 100
    pst.pestpp_options['mou_dv_population_file'] = './pest/initial.dv_pop.csv'
    pst.pestpp_options['mou_pso_dv_bound_handling'] = 'reperturb'
    pst.write(os.path.join('template_inner', 'pest', os.path.basename(pst_files[0])), version=2)
    write_external_files(os.path.join('template_inner', 'pest', os.path.basename(pst_files[0])))
    print(f"\n{datetime.datetime.now()}: inner template prepped \n")

def parse_all_io(inner_dirs):
    csvfiles = sorted(glob.glob(f"{inner_dirs[-1]}/pest/*[0-999].dv_pop.csv", recursive=True), 
                      key=lambda x: int(x.split(".dv")[0].split(".")[1]))
    all_dv_list = []
    for file in csvfiles:
        generation = int(file.split(".dv")[0].split(".")[1])
        df = pd.read_csv(file).assign(generation=generation)
        df = df[['generation'] + [col for col in df.columns if col != 'generation']] 
        all_dv_list.append(df)
    all_dv = pd.concat(all_dv_list, ignore_index=True)
    # all_dv.to_csv(os.path.join(inner_dirs[-1], "dv.summary.csv"), index=False)
    all_dv.drop(columns=['generation'], inplace=True)

    csvfiles = sorted(glob.glob(f"{inner_dirs[-1]}/pest/*[0-999].obs_pop.csv", recursive=True), 
                      key=lambda x: int(x.split(".obs")[0].split(".")[1]))
    all_obs_list = []
    for file in csvfiles:
        generation = int(file.split(".obs")[0].split(".")[1])
        df = pd.read_csv(file).assign(generation=generation)
        df = df[['generation'] + [col for col in df.columns if col != 'generation']] 
        all_obs_list.append(df)
    all_obs = pd.concat(all_obs_list, ignore_index=True)
    all_obs.to_csv(os.path.join(inner_dirs[-1], "pest", "obs.summary.csv"), index=False)
    all_obs.drop(columns=['generation'], inplace=True)

    return all_dv, all_obs

def resample(inner_dirs, outer_dirs):
    #get current training dv and obs dataset
    training_dv = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/gp_model/gp_0.dv_training.csv", recursive=True)[0])
   
    #get all dv and obs visited in inner iters
    all_dv, all_obs = parse_all_io(inner_dirs)
    inner_pareto_summary = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/pest/*.pareto.summary.csv", recursive=True)[0])

    dv_parname = [col for col in all_dv.columns if col in training_dv.columns and col != 'real_name']
    duplicates = pd.merge(all_dv[['real_name'] + dv_parname], training_dv[dv_parname], on=dv_parname, how='inner')
    inner_pareto = inner_pareto_summary[~inner_pareto_summary['member'].isin(duplicates['real_name'])]

    n_infill = 0
    iter_n = max(inner_pareto_summary['generation'])
    infill_pool = pd.DataFrame(columns = inner_pareto_summary.columns.values)

    while n_infill < max_infill and iter_n >= 0:
        infill_sort_gen = inner_pareto[(inner_pareto["generation"] == iter_n) & (~inner_pareto['member'].isin(infill_pool['member']))].drop_duplicates(subset='member')
        max_front_idx = max(infill_sort_gen['nsga2_front'])

        front_idx = 1

        while n_infill < max_infill and front_idx <= max_front_idx:
            infill_sort_front = infill_sort_gen[(infill_sort_gen['nsga2_front'] == front_idx) & (~infill_sort_gen['member'].isin(infill_pool['member']))].drop_duplicates(subset='member')
            size_to_fill = max_infill - n_infill
            
            if not infill_sort_front.empty:
                if size_to_fill < infill_sort_front.shape[0]:
                    infill_sort_front = infill_sort_front.sort_values(by='nsga2_crowding_distance', ascending=False)
                    infill_sort_front = infill_sort_front.head(size_to_fill)
                if infill_pool.empty:
                    infill_pool = infill_sort_front.copy()
                else:
                    infill_pool = pd.concat([infill_pool, infill_sort_front], ignore_index=True)
                
                    n_infill = infill_pool.shape[0]
            else:
                front_idx += 1 
               
        iter_n -= 1

    if infill_pool.shape[0] < max_infill:
        print(f"\n{datetime.datetime.now()}: WARNING: not enough infill points to fill pool. Continuing... \n")
    
    infill_pool_dv = all_dv[all_dv['real_name'].isin(infill_pool['member'].values)]
    infill_pool_dv.to_csv(os.path.join("template_outer", "pest", "infill.dv_pop.csv"), index=False)
    
    print(f"\n{datetime.datetime.now()}: infill ensemble saved \n")

    fixedpumps = [col for col in infill_pool_dv.columns 
                if col not in training_dv.columns and col != 'real_name']
    for pump in fixedpumps:
        training_dv[pump] = infill_pool_dv[pump].iloc[0]
    training_dv = training_dv[infill_pool_dv.columns]

    #sampling from decision space for restart dv file
    augmented_training_dv = pd.concat([training_dv, infill_pool_dv], ignore_index=True)
    kmeans = KMeans(n_clusters=pop_size).fit(augmented_training_dv.drop(columns='real_name'))
    restart_pool_dv = pd.DataFrame(kmeans.cluster_centers_, columns=training_dv.drop(columns='real_name').columns)
    restart_pool_dv.insert(0, 'real_name', [f'gen={max(inner_pareto["generation"]) * len(inner_dirs)}_restart={i+1}' for i in range(len(restart_pool_dv))])    
    restart_pool_dv.to_csv(os.path.join(tmpl_in, "pest", "initial.dv_pop.csv"), index=False)

    print(f"\n{datetime.datetime.now()}: restart population saved \n")
    
def get_outer_outputs(outer_dir,output_dir):
    outer_run_dir = os.path.join(output_dir, os.path.basename(os.getcwd()), outer_dir)
    os.makedirs(outer_run_dir, exist_ok=True)
    
    exclude_suffixes = ['.archive.summary.csv', '.lhs.dv_pop.csv', 
                        '.lineage.csv', 'starter.dv_pop.csv', '0.archive.dv_pop.csv',
                        '0.archive.obs_pop.csv', 'infill.dv_pop.csv']
    csv_files = [
        file for file in glob.glob(os.path.join(outer_dir, 'pest', '*.csv'))
        if not any(os.path.basename(file).endswith(suffix) for suffix in exclude_suffixes)
    ]
    for file in csv_files:
        shutil.copy(file, outer_run_dir)

    outer_repo_path = os.path.join(outer_dir, 'outer_repo')
    if os.path.exists(outer_repo_path) and os.path.isdir(outer_repo_path):
        shutil.copytree(outer_repo_path, os.path.join(outer_run_dir, 'outer_repo'))

def get_inner_outputs(inner_dir, output_dir):
    inner_run_dir = os.path.join(output_dir, os.path.basename(os.getcwd()), inner_dir)
    os.makedirs(inner_run_dir, exist_ok=True)
    
    exclude_suffixes = ['.trimmed.archive.summary.csv', '.lhs.dv_pop.csv', '.lineage.csv']
    exclude_patterns = [
        r'.*\.\d+\.dv_pop\.csv$',
        r'.*\.\d+\.obs_pop\.csv$', 
        r'.*\.\d+\.archive\.dv_pop\.csv$',
        r'.*\.\d+\.archive\.obs_pop\.csv$'
    ]
    csv_files = [
        file for file in glob.glob(os.path.join(inner_dir, 'pest', '*.csv'))
        if not any(os.path.basename(file).endswith(suffix) for suffix in exclude_suffixes) and
        not any(re.match(pattern, os.path.basename(file)) for pattern in exclude_patterns)
    ]

    for file in csv_files:
        shutil.copy(file, inner_run_dir)

def intermediate_processing_1():
    inner_dirs, outer_dirs = get_dirlist()
    if len(outer_dirs) > 1:
        update_outer_repo(outer_dirs)
    
    inner_dirs, outer_dirs = get_dirlist()  
    inner_prep(inner_dirs, outer_dirs)

def intermediate_processing_2():

    inner_dirs, outer_dirs = get_dirlist()  
    resample(inner_dirs, outer_dirs)

if __name__ == "__main__":

    #intermediate_processing_1()
    intermediate_processing_2()




