import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import os
import datetime
import argparse
import glob
import shutil


#function parameters
dimensions = 6
space = [Real(0.0, 1.0, name=f'x{i}') for i in range(dimensions)]

#BGO parameters
bgo_iter = 100
n_initial_points = 100
n_calls = n_initial_points + bgo_iter # Number of function evaluations

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()
output_dir = args.output_dir
seed = args.seed


def levy(x):
    x = np.array(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3


all_inputs = []
all_outputs = []
@use_named_args(space)
def objective_function(**kwargs):
    x = np.array(list(kwargs.values()))
    value = levy(x)
    
    all_inputs.append(x)
    all_outputs.append(value)
    
    print(f"Iteration {len(all_outputs)}: Best value = {min(all_outputs):.4e}, Current value = {value:.4e}")
    
    return value


if __name__ == "__main__":

    print('Saving outputs to:', output_dir)

    start_time = datetime.datetime.now()
    print(f"\n{datetime.datetime.now()}: starting Classic BGO run \n")

    x0 = pd.read_csv('gp.lhs.dv_pop.csv').drop(columns=['real_name'])
    
    result = gp_minimize(
    objective_function,
    space,
    n_calls=n_calls,
    n_initial_points=0,
    n_restarts_optimizer=10,
    x0 = x0.values.tolist(),
    acq_func="EI",
    noise=1e-10,
    n_jobs = -1,
    verbose=True
    )
    print(f"\n{datetime.datetime.now()}: Classic BGO run complete \n")

    # Print the best result
    print("\nOptimization Results:")
    print(f"Best function value: {result.fun:.4e}")
    print(f"Best parameters: {result.x}")

    # Save inputs to CSV
    inputs_df = pd.DataFrame(all_inputs, columns=[f'x{i}' for i in range(dimensions)])
    inputs_df.index.name = 'iteration'
    inputs_df.to_csv('inputs_summary.csv')
    print(f"{datetime.datetime.now()}: Input values saved to 'inputs_summary.csv'\n")

    # Save outputs to CSV
    outputs_df = pd.DataFrame({
        'infill_objective_value': all_outputs,
        'minimum_objective_value': np.minimum.accumulate(all_outputs)
    })
    outputs_df.index.name = 'iteration'
    outputs_df.to_csv('outputs_summary.csv')
    print(f"{datetime.datetime.now()}: Objective values saved to 'outputs_summary.csv'\n")

    #get outputs
    #run_dir = os.path.join(output_dir, os.path.basename(os.getcwd()))
    #os.makedirs(run_dir, exist_ok=True)

    #csv_files = [file for file in glob.glob('*.csv')]
    #for file in csv_files:
    #    shutil.copy(file, run_dir)

    print(f"total run time: {datetime.datetime.now() - start_time} \n")