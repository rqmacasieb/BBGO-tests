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
dimensions = 50
space = [Real(-5.0, 10.0, name=f'x{i}') for i in range(dimensions)]

#BGO parameters
bgo_iter = 1
n_initial_points = 0
n_calls = n_initial_points + bgo_iter # Number of function evaluations

# parser = argparse.ArgumentParser()
# parser.add_argument('--output-dir', type=str)
# parser.add_argument('--seed', type=int)
# args = parser.parse_args()
# output_dir = args.output_dir
# seed = args.seed

output_dir = '.'

def rosenbrock(x):
    x = np.array(x)
    # Rosenbrock function formula: sum_{i=1}^{d-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


all_inputs = []
all_outputs = []
@use_named_args(space)
def objective_function(**kwargs):
    x = np.array(list(kwargs.values()))
    value = rosenbrock(x)
    
    all_inputs.append(x)
    all_outputs.append(value)
    
    print(f"Iteration {len(all_outputs)}: Best value = {min(all_outputs):.4e}, Current value = {value:.4e}")
    
    return value

def get_max_EI():
    print('Saving outputs to:', output_dir)

    start_time = datetime.datetime.now()
    print(f"\n{datetime.datetime.now()}: starting Classic BGO run \n")

    x0 = pd.read_csv('gp_0.dv_training.csv').drop(columns=['real_name'])
    y0 = pd.read_csv('gp_0.obs_training.csv').drop(columns=['real_name'])['func']
    
    result = gp_minimize(
    objective_function,
    space,
    n_calls=n_calls,
    n_initial_points=0,
    n_restarts_optimizer=10,
    x0 = x0.values.tolist(),
    y0 = y0.values.tolist(),
    acq_func="EI",
    noise=1e-10,
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
    inputs_df.to_csv('cbgo_inputs_summary.csv')
    print(f"{datetime.datetime.now()}: Classic BGO input values saved to 'cbgo_inputs_summary.csv'\n")

    # Save outputs to CSV
    outputs_df = pd.DataFrame({
        'infill_objective_value': all_outputs,
        'minimum_objective_value': np.minimum.accumulate(all_outputs)
    })
    outputs_df.index.name = 'iteration'
    outputs_df.to_csv('cbgo_outputs_summary.csv')
    print(f"{datetime.datetime.now()}: Classic BGO objective values saved to 'cbgo_outputs_summary.csv'\n")

    #get outputs
    #run_dir = os.path.join(output_dir, os.path.basename(os.getcwd()))
    #os.makedirs(run_dir, exist_ok=True)

    #csv_files = [file for file in glob.glob('*.csv')]
    #for file in csv_files:
    #    shutil.copy(file, run_dir)

    print(f"total Classic BGO run time: {datetime.datetime.now() - start_time} \n")

if __name__ == "__main__":
    get_max_EI()
