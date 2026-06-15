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
dimensions = 10
space = [Real(-10.0, 10.0, name=f'x{i}') for i in range(dimensions)]

#BGO parameters
bgo_iter = 5
n_initial_points = 100
n_calls = n_initial_points + bgo_iter # Number of function evaluations

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()
output_dir = args.output_dir
seed = args.seed

# output_dir = '.'

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

all_inputs = []
all_outputs = []
all_f_values = []
all_penalties = []
@use_named_args(space)
def objective_function(**kwargs):
    x = np.array(list(kwargs.values()))
    result = g07(x)
    
    # Extract the objective function value
    f_value = result["func"]
    
    # Check constraints
    constraints = [result[f"g{i+1}"] for i in range(8)]
    
    # Apply penalty for constraint violations
    penalty = 0
    for constraint in constraints:
        if constraint > 0:  # Constraint violation
            penalty += 1000 * constraint**2  # Quadratic penalty
    
    # Final objective value with penalty
    value = f_value + penalty
    
    all_inputs.append(x)
    all_outputs.append(value)
    all_f_values.append(f_value)
    all_penalties.append(penalty)
    
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
    # For each iteration, find the index of the minimum value up to that point
    min_indices = [np.argmin(all_outputs[:i+1]) for i in range(len(all_outputs))]
    
    outputs_df = pd.DataFrame({
        'infill_objective_value': all_outputs,
        'minimum_objective_value': np.minimum.accumulate(all_outputs),
        'f_values': [all_f_values[idx] for idx in min_indices],
        'penalties': [all_penalties[idx] for idx in min_indices]
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