# File for running simulation
import subprocess
import numpy as np
from scipy.stats import qmc

def main():
    n_samples = 1 #3

    #         beta,    ebar,         alpha,    k          gamma     tau
    p_sets = [1.5e-00, 4.0e-04 * 0.5, 9.0e-01, 1.0 * 10.0, 1.0, 3.0e-04]

    for i in range(n_samples):
        # Command to run AC_global_const_2D.py
        cmd1 = [
            'mpiexec', '-n', '32', 'python3', 'KRF1.py',
            str(p_sets[0]), str(p_sets[1]), str(p_sets[2]), str(p_sets[3]), str(p_sets[4]), str(p_sets[5]) 
        ]

        # Command to run plot_growth.py
        cmd2 = [
            'mpiexec', '-n', '1', 'python3', 'plot_run.py', 'analysis_apoptosis/*.h5',
            str(p_sets[0]), str(p_sets[1]), str(p_sets[2]), str(p_sets[3]), str(p_sets[4]), str(p_sets[5])
        ]

        # Run the first command
        process1 = subprocess.run(cmd1, check=True)

        try:
            # Run the second command
            process2 = subprocess.run(cmd2, check=True)
            continue
        except subprocess.CalledProcessError as e:
            print(f"Error running cmd2: {e}")
            continue

if __name__ == "__main__":
    main()
