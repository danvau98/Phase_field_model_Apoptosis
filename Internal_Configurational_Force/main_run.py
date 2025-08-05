# File for running simulation
import subprocess
import numpy as np
from scipy.stats import qmc

def main():
    #         beta,     ebar,        alpha,  k        gamma      tau
    p_sets = [1.5e-00, 4.0e-04/2.0, 9.0e-01, (0.9/np.pi) * 1.01, 1.00e+01, 3.0e-04]

    for i in range(n_samples):
        # Command to run AC_global_const_2D.py
        cmd1 = [
            'mpiexec', '-n', '8', 'python3', 'Apoptosis_config_mech.py',
            str(p_sets[0]), str(p_sets[1]), str(p_sets[2]), str(p_sets[3]), str(p_sets[4]), str(p_sets[5]) 
        ]

        # Run the first command
        process1 = subprocess.run(cmd1, check=True)

if __name__ == "__main__":
    main()
