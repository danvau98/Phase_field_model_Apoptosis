import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.core.system import *
from mpi4py import MPI
import logging
import vtk_io as io
from adaptive_step import *
from initialisation import *
from utils import * 

# Initialize MPI and logging
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_problem(Lx, Ly, Nx, Ny, beta, ebar, alpha, k, gamma, tau, dealias, dtype):
    import numpy as np

    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)
    
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx, Lx), dealias=dealias)
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly, Ly), dealias=dealias)

    x = dist.local_grids(xbasis)
    x_field = dist.Field(name='x_field', bases=(xbasis,))
    x_field['g'] = x
    y = dist.local_grids(ybasis)
    y_field = dist.Field(name='y_field', bases=(ybasis,))
    y_field['g'] = y

    phi = dist.Field(name='phi', bases=(xbasis,ybasis))
    ep = dist.Field(name='ep', bases=(xbasis,ybasis))
    ep_prime = dist.Field(name='ep_prime', bases=(xbasis,ybasis))
    sig = dist.Field(name='sig', bases=(xbasis,ybasis))
    r = dist.Field(name='r', bases=(xbasis,ybasis))
    theta = dist.Field(name='theta', bases=(xbasis,ybasis))
    f_1 = dist.Field(name='f_1', bases=(xbasis,ybasis))
    f_2 = dist.Field(name='f_2', bases=(xbasis,ybasis))
    #f = dist.Field(name='f', bases=(xbasis,ybasis))
    f = dist.VectorField(coords, name='f', bases=(xbasis,ybasis))
        
    dx = lambda A: d3.Differentiate(A, coords['x'])
    dy = lambda A: d3.Differentiate(A, coords['y'])
    
    problem = d3.IVP([phi, ep, ep_prime, sig, r, theta, f, f_1, f_2], namespace=locals())
    problem.add_equation("(dt(phi)) + (phi/(2*tau)) = (1.0/tau) * (-dx(ep * ep_prime * dy(phi)) + dy(ep * ep_prime * dx(phi)) + div((ep**2)*grad(phi)) + (phi**2) + (phi * r) - (phi**3) + ((phi**2)/2) - ((phi**2) * r) )")
    problem.add_equation("dt(sig) - (lap(sig)) - (beta * dt(phi)) = -r*(phi - (phi**2))")
    problem.add_equation("r = - (0.9/np.pi) * np.arctan((k * sig))")
    problem.add_equation("theta = np.arctan(dy(phi)/dx(phi))")
    problem.add_equation("ep = (ebar/4.0e-02) + (ebar * np.cos(theta))")
    problem.add_equation("ep_prime = -ebar*np.sin(theta)")
    problem.add_equation("f_1 - (tau * dt(phi)) = (r * (phi - (phi**2)))")
    problem.add_equation("f_2 - dt(sig) + (beta * dt(phi)) = (r * (phi - (phi**2)))")
    #problem.add_equation("f = (f_1) + (f_2)")
    problem.add_equation("f = (f_1 * grad(phi)) + (f_2 * grad(sig))")
    
    # No flux BCs
    x0, y0 = 0, 0  # Center of the ellipsoid
    radius = 3.0
    phi['g'] = circle_with_noise(x, y,x0, y0, radius)
    theta['g'] = 0.0
    f_1['g'] = 0.0
    f_2['g'] = 0.0
    
    return problem, dist, xbasis, ybasis, phi, ep, ep_prime, sig, r, theta, f, x, y

def main(beta, ebar, alpha, k, gamma, tau):
    import numpy as np
    Lx, Ly = 4.0, 4.0
    Nx, Ny = 512, 512

    max_dt = 1e-03
    min_dt = 1e-14
    initial_dt = 1e-06
    stop_sim_time = 0.5

    tol = 1e-04 * 4.0 # increase this
    adapt_fac = 0.90

    dealias = 3/2
    timestepper = d3.RK443 
    dtype = np.float64

    problem, dist, xbasis, ybasis, phi, ep, ep_prime, sig, r, theta, f, x, y = setup_problem(Lx, Ly, Nx, Ny, beta, ebar, alpha, k, gamma, tau, dealias, dtype)
    initialize_fields(x, y, Lx, Ly, Nx, Ny, phi, ep, ep_prime, sig, r, theta, beta, ebar, alpha, k, gamma, tau)

    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    n_snaps = 100 # Number of frames generated
    analysis = solver.evaluator.add_file_handler('analysis_apoptosis', sim_dt=stop_sim_time/n_snaps, max_writes=100000)
    
    analysis.add_task(phi, name='phi')
    analysis.add_task(sig, name='sig')

    try:
        logger.info('Starting main loop')
        current_dt = initial_dt
        while solver.proceed:
            solver.step(current_dt)
            if (solver.iteration-1) % 10 == 0:
                output_time = solver.sim_time
                local_error = error_calc(solver, current_dt, phi, ep, ep_prime, sig, r, theta)
                global_error = comm.allreduce(local_error, MPI.MAX)
                current_dt = adapt_dt(global_error, tol, current_dt, adapt_fac, max_dt, min_dt)
                logger.info('Iteration=%i, Time=%e, dt=%e' % (solver.iteration, output_time, current_dt))
    except Exception as e:
        logger.error('Exception details: %s' % str(e))
        raise
    finally:
        solver.log_stats()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AC_global_const_2D with parameters.')
    parser.add_argument('beta', type=float, help='Parameter beta')
    parser.add_argument('ebar', type=float, help='Parameter ebar')
    parser.add_argument('alpha', type=float, help='Parameter alpha')
    parser.add_argument('k', type=float, help='Parameter k')
    parser.add_argument('gamma', type=float, help='Parameter gamma')
    parser.add_argument('tau', type=float, help='Parameter tau')
    
    args = parser.parse_args()

    main(args.beta, args.ebar, args.alpha, args.k, args.gamma, args.tau)