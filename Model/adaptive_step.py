# File for adaptive stepper
# Adaptive timestep based on Richardson extrapolation
def error_calc(solver, init_dt, phi, ep, ep_prime, sig, r, theta):
    import numpy as np
    # Check if arrays are larger than 1024x1024
    #if phi['g'].shape[0] > 1024 and phi['g'].shape[1] > 1024 and sigma['g'].shape[0] > 1024 and sigma['g'].shape[1] > 1024:
        # Check if GPU is available
        #if cp.cuda.runtime.getDeviceCount() > 0:
            # Use CuPy for GPU calculations
            #xp = cp
        #else:
            # Use NumPy for CPU calculations
            #xp = np
        #xp = np
    #else:
        # Use NumPy for CPU calculations
        #xp = np
   
    xp = np
    solver_state = solver.state
    sim_time = solver.sim_time
    solver_iter = solver.iteration
    phi_save = phi['g'].copy()
    ep_save = ep['g'].copy()
    sig_save = sig['g'].copy()
    r_save = r['g'].copy()
    ep_prime_save = ep_prime['g'].copy()
    theta_save = theta['g'].copy()
 
    solver.step(init_dt)
    phi_full = xp.array(phi['g'].copy())
    sig_full = xp.array(sig['g'].copy())
 
    solver.state = solver_state
    solver.sim_time = sim_time
    solver.iteration = solver_iter
    phi['g'] = phi_save.copy()
    ep['g'] = ep_save.copy()
    sig['g'] = sig_save.copy()
    r['g'] = r_save.copy()
    ep_prime['g'] = ep_prime_save.copy()
    theta['g'] = theta_save.copy()
 
    solver.step(init_dt / 2.0)
    solver.step(init_dt / 2.0)
    phi_half = xp.array(phi['g'].copy())
    sig_half = xp.array(sig['g'].copy())
 
    solver.state = solver_state
    solver.sim_time = sim_time
    solver.iteration = solver_iter
    phi['g'] = phi_save.copy()
    ep['g'] = ep_save.copy()
    sig['g'] = sig_save.copy()
    r['g'] = r_save.copy()
    ep_prime['g'] = ep_prime_save.copy()
    theta['g'] = theta_save.copy()
 
    error_estimate_phi = xp.max(xp.abs(phi_half - phi_full))
    error_estimate_sig = xp.max(xp.abs(sig_half - sig_full))
    error_estimate = xp.max([error_estimate_phi, error_estimate_sig])
 
    return float(error_estimate)  # Ensure the result is a Python float

def adapt_dt(error_estimate, tol, init_dt, adapt_fac, max_dt, min_dt):
    if error_estimate > tol:
        out_dt = init_dt * adapt_fac
    else:
        out_dt = min(init_dt / adapt_fac, max_dt)
    out_dt = max(min(out_dt, max_dt), min_dt)
    return out_dt