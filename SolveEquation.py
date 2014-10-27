# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 12:05:14 2014

@author: stuart
"""

import numpy as np
from scipy import integrate, special
import hodgkin_huxley_channels as hh
import matplotlib.pyplot as plt


def g_twiddle(N, epsilon, s, L):
    threshold = 50
    ns = np.linspace(1, N, N)
    
    BK0 = special.kn(0,ns[ns*np.pi*epsilon/L<threshold]*np.pi*epsilon/L)
    BK1 = special.kn(1,ns[ns*np.pi*epsilon/L<threshold]*np.pi*epsilon/L)
    BI0 = special.iv(0,ns[ns*np.pi*epsilon/L<threshold]*np.pi*epsilon/L)
    BI1 = special.iv(1,ns[ns*np.pi*epsilon/L<threshold]*np.pi*epsilon/L)
    
    g = np.zeros(N)

    g[ns*np.pi*epsilon/L<threshold] = s*BK1*BI1*np.pi / (L**2 * (s*BK1*BI0 + BI1*BK0))
    g[ns*np.pi*epsilon/L>=threshold] = s*np.pi/L**2*(1/(s+1))
    
    return g*ns

def input_current(t, tau):
    if tau*t < 2e-3:
        return 1
    else:
        return 0

def hh_cur(v, t, n, m, h, pars, typical_potential, resting_potential):
    v_dim = typical_potential*v + resting_potential
    return pars[0]*n**4*(v_dim-pars[3]) + pars[1]*m**3*h*(v_dim-pars[4]) + pars[2]*(v_dim-pars[5])

def hh_test(y, t, tau, typical_potential, resting_potential):
    v = np.array([y[0]])
    n = np.array([y[1]])
    m = np.array([y[2]])
    h = np.array([y[3]])
    
    dv = -hh_cur(v, t, n, m, h) + 0.05*input_current(t)
    dn = tau * hh.dndt(v, n, typical_potential, resting_potential)
    dm = tau * hh.dmdt(v, m, typical_potential, resting_potential)
    dh = tau * hh.dhdt(v, h, typical_potential, resting_potential)
    
    return np.concatenate((dv, dn, dm, dh))

def make_initial_condition(p_i, XSteps, typical_potential, resting_potential):
    n_i = hh.n_inf(p_i, typical_potential, resting_potential)
    m_i = hh.m_inf(p_i, typical_potential, resting_potential)
    h_i = hh.h_inf(p_i, typical_potential, resting_potential)
    ic = np.zeros(4*len(p_i), dtype = np.complex)
    ic[0:len(p_i)] = np.fft.fft(p_i) / XSteps
    ic[len(p_i):2*len(p_i)] = n_i
    ic[2*len(p_i):3*len(p_i)] = m_i
    ic[3*len(p_i):] = h_i
    return ic

def time_derivative(t, y, xs, C, tau, L, hh_pars, XSteps, green_fun_transform, typical_potential, resting_potential):
    
    p = y[0:XSteps]
    n = y[1*XSteps:2*XSteps]
    m = y[2*XSteps:3*XSteps]
    h = y[3*XSteps:4*XSteps]
    
    v = XSteps*np.real(np.fft.ifft(p))

    J = -hh_cur(v, t, n, m, h, hh_pars, typical_potential, resting_potential)
    J[len(p)/2-2:len(p)/2+2] += 1*input_current(t, tau)
    ji = np.fft.fft(J)/XSteps

    dP = np.zeros_like(p)
    
    dP[0] = ji[0]/C    
    dP[1:len(p)/2+1] = -(L*green_fun_transform*p[1:len(p)/2+1]-ji[1:len(p)/2+1])/C
    dP[len(p)/2+1:] = -(L*green_fun_transform[-1:0:-1]*p[len(p)/2+1:]-ji[len(p)/2+1:])/C

    dn = tau * hh.dndt(v, n, typical_potential, resting_potential)
    dm = tau * hh.dmdt(v, m, typical_potential, resting_potential)
    dh = tau * hh.dhdt(v, h, typical_potential, resting_potential)

    return np.concatenate((dP, dn, dm, dh))

def solve_fourier_transformed_eq(epsilon, s, L, XSteps):

    simulation_length = 30e-3

    C = 1./40

    tau = 1e-3
    TEnd = simulation_length / tau

    typical_potential = 25e-3
    resting_potential = -65e-3
    jb = 1000

    gk = 36
    gna = 120
    gl = 0.3
    vk  = -77*1e-3
    vna =  50*1e-3
    vl = -54.402*1e-3
    
    hh_pars = [gk, gna, gl, vk, vna, vl]

    TSteps = 2500

    xs,dx = np.linspace(L*(-1+1./XSteps), L*(1-1./XSteps), XSteps, retstep = True)
    ts,dt = np.linspace(0,TEnd,TSteps,retstep=True)
    
    initial_potential = np.zeros(XSteps)
    initial_transform = np.fft.fft(initial_potential) / XSteps

    green_fun_transform = g_twiddle(XSteps/2, epsilon, s, L)

    initial_condition = make_initial_condition(initial_potential, XSteps, typical_potential, resting_potential)

    initial_condition[0:XSteps] = initial_transform

    r = integrate.ode(time_derivative).set_integrator('vode', method='bdf', order=15)
    r.set_initial_value(initial_condition, 0)
    r.set_f_params(xs, C, tau, L, hh_pars, XSteps, green_fun_transform, typical_potential, resting_potential)

    sol = np.zeros([len(ts)+1,4*XSteps])

    i = 1
    
    while r.successful() and r.t<TEnd:
        r.integrate(r.t+dt)
        sol[i,:] = r.y
        i += 1

    p = sol[:-1,0:XSteps]

    phi = np.real(np.fft.ifft(p))

    n = sol[:-1,1*XSteps:2*XSteps]
    m = sol[:-1,2*XSteps:3*XSteps]
    h = sol[:-1,3*XSteps:4*XSteps]
    
    return xs, ts, phi, n, m, h

def calculate_speed(phi, xs, ts, t1, t2, threshold):
    try:
        t0 = np.where(phi[:,0]>threshold)[0][0]
    except IndexError:
        t0 = len(ts)
    if t0 > t2:
        t2 = np.floor(0.9*t0)
    try:
        x1 = np.where(phi[t1,:]>threshold)[0][0]
        x2 = np.where(phi[t2,:]>threshold)[0][0]
    except IndexError:
        return np.nan
    return (xs[x1]-xs[x2]) / (ts[t2]-ts[t1])

epsilon_1 = 0.01
epsilon_2 = 0.1
n_epsilons = 100

sbar_1 = 0.01
sbar_2 = 0.1
n_sbars = 2

eps = np.linspace(epsilon_1, epsilon_2, n_epsilons)
sbars = np.linspace(sbar_1, sbar_2, n_sbars)

speeds = np.zeros([len(eps), len(sbars)])

t1 = 400
t2 = 800

threshold = 0.01

L = 100
XSteps = 2**8

save_folder = "/Users/Stuart/Desktop/LargeAxonsData/"

for i in range(len(eps)):
    for j in range(len(sbars)):
        xs, ts, phi, n, m, h = solve_fourier_transformed_eq(eps[i], sbars[j], L, XSteps)
        speeds[i, j] = calculate_speed(phi, xs, ts, t1, t2, threshold)
        file_name = "ep_" + str(eps[i]) + "_sbar_" + str(sbars[j])
        head_str = ("Axon radius = " + str(eps[i]) +
                    "\nConductivity ratio = " + str(sbars[j]) + 
                    "\nL = " + str(L) + "\nXSteps = " + str(XSteps))
        np.savetxt(save_folder+file_name, phi, header=head_str, comments='#')
        






