# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 12:49:39 2014

@author: stuart
"""

import numpy as np

def alpha_n(v, typical_potential, resting_potential):
    dim_v = typical_potential*v + resting_potential
    inds = np.abs(dim_v+5.5e-2)>1e-8
    val = np.zeros_like(v)
    val[inds] = 1e4*(dim_v[inds]+5.5e-2) / (1-np.exp(-1e2*(dim_v[inds]+5.5e-2)))
    val[np.logical_not(inds)] = 100
    return val


def alpha_m(v, typical_potential, resting_potential):
    dim_v = typical_potential*v + resting_potential
    inds = np.abs(dim_v+4e-2)>1e-8
    val = np.zeros_like(v)
    val[inds] = 1e5*(dim_v[inds]+4e-2) / (1-np.exp(-1e2*(dim_v[inds]+4e-2)))
    val[np.logical_not(inds)] = 1000
    return val

def alpha_h(v, typical_potential, resting_potential):
    dim_v = typical_potential*v + resting_potential
    return 7e1 * np.exp(-5e1*(dim_v+6.5e-2))

def beta_n(v, typical_potential, resting_potential):    
    dim_v = typical_potential*v + resting_potential
    return 1.25e2*np.exp(-1.25e1*(dim_v+6.5e-2))
    
def beta_m(v, typical_potential, resting_potential):    
    dim_v = typical_potential*v + resting_potential
    return 4e3 * np.exp(-5.56e1*(dim_v+6.5e-2))
    
def beta_h(v, typical_potential, resting_potential):    
    dim_v = typical_potential*v + resting_potential
    return 1e3 / (1 + np.exp(-1e2*(dim_v+3.5e-2)))

def n_inf(v, typical_potential, resting_potential):
    return alpha_n(v, typical_potential, resting_potential) * tau_n(v, typical_potential, resting_potential)

def m_inf(v, typical_potential, resting_potential):
    return alpha_m(v, typical_potential, resting_potential) * tau_m(v, typical_potential, resting_potential)

def h_inf(v, typical_potential, resting_potential):
    return alpha_h(v, typical_potential, resting_potential) * tau_h(v, typical_potential, resting_potential)

def tau_n(v, typical_potential, resting_potential):
    return 1 / (alpha_n(v, typical_potential, resting_potential) + beta_n(v, typical_potential, resting_potential))

def tau_m(v, typical_potential, resting_potential):
    return 1 / (alpha_m(v, typical_potential, resting_potential) + beta_m(v, typical_potential, resting_potential))

def tau_h(v, typical_potential, resting_potential):
    return 1 / (alpha_h(v, typical_potential, resting_potential) + beta_h(v, typical_potential, resting_potential))


def dndt(v, n, typical_potential, resting_potential):
    return (alpha_n(v, typical_potential, resting_potential)*(1-n) - beta_n(v, typical_potential, resting_potential)*n)
def dmdt(v, m, typical_potential, resting_potential):
    return (alpha_m(v, typical_potential, resting_potential)*(1-m) - beta_m(v, typical_potential, resting_potential)*m)
def dhdt(v, h, typical_potential, resting_potential):
    return (alpha_h(v, typical_potential, resting_potential)*(1-h) - beta_h(v, typical_potential, resting_potential)*h)
    
