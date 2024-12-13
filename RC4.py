#QUESTION 1


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

A1, B1, C1 = 15600, 7540, 20140
A2, B2, C2 = 18760, 2750, 18610
A3, B3, C3 = 17610, 14630, 13480
A4, B4, C4 = 19170, 610, 18390

t1, t2, t3, t4 = 0.07074, 0.07220, 0.07690, 0.07242
c = 299792.458  

def equations(vars):
    x, y, z, d = vars
    
    r1 = np.sqrt((x - A1)**2 + (y - B1)**2 + (z - C1)**2) - c * (t1 - d)
    r2 = np.sqrt((x - A2)**2 + (y - B2)**2 + (z - C2)**2) - c * (t2 - d)
    r3 = np.sqrt((x - A3)**2 + (y - B3)**2 + (z - C3)**2) - c * (t3 - d)
    r4 = np.sqrt((x - A4)**2 + (y - B4)**2 + (z - C4)**2) - c * (t4 - d)
    
    return [r1, r2, r3, r4]

initial_guess = [0, 0, 6370, 0]

solution = root(equations, initial_guess)

x, y, z, d = solution.x

print(f"Receiver position: x = {x:.3f} km, y = {y:.3f} km, z = {z:.3f} km")
print(f"Time correction: d = {d:.6f} seconds")


## QUESTION 2
def distance(x, y, z, A, B, C):
    return np.sqrt((x - A)**2 + (y - B)**2 + (z - C)**2)

def linear_system(x, y, z, d):
    u_x = (x - A1) / distance(x, y, z, A1, B1, C1)
    u_y = (y - B1) / distance(x, y, z, A1, B1, C1)
    u_z = (z - C1) / distance(x, y, z, A1, B1, C1)
    
    u_x2 = (x - A2) / distance(x, y, z, A2, B2, C2)
    u_y2 = (y - B2) / distance(x, y, z, A2, B2, C2)
    u_z2 = (z - C2) / distance(x, y, z, A2, B2, C2)
    
    u_x3 = (x - A3) / distance(x, y, z, A3, B3, C3)
    u_y3 = (y - B3) / distance(x, y, z, A3, B3, C3)
    u_z3 = (z - C3) / distance(x, y, z, A3, B3, C3)

    u_x4 = (x - A4) / distance(x, y, z, A4, B4, C4)
    u_y4 = (y - B4) / distance(x, y, z, A4, B4, C4)
    u_z4 = (z - C4) / distance(x, y, z, A4, B4, C4)
    
    A_matrix = np.array([[u_x, u_y, u_z, 1],
                         [u_x2, u_y2, u_z2, 1],
                         [u_x3, u_y3, u_z3, 1]])
    
    b_vector = np.array([c * (t1 - d), c * (t2 - d), c * (t3 - d)])
    
    return A_matrix, b_vector

initial_guess = [0, 0, 6370, 0]

def solve_quadratic_system(x0, y0, z0, d0):
    A_matrix, b_vector = linear_system(x0, y0, z0, d0)
    solution = np.linalg.lstsq(A_matrix, b_vector, rcond=None)  
    x_sol, y_sol, z_sol, d_sol = solution[0]
    
    print(f"Receiver position: x = {x_sol:.3f} km, y = {y_sol:.3f} km, z = {z_sol:.3f} km")
    print(f"Time correction: d = {d_sol:.6f} seconds")
    return x_sol, y_sol, z_sol, d_sol

solve_quadratic_system(*initial_guess)


# QUESTION 4
rho = 26570  
c = 299792.458  
x_receiver = 0  
y_receiver = 0  
z_receiver = 6370  
d = 0.0001  

phi = np.array([np.pi/6, np.pi/4, np.pi/3, np.pi/2])  
theta = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/6])  

A = rho * np.cos(phi) * np.cos(theta)
B = rho * np.cos(phi) * np.sin(theta)
C = rho * np.sin(phi)

R = np.sqrt((A - x_receiver)**2 + (B - y_receiver)**2 + (C - z_receiver)**2)

t = d + R / c

print("Satellite Positions (A, B, C):")
for i in range(4):
    print(f"Satellite {i+1}: A = {A[i]:.2f} km, B = {B[i]:.2f} km, C = {C[i]:.2f} km")

print("\nSatellite Ranges (R_i) and Travel Times (t_i):")
for i in range(4):
    print(f"Satellite {i+1}: R = {R[i]:.2f} km, t = {t[i]:.6f} seconds")


delta_t = 1e-8  

def calculate_range(A, B, C, x_receiver, y_receiver, z_receiver):
    return np.sqrt((A - x_receiver)**2 + (B - y_receiver)**2 + (C - z_receiver)**2)

def calculate_travel_time(R, d, c):
    return d + R / c

R = calculate_range(A, B, C, x_receiver, y_receiver, z_receiver)
t = calculate_travel_time(R, d, c)

delta_t_variations = np.array([delta_t, -delta_t, delta_t, -delta_t])  

t_perturbed = t + delta_t_variations

R_perturbed = (t_perturbed - d) * c  # R_i = (t_i - d) * c

def compute_position_change(R_perturbed, R):
    delta_R = R_perturbed - R
    delta_position = np.linalg.norm(delta_R)
    return delta_position

delta_position = compute_position_change(R_perturbed, R)

EMF = delta_position / (delta_t * c)  
print(f"Change in position (Δx, Δy, Δz): {delta_position:.2f} meters")
print(f"Error Magnification Factor (EMF): {EMF:.2f}")

EMF_values = []
for i in range(4):
    delta_t_variation = np.zeros(4)
    delta_t_variation[i] = delta_t  
    t_perturbed = t + delta_t_variation
    R_perturbed = (t_perturbed - d) * c
    delta_position = compute_position_change(R_perturbed, R)
    EMF_values.append(delta_position / (delta_t * c))

condition_number = max(EMF_values)
print(f"Condition number of the problem: {condition_number:.2f}")


#QUESTION 5
phi_tightly_grouped = np.array([np.pi / 6 * (1 + 0.05), np.pi / 6 * (1 + 0.03), np.pi / 6 * (1 + 0.01), np.pi / 6 * (1 + 0.04)])
theta_tightly_grouped = np.array([np.pi / 2 * (1 + 0.04), np.pi / 2 * (1 + 0.02), np.pi / 2 * (1 + 0.03), np.pi / 2 * (1 + 0.05)])

A_tightly_grouped = rho * np.cos(phi_tightly_grouped) * np.cos(theta_tightly_grouped)
B_tightly_grouped = rho * np.cos(phi_tightly_grouped) * np.sin(theta_tightly_grouped)
C_tightly_grouped = rho * np.sin(phi_tightly_grouped)

R_tightly_grouped = calculate_range(A_tightly_grouped, B_tightly_grouped, C_tightly_grouped, x_receiver, y_receiver, z_receiver)
t_tightly_grouped = calculate_travel_time(R_tightly_grouped, d, c)

delta_t_variations_tightly = np.array([delta_t, -delta_t, delta_t, -delta_t])  
t_perturbed_tightly = t_tightly_grouped + delta_t_variations_tightly
R_perturbed_tightly = (t_perturbed_tightly - d) * c

delta_position_tightly = compute_position_change(R_perturbed_tightly, R_tightly_grouped)

EMF_tightly = delta_position_tightly / (delta_t * c)
print(f"Tightly Grouped Satellites - Change in position (Δx, Δy, Δz): {delta_position_tightly:.2f} meters")
print(f"Tightly Grouped Satellites - Error Magnification Factor (EMF): {EMF_tightly:.2f}")

phi_loose = np.array([np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])  
theta_loose = np.array([np.pi / 2, np.pi / 3, np.pi / 4, np.pi / 6])  

A_loose = rho * np.cos(phi_loose) * np.cos(theta_loose)
B_loose = rho * np.cos(phi_loose) * np.sin(theta_loose)
C_loose = rho * np.sin(phi_loose)

R_loose = calculate_range(A_loose, B_loose, C_loose, x_receiver, y_receiver, z_receiver)
t_loose = calculate_travel_time(R_loose, d, c)

t_perturbed_loose = t_loose + delta_t_variations_tightly
R_perturbed_loose = (t_perturbed_loose - d) * c

delta_position_loose = compute_position_change(R_perturbed_loose, R_loose)

EMF_loose = delta_position_loose / (delta_t * c)
print(f"Loosely Grouped Satellites - Change in position (Δx, Δy, Δz): {delta_position_loose:.2f} meters")
print(f"Loosely Grouped Satellites - Error Magnification Factor (EMF): {EMF_loose:.2f}")

print(f"EMF comparison: Tightly Grouped = {EMF_tightly:.2f}, Loosely Grouped = {EMF_loose:.2f}")
