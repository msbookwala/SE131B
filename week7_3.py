"""
Optimization of a 17-bar truss structure.
The design variables are multipliers of the inner dimensions and
the thickneses of the square hollow-tube cross sections of the bars.
The objective is to minimize the mass of the structure, while keeping the
maximum deflection below a specified limit.
"""
from math import pi
import numpy
from scipy.optimize import minimize
from pystran import model
from pystran import section
from pystran import freedoms
from pystran import geometry
from pystran import plots

# 1. Define some constants.
h = 2.5
L = 3
E = 2.0e11
rho = 8000
W = 6000

bottom_chord = [[1, 2], [2, 3], [3, 4], [4, 5]]
top_chord = [[6, 7], [7, 8], [8, 9], [9, 10]]
verticals = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]
diagonals = [[1, 7], [3, 9], [3, 7], [5, 9]]

INNER_DIM_MIN = 0.012 # M
THICKNESS_MIN = 0.0015 # M
MAXTIPD = 0.01
NBARS = 17
NGROUPS = 4
dvs0 = numpy.ones(2 * NGROUPS)
def _area(d, t):
    return (d + 2*t)**2 - d**2 

# 2. Define the function to create the model of the structure.
# 
# This function defines the model of the structure, based on the values of the
# design variables. The design variables are multipliers of the initial area,
# and so their values are around 1.0.
def truss_model(dvs):
    m = model.create(2)
    model.add_joint(m, 1, [-2*L, -h])
    model.add_joint(m, 2, [-L, -h])
    model.add_joint(m, 3, [0, -h])
    model.add_joint(m, 4, [L, -h])
    model.add_joint(m, 5, [2*L, -h])
    model.add_joint(m, 6, [-2*L, 0])
    model.add_joint(m, 7, [-L, 0])
    model.add_joint(m, 8, [0, 0])
    model.add_joint(m, 9, [L, 0])
    model.add_joint(m, 10, [2*L, 0])
    model.add_support(m["joints"][1], freedoms.U2)
    model.add_support(m["joints"][5], freedoms.U2)
    model.add_support(m["joints"][8], freedoms.U1)
    model.add_load(m["joints"][2], freedoms.U2, -W)
    model.add_load(m["joints"][3], freedoms.U2, -W)
    model.add_load(m["joints"][4], freedoms.U2, -W)
    g = 0
    d, t = dvs[g] * INNER_DIM_MIN, dvs[g + NGROUPS] * THICKNESS_MIN
    A = _area(d, t)
    s = section.truss_section(f"s{g}", E=E, A=A, rho=rho)
    for k, c in enumerate(bottom_chord):
        model.add_truss_member(m, f"bot{k}", c, s)
    g = 1
    d, t = dvs[g] * INNER_DIM_MIN, dvs[g + NGROUPS] * THICKNESS_MIN
    A = _area(d, t)
    s = section.truss_section(f"s{g}", E=E, A=A, rho=rho)
    for k, c in enumerate(top_chord):
        model.add_truss_member(m, f"top{k}", c, s)
    g = 2
    d, t = dvs[g] * INNER_DIM_MIN, dvs[g + NGROUPS] * THICKNESS_MIN
    A = _area(d, t)
    s = section.truss_section(f"s{g}", E=E, A=A, rho=rho)
    for k, c in enumerate(verticals):
        model.add_truss_member(m, f"vert{k}", c, s)
    g = 3
    d, t = dvs[g] * INNER_DIM_MIN, dvs[g + NGROUPS] * THICKNESS_MIN
    A = _area(d, t)
    s = section.truss_section(f"s{g}", E=E, A=A, rho=rho)
    for k, c in enumerate(diagonals):
        model.add_truss_member(m, f"diag{k}", c, s)
    return m

# m = truss_model(dvs0)
# plots.setup(m)
# plots.plot_joint_ids(m)
# plots.plot_members(m)
# plots.plot_member_ids(m)
# plots.plot_applied_forces(m)
# plots.plot_translation_supports(m)
# plots.show(m)

# 3. Define the function that computes the design responses 
# of the structure from the current values of the design variables. 

# First some helper functions. This one to compute the current mass of the structure.
def current_mass(m, dvs):
    mass = 0.0
    for member in m["truss_members"].values():
        c = member["connectivity"]
        i, j = m["joints"][c[0]], m["joints"][c[1]]
        sect = member["section"]
        e_x, _, h = geometry.member_2d_geometry(i, j)
        mass += sect['A'] * h * sect["rho"]
    return mass


# Helper function to compute the maximum deflection.
def max_defl(m):
    mtd = 0.0
    for j in m["joints"].values():
        mtd = max(mtd, max(abs(j["displacements"])))
    return mtd

# This function computes the design responses. Static response is obtained.
# Current mass of the structure and the maximum tip deflection are computed as
# the design responses.
def solve(dvs):
    m = truss_model(dvs)
    model.number_dofs(m)
    model.solve_statics(m)
    drs = (current_mass(m, dvs), max_defl(m))
    return drs

# 4. Report the initial performance of the structure

drs = solve(dvs0)
initial_mass = drs[0]
print("Initial inner dim: ", INNER_DIM_MIN * dvs0[0:4])
print('Initial thickness: ', THICKNESS_MIN * dvs0[NGROUPS:])
print("Initial mass: ", initial_mass)
print("Initial deflection: ", drs[1])

# 5. Define the normalized objective function and the normalized constraints

# Objective function is the normalized mass.
def objective(dvs):
    drs = solve(dvs)
    return drs[0] / initial_mass


# Constraint function must return a non-negative value for each constraint.
# This is the constraint on the maximum deflection.
def constraint_max_deflection(dvs):
    drs = solve(dvs)
    mtd = drs[1]
    return (MAXTIPD - mtd) / MAXTIPD

# Define an array of constraints. The only constraint 
# here is an inequality.
cons = [
    {"type": "ineq", "fun": constraint_max_deflection},
]

# Define lower bounds for the design variables. There are no upper bounds.
bounds = [(1.0, None) for _ in dvs0]


# 6. Invoke the optimization function. 

solution = minimize(
    objective,
    dvs0,
    method="SLSQP",
    bounds=bounds,
    constraints=cons,
    options={"ftol": 1e-7, "maxiter": 1000, "disp": True},
)

# 7. Report the solution of the optimization problem.

# Retrieve the values of the design variables, and compute the design responses
# for the optimal design variables.
dvs = solution.x
drs = solve(dvs)

print("Optimal inner dim: ", INNER_DIM_MIN * dvs[0:4])
print('Optimal thickness: ', THICKNESS_MIN * dvs[NGROUPS:])
print("Optimal mass: ", drs[0])
print("Optimal deflection: ", drs[1])

# 8. Plot the optimal structure.
m = truss_model(dvs)
model.number_dofs(m)
model.solve_statics(m)
plots.setup(m)
plots.plot_members(m, max_linewidth=16)
plots.plot_axial_forces(m)
plots.show(m)

