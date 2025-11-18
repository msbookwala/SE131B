"""
Optimization of a two-bar truss structure.
The structure is loaded at the top joint, and supported at the two bottom joints.
The design variables are multipliers of the cross sectional areas of the two bars.
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
h = 2
L = 2
E = 2.1e11
rho = 8000
F = 2.0e6

A_INITIAL = 0.1
SMALLEST_AREA = 0.0001
MAXTIPD = L / 1000

# 2. Define the function to create the model of the structure.
# 
# This function defines the model of the structure, based on the values of the
# design variables. The design variables are multipliers of the initial area,
# and so their values are around 1.0.
def two_bar_model(dvs):
    m = model.create(2)
    model.add_joint(m, 1, [0.0, 0.0])
    model.add_joint(m, 2, [0.0, h])
    model.add_joint(m, 3, [L, 0.0])
    model.add_support(m["joints"][1], freedoms.TRANSLATION_DOFS)
    model.add_support(m["joints"][3], freedoms.TRANSLATION_DOFS)
    model.add_load(m["joints"][2], freedoms.U1, F)
    bar_connectivities = [
        [1, 2],
        [2, 3],
    ]
    for k, c in enumerate(bar_connectivities):
        s = section.truss_section(f"s{k}", E=E, A=A_INITIAL * dvs[k], rho=rho)
        model.add_truss_member(m, k, c, s)
    return m

# 3. Define the function that computes the design responses 
# of the structure from the current values of the design variables. 

# First some helper functions. This one to compute the current mass of the structure.
def current_mass(m, dvs):
    mass = 0.0
    for member in m["truss_members"].values():
        A = A_INITIAL * dvs[member["mid"]]
        c = member["connectivity"]
        i, j = m["joints"][c[0]], m["joints"][c[1]]
        sect = member["section"]
        e_x, _, h = geometry.member_2d_geometry(i, j)
        mass += A * h * sect["rho"]
    return mass


# Helper function to compute the maximum deflection.
def max_tip_deflection(m):
    mtd = 0.0
    for j in m["joints"].values():
        mtd = max(mtd, max(abs(j["displacements"])))
    return mtd

# This function computes the design responses. Static response is obtained.
# Current mass of the structure and the maximum tip deflection are computed as
# the design responses.
def solve(dvs):
    m = two_bar_model(dvs)
    model.number_dofs(m)
    model.solve_statics(m)
    drs = (current_mass(m, dvs), max_tip_deflection(m))
    return drs

# 4. Report the initial performance of the structure

# Initial guess -- these are the fractions of the initial area.
dvs0 = numpy.array([1.0, 1.0])

drs = solve(dvs0)
initial_mass = drs[0]
print("Initial areas: ", A_INITIAL * dvs0)
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
bounds = [(SMALLEST_AREA / A_INITIAL, None) for _ in dvs0]


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

print("Optimal areas: ", A_INITIAL * dvs)
print("Optimal mass: ", drs[0])
print("Optimal deflection: ", drs[1])

# 8. Plot the optimal structure.
m = two_bar_model(dvs)
plots.setup(m)
plots.plot_members(m, max_linewidth=6)
plots.show(m)

