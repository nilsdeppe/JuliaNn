using ModelingToolkit, DifferentialEquations, Plots

# This example is based on the SciML tutorial:
# https://docs.sciml.ai/Overview/stable/getting_started/first_simulation/#Our-Problem:-Simulate-the-Lotka-Volterra-Predator-Prey-Dynamics

# Define our state variables: state(t) = initial condition
@independent_variables t
@variables x(t)=1 y(t)=1 z(t)=2

# Define our parameters
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0

# Define our differential: takes the derivative with respect to `t`
D = Differential(t)

# Define the differential equations
eqs = [D(x) ~ α * x - β * x * y, D(y) ~ -γ * y + δ * x * y, z ~ x + y]

# Bring these pieces together into an ODESystem with independent variable t
@mtkbuild sys = ODESystem(eqs, t)

# Convert from a symbolic to a numerical problem to simulate
tspan = (0.0, 10.0)
prob = ODEProblem(sys, [], tspan)

# Solve the ODE
sol = solve(prob, Vern9(), reltol = 1e-10, abstol = 1e-15)

# Plot the solution
p1 = plot(sol, title = "Rabbits vs Wolves")
p2 = plot(sol, idxs = z, title = "Total Animals")

plot(p1, p2, layout = (2, 1))
