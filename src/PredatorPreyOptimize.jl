using DifferentialEquations,
  Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using ForwardDiff, Plots
import Compat, Optim, Enzyme

# In this example we create data from the Volterra-Lotka equation, which
# is used as an example of a (simple) predator-prey. The SciML example
# only searches for one parameter, but we want to learn all 4 parameters.
# This presents a (significantly) bigger challenge because the optimization
# space is larger and the system can become stiff if the parameters end up
# being different by a couple orders of magnitude (or if one switches sign).
# To deal with this we ultimately use a box-constrained minimization algorithm,
# specifically we use BFGSB to prevent us from wandering outside the allowed
# space. We can then use the Verner-9(8) ODE solver to get high-precision
# internal loops of the minimization problem. This allows us to find the
# optimal parameters to about almost machine precision. An important
# modification necessary to the SciML example is that we take the square root
# in the loss function. Without that we can only achieve sqrt(eps)~1e-7
# precision on the parameters. We of course also need very high precision
# on the integration method if we wish to get consistent answers that aren't
# subject to truncation in the ODE algorithm.
#
# This code is based on the ideas from the SciML tutorial:
# https://docs.sciml.ai/Overview/stable/getting_started/fit_simulation/

function pred_prey!(du, u, p, t)
  x, y = u
  alpha, beta, delta, gamma = p
  # Why does the tutorial do du[1] = dx?
  du[1] = alpha * x - beta * x * y
  du[2] = -delta * y + gamma * x * y
  return nothing
end

# Set up the ODE problem to solve with the "expected" parameters
#
# Initial condition
u_0 = [1.0, 1.0]
# Simulation interval
t_span = (0.0, 10.0)
# LV equation parameter. p = [α, β, δ, γ]
p_true = [1.0, 1.5, 2.5, 1.3]

problem = ODEProblem(pred_prey!, u_0, t_span, p_true)

# Solve the ODE
t_data = 0:10
initial_sol = solve(problem, reltol = 1.0e-13, abstol = 1.0e-30, saveat = 1)
x_data = zeros(Float64, size(initial_sol.u))
y_data = zeros(Float64, size(initial_sol.u))
for i in 1:size(initial_sol.u)[1]
  x_data[i] = initial_sol.u[i][1]
  y_data[i] = initial_sol.u[i][2]
end
# Julia is _insane_ and we need to use the ' operator to convert our 1d arrays
# (a vector) into a 1xN matrix.
xy_data = vcat(x_data', y_data')

# Plot the initial solution that we will try to search for.
initial_plot = plot(initial_sol, title = "Rabbits vs Wolves")
scatter!(
  initial_plot,
  t_data,
  hcat(x_data, y_data),
  label = ["x Data" "y Data"],
)

# Now that we have set up a condition for whose parameters we want to
# optimize, we need to set up the loss function.
#
# The loss function is called by the optimizer at each step. Since the optimizer
# is working in Greek space, we will be passed the parameters, then we solve the
# ODE over time, and then we compute the difference between the data and the
# solution using some suitable norm (e.g. L2).
function loss_learn_parameters(new_parameters)
  # Create the new ODEProblem with the current parameters
  # new_problem = remake(problem, p = new_parameters)
  new_problem = ODEProblem(pred_prey!, u_0, t_span, new_parameters)
  # Solve it, dumping at the times we have data.
  sol = solve(new_problem, Vern9(), reltol = 1e-13, abstol = 1e-30, saveat = 1)
  # For the lass, just sum up the squares of the differences and take the
  # square root. The square root is important since otherwise we are counting
  # the digits/accuracy of the square of the problem.
  loss =
    sqrt(sum(abs2, collect(Compat.Iterators.flatten(sol.u .- initial_sol.u))))
  return loss
end

function callback_learn_parameters(state, l)
  display(l)
  newprob = remake(problem, p = state.u)
  sol = solve(newprob, saveat = 0.1)
  plt = plot(
    sol,
    ylim = (0, 6),
    label = ["Current x Prediction" "Current y Prediction"],
  )
  scatter!(plt, t_data, hcat(x_data, y_data), label = ["x Data" "y Data"])
  display(plt)
  return false
end

function learn_parameters()
  # This function learns the α,β,γ,δ parameters of the Lotka-Volterra equation.
  #
  # Set up the optimization problem with our loss function and initial guess
  adtype = AutoForwardDiff()
  p_guess = [1.5, 1.0, 2.71, 1.5]
  optf = OptimizationFunction((x, _) -> loss_learn_parameters(x), adtype)
  optprob = OptimizationProblem(
    optf,
    p_guess,
    lb = [0.5, 0.5, 0.5, 0.5],
    ub = [5.5, 5.5, 5.5, 5.5],
  )

  # Optimize the ODE parameters for best fit to our data
  pfinal = solve(
    optprob,
    Optim.BFGS(),
    # callback = callback_learn_parameters,
    abstol = 1.0e-60,
    reltol = 1.0e-13,
    maxiters = 200,
  )
  α, β, γ, δ = round.(pfinal, digits = 1)
  print("Final params:   ", pfinal)
  print("Correct params: ", p_true)
  return nothing
end

function learn_parameters_no_box()
  function loss_learn_parameters_no_box(new_parameters)
    lower_bound = 0.5
    upper_bound = 5.5
    scale_factor = 1.0e2
    bound_loss = (
      scale_factor * abs2(
        min(0, (new_parameters[1] - lower_bound)) +
        min(0, (new_parameters[2] - lower_bound)) +
        min(0, (new_parameters[3] - lower_bound)) +
        min(0, (new_parameters[4] - lower_bound)),
      ) +
      scale_factor * abs2(
        max(0, (new_parameters[1] - upper_bound)) +
        max(0, (new_parameters[2] - upper_bound)) +
        max(0, (new_parameters[3] - upper_bound)) +
        max(0, (new_parameters[4] - upper_bound)),
      )
    )
    # Because the ODE solve is unstable for values outside of our bounds,
    # we don't do the ODE solve at all in those cases.
    if bound_loss > 0.0
      return bound_loss
    end

    # Create the new ODEProblem with the current parameters
    # new_problem = remake(problem, p = new_parameters)
    new_problem = ODEProblem(pred_prey!, u_0, t_span, new_parameters)
    # Solve it, dumping at the times we have data.
    sol =
      solve(new_problem, Vern9(), reltol = 1e-13, abstol = 1e-30, saveat = 1)
    # For the loss, just sum up the squares of the differences and take the
    # square root. The square root is important since otherwise we are counting
    # the digits/accuracy of the square of the problem.
    ode_loss = sqrt(sum(abs2, sol .- xy_data))
    loss = ode_loss
    return loss
  end
  # This function learns the α,β,γ,δ parameters of the Lotka-Volterra equation.
  #
  # Set up the optimization problem with our loss function and initial guess
  adtype = AutoForwardDiff()
  p_guess = [1.5, 1.0, 2.71, 1.5]
  optf = OptimizationFunction((x, _) -> loss_learn_parameters_no_box(x), adtype)
  optprob = OptimizationProblem(optf, p_guess)

  # Optimize the ODE parameters for best fit to our data
  pfinal = solve(
    optprob,
    Optim.BFGS(),
    # callback = callback_learn_parameters,
    abstol = 1.0e-60,
    reltol = 1.0e-13,
    maxiters = 200,
  )
  α, β, γ, δ = round.(pfinal, digits = 1)
  print("Final params:   ", pfinal)
  print("Correct params: ", p_true)
  return nothing
end

function learn_initial_conditions(true_parameters)
  # Now that we have set up a condition for whose parameters we want to
  # optimize, we need to set up the loss function.
  #
  # The loss function is called by the optimizer at each step. Since the
  # optimizer is working in Greek space, we will be passed the parameters, then
  # we solve the ODE over time, and then we compute the difference between the
  # data and the solution using some suitable norm (e.g. L2).
  function loss_initial_conditions(new_initial_conditions)
    # Create the new ODEProblem with the current parameters
    # new_problem = remake(problem, p = new_parameters)
    new_problem =
      ODEProblem(pred_prey!, new_initial_conditions, t_span, true_parameters)
    # Solve it, dumping at the times we have data.
    sol =
      solve(new_problem, Vern9(), reltol = 1e-13, abstol = 1e-30, saveat = 1)
    # For the lass, just sum up the squares of the differences and take the
    # square root. The square root is important since otherwise we are counting
    # the digits/accuracy of the square of the problem.
    loss =
      sqrt(sum(abs2, collect(Compat.Iterators.flatten(sol.u .- initial_sol.u))))
    return loss
  end

  u_0_guess = [0.5, 1.5]
  # This function learns the initial conditions of the Lotka-Volterra equation.
  #
  # Set up the optimization problem with our loss function and initial guess
  adtype = AutoForwardDiff()
  optf = OptimizationFunction((x, _) -> loss_initial_conditions(x), adtype)
  optprob =
    OptimizationProblem(optf, u_0_guess, lb = [0.1, 0.1], ub = [5.5, 5.5])

  # Optimize the ODE parameters for best fit to our data
  pfinal = solve(
    optprob,
    Optim.BFGS(),
    # callback = callback,
    abstol = 1.0e-60,
    reltol = 1.0e-13,
    maxiters = 200,
  )
  print("Final initial conditions:   ", pfinal)
  print("Correct initial conditions: ", u_0)
  return nothing
end

# Finally, we try to learn both the initial conditions _and_ the parameters.
# This case is a lot more challenging because of a lot of local minima in the
# parameter space. If you choose initial guess "close enough", then the solution
# converges. I haven't done a detailed study of how close "close enough" is, but
# the values right now seem to work, while if you use guesses for the initial
# conditions of [0.5, 1.5] then you end up stuck in a local minimum.
function learn_parameters_and_initial_conditions()
  # Now that we have set up a condition for whose parameters we want to
  # optimize, we need to set up the loss function.
  #
  # The loss function is called by the optimizer at each step. Since the
  # optimizer is working in Greek space, we will be passed the parameters, then
  # we solve the ODE over time, and then we compute the difference between the
  # data and the solution using some suitable norm (e.g. L2).
  function loss_parameters_and_initial_conditions(
    new_initial_conditions_and_parameters,
  )
    # Create the new ODEProblem with the current parameters
    # new_problem = remake(problem, p = new_parameters)
    new_problem = ODEProblem(
      pred_prey!,
      new_initial_conditions_and_parameters[1:2],
      t_span,
      new_initial_conditions_and_parameters[3:end],
    )
    # Solve it, dumping at the times we have data.
    sol =
      solve(new_problem, Vern9(), reltol = 1e-13, abstol = 1e-30, saveat = 1)
    # For the lass, just sum up the squares of the differences and take the
    # square root. The square root is important since otherwise we are counting
    # the digits/accuracy of the square of the problem.
    loss =
      sqrt(sum(abs2, collect(Compat.Iterators.flatten(sol.u .- initial_sol.u))))
    return loss
  end

  initial_guess = [0.8, 1.1, 1.5, 1.0, 2.71, 1.5]
  # This function learns the initial conditions of the Lotka-Volterra equation.
  #
  # Set up the optimization problem with our loss function and initial guess
  adtype = AutoForwardDiff()
  optf = OptimizationFunction(
    (x, _) -> loss_parameters_and_initial_conditions(x),
    adtype,
  )
  optprob = OptimizationProblem(
    optf,
    initial_guess,
    lb = [0.1, 0.1, 0.5, 0.5, 0.5, 0.5],
    ub = [5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
  )

  # Optimize the ODE parameters for best fit to our data
  pfinal = solve(
    optprob,
    Optim.BFGS(),
    # callback = callback,
    abstol = 1.0e-60,
    reltol = 1.0e-13,
    maxiters = 200,
  )
  print(
    "Final loss function: ",
    loss_parameters_and_initial_conditions(pfinal.u),
    "\n",
  )
  print("Final initial conditions:   ", pfinal.u[1:2], pfinal.u[3:end], "\n")
  print("Correct initial conditions: ", u_0, p_true)
  return nothing
end

learn_parameters_no_box()

# learn_parameters()
# learn_initial_conditions(p_true)
# learn_parameters_and_initial_conditions()
