using Revise

using DTWA
using CSV
using DataFrames
using Distributed

addprocs()

pos = CSV.File("pos.csv") |> Tables.matrix;
pos = pos'

# J = get_J_dipolar(3.0, params)
J = get_J(pos, PowerLaw(6, 2π*267000));
system = XYZSystem(J, [0.0, 0.0, 0.0], [1.0, 1.0, -0.6])

N = size(J, 2)
s = [1.0, 0.0, 0.0]
Ψ = ProductState(N, s)

times = LinRange(0, 10, 500)
problem = DTWAProblem(system, Ψ, times)
sol = solve_dtwa(problem, 10);

dtwa_df = DataFrame(:time => times, :magn => 0.5 * sol[:, 1])
CSV.write("dtwa_simulation.csv", dtwa_df)