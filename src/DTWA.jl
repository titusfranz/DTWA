module DTWA

using DataStructures: OrderedDict
using LinearAlgebra
using UnsafeArrays
using Distributions
using DifferentialEquations
using Parameters
using StaticArrays

include("mean_field.jl")
export hamiltonian!

include("spin_states.jl")
export ProductState
export magnetization
export dtwa_state

include("hamiltonian_engineering.jl")
export Sequence
export absolute_times, clostest_index, create_time_choice, create_affect!
export wahuha

include("hamiltonian_explosion.jl")
export solve_particle_system, get_velocity, get_J, VanderWaalsParticleSystem

include("dtwa_problem.jl")
export DTWAProblem, solve_dtwa, XYZSystem

end  # module DTWA
