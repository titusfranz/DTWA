using DifferentialEquations: DiffEqArrayOperator, LinearExponential, solve, ODEProblem, MagnusGL6
import DifferentialEquations
using SciMLBase: AbstractDiffEqOperator
using LinearAlgebra

mutable struct TestDiffEqOperator{T,AType<:AbstractVector{T},F} <: AbstractDiffEqOperator{T}
    field::AType
    du_cache::F
end

Base.size(L::TestDiffEqOperator) = size(L.field)
Base.size(L::TestDiffEqOperator, i) = size(L.field, 1)


function (L::TestDiffEqOperator)(u,p,t::Number)
    update_coefficients!(L,u,p,t)
    du = field × u
    du
end

function (L::TestDiffEqOperator)(du,u,p,t::Number)
    update_coefficients!(L,u,p,t)
    # L.du_cache === nothing && error("Can only use inplace TestDiffEqOperator if du_cache is given.")
    du_cache = L.du_cache
    # fill!(du,zero(first(du)))
    du .= field × u
    println("Doing")
    println(du)
    du
end

function update_coefficients!(L::TestDiffEqOperator,u,p,t)
    # TODO: Make type-stable via recursion
end

isconstant(L::TestDiffEqOperator) = true
LinearAlgebra.ishermitian(L::TestDiffEqOperator) = false
LinearAlgebra.opnorm(L::TestDiffEqOperator, p::Real=2) = 1

function (Base.:*)(L::TestDiffEqOperator, a::Number)
    L.field .= a * L.field
    L
end
Base.:*(a::Number, L::TestDiffEqOperator) = (L.field .= a * L.field; L)

field = [1.0, 0.0, 0.0]
A = TestDiffEqOperator(field, nothing)

s = [0.0, 0.0, 1.0]
A(s, s, nothing, 0.1)
A * 0.3
0.3 * A
A
prob = ODEProblem(A, [0.0, 0.0, 1.0], (0.0, 2π * 1.0))
sol = solve(prob, DifferentialEquations.CayleyEuler(),dt=1/4)

using ExponentialUtilities: arnoldi

arnoldi(A, s)