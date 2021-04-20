###############################################################################
#### Single spin ##############################################################
###############################################################################

# SingleSpin(x::T, y::T, z::T) where T = [x, y, z]

###############################################################################
#### Product states ###########################################################
###############################################################################

ProductState(states::Array{Array{Float64, 1}, 1}) where T = hcat(states...)
ProductState(N::Int, state::AbstractVector{Float64}) where T = ProductState([state for i in 1:N])


###############################################################################
#### magnetization ############################################################
###############################################################################

function magnetization(s::AbstractMatrix, x::Symbol=:x)
    match = Dict(:x => 1, :y => 2, :z=>3)

    N = size(s, 2)
    1/N * sum(s[match[x], :])
end

magnetization(sol::OrdinaryDiffEq.ODECompositeSolution, t) = magnetization(sol(t)')
magnetization(sim::EnsembleSolution, t) = magnetization(timepoint_mean(sim, t)')
###############################################################################
#### dtwa states ############################################################
###############################################################################

function dtwa_state(x::AbstractVector)
    res = normalize(x)
    nullspace(permutedims(x))[:, 1]
    vec(res .+ sum(rand([-1, 1], 1, 2)  .* nullspace(permutedims(res)), dims=2))
end

function dtwa_state(x::AbstractMatrix)
    res = similar(x)
    for i in 1:size(x, 2)
        res[:, i] .= dtwa_state(x[:, i])
    end
    res
end


# end #module SpinState
