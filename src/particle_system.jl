@with_kw struct ParticleSystem{T}
    m::T = 87 * 1.661e-27  # in kg
    coupling::T = -32e9 * 6.626e-34 * 1e-12 * 1e12  # in kg um^6 / us^2
    k_b::T = 1.38e-23
    range::Int64 = 6
    temperature::T = 0.0

    c_m::T = coupling / m  # in um^6/us^2
end

function sum_forces!(force::AbstractVector{T}, pos_1::AbstractVector{T}, pos_2::AbstractVector{T},
    particle_system::ParticleSystem{T}) where T
    range = particle_system.range
    force .= force .- range * particle_system.c_m .* (pos_1 .- pos_2) ./ norm(pos_1 .- pos_2)^(range + 2) 
end

function sum_forces!(force::AbstractVector{T}, dist::AbstractVector{T},
    particle_system::ParticleSystem{T}) where T
    range = particle_system.range
    force .= force .- range * particle_system.c_m .* dist ./ norm(dist)^(range + 2) 
end

const dist_particle_system = zeros(3)
const force_particle_system = zeros(3)

function hamiltonian_explosian!(dv::Array{T, 2}, v::Array{T, 2}, u::Array{T, 2},
    system::ParticleSystem{T}, t::T) where T

    for i in 1:size(u, 2)
        force_particle_system .= 0
        for j in 1:size(u, 2)
            if i == j
                continue
            end
            pos_1 = @view u[:, i]
            pos_2 = @view u[:, j]
            dist_particle_system .= pos_1 .- pos_2
            sum_forces!(force_particle_system, dist_particle_system, system)
        end
        dv[:, i] .= force_particle_system
    end
end

function get_velocity(N::Int64; system=ParticleSystem())
    sigma = sqrt(system.k_b * system.temperature / system.m)
    gaussian_distribution = Normal(0.0, sigma)
    multivariate = product_distribution([gaussian_distribution, gaussian_distribution, gaussian_distribution])
    return rand(multivariate, N)
end

function solve_particle_system(
    positions::Matrix{T}, times::AbstractVector{T};
    system=ParticleSystem()
    ) where T
    N = size(positions)[2]
    velocities = get_velocity(N, system=system)
    prob = SecondOrderODEProblem(
        hamiltonian_explosian!, velocities, positions, (times[1], times[end]), system
        )
    return solve(prob, DPRKN6())
end

function get_J(sol::DifferentialEquations.ODESolution, t)
    pos = sol(t).x[2]
    J = get_J(pos, PowerLaw(6, 60e3))
    return @SMatrix[J]
end

function get_J_itp(
    positions, times;
    system::ParticleSystem=ParticleSystem()
    )

    sol = solve_particle_system(positions, times, system=system)
    itp = interpolate([get_J(sol, t) for t in times], BSpline(Linear()))
    f_itp = Interpolations.scale(itp, times)
    return (t) -> f_itp(t).data[1]
end
