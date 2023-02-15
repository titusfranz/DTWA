abstract type SpinSystem end

# Spin system

struct XYZSystem{T} <: SpinSystem
    J::Matrix{T}
    h::VecOrMat{T}
    anisotropy::Vector{T}
end

get_dt(system::XYZSystem) = 1 / maximum(maximum(system.J, dims=1)) / 4

function build_hamiltonian!(system::XYZSystem)
    J, h, anisotropy = system.J, system.h, system.anisotropy
    (ds, s, p, t) -> hamiltonian!(ds, s, J, h, anisotropy, t)
end

# Spin system with finite temperature

struct XYZSystem_t <: SpinSystem
    J_t
    h_t
    anisotropy
end

function build_hamiltonian!(system::XYZSystem_t)
    J_t, h_t, anisotropy = system.J_t, system.h_t, system.anisotropy
    (ds, s, p, t) -> hamiltonian!(ds, s, J_t(t), h_t(t), anisotropy, t)
end

get_dt(system::XYZSystem_t) = 1 / maximum(maximum(system.J_t(0.0), dims=1)) / 4

function reduction(u, data, i)
    (u .+ sum(data), false)
end

function dtwa_prob_func(prob, Ψ)
    prob.u0 .= permutedims(dtwa_state(Ψ))
    return prob
end

function conserve_spin_norm(resid, u, p, t)
    for i in 1:size(u, 1)
        u_i = @view u[i, :]
        resid[i, 1] = norm(u_i) - sqrt(3.0)
    end
    resid[:, 2:end] .= 0
end

conserve_spin_callback = ManifoldProjection(conserve_spin_norm)

function DTWAProblem(system, Ψ, saveat; kwargs...)
    dt = get(kwargs, :dt, get_dt(system))
    hamiltonian! = build_hamiltonian!(system)
    Ψ_ini = permutedims(dtwa_state(Ψ))
    prob = ODEProblem(hamiltonian!, Ψ_ini, (saveat[1], saveat[end]),
        dense=false,
        dt=dt,
        force_dtmin=true,
        adaptive=true,
        saveat=saveat
    )

    EnsembleProblem(
        prob,
        prob_func=(prob, i, repeat) -> dtwa_prob_func(prob, Ψ),
        output_func=(sol, i) -> (vcat(mean.(sol.u, dims=1)...), false),
        reduction=reduction,
        u_init=zeros(Float64, (length(saveat), 3)),
    )
end

function DTWAProblem_noreduction(system::SpinSystem, Ψ, saveat; kwargs...)
    dt = get(kwargs, :dt, get_dt(system))
    hamiltonian! = build_hamiltonian!(system)
    Ψ_ini = permutedims(dtwa_state(Ψ))
    prob = ODEProblem(hamiltonian!, Ψ_ini, (saveat[1], saveat[end]),
        dense=false,
        dt=dt,
        force_dtmin=true,
        adaptive=true,
        saveat=saveat
    )

    EnsembleProblem(
        prob,
        prob_func=(prob, i, repeat) -> dtwa_prob_func(prob, Ψ),
        output_func=(sol, i) -> (sol.u, false),
        reduction=(u, data, I) -> (append!(u, data), false),
        u_init=[]
    )
end

function solve_dtwa(ensemble_prob, trajectories, alg=Tsit(); kwargs...)
    tspan = ensemble_prob.prob.tspan
    nsteps = size(ensemble_prob.u_init, 1)
    sol = DifferentialEquations.solve(
        ensemble_prob,
        alg,
        # AutoTsit5(Rosenbrock23()),
        # EnsembleSerial(),
        # EnsembleThreads(),
        # EnsembleDistributed(),
        trajectories=trajectories;
        #saveat=collect(LinRange(tspan[1], tspan[2], nsteps));
        kwargs...
    )
    sol.u ./= trajectories
end

function solve_dtwa(ensemble_prob, tracectories, s::Sequence; kwargs...)
    cb = IterativeCallback(s)
    solve_dtwa(ensemble_prob, tracectories; callback=cb, kwargs...)
end
