@deprecate
function hamiltonian!(ds, s, p::ArrayPartition, t)
    J = p.x[1]
    h = p.x[2]
    anisotropy = p.x[3]
    hamiltonian!(ds, s, J, h, anisotropy, t)
end


function DTWAProblem(hamiltonian!, Ψ, saveat, p, )
    J = p.x[1]
    prob = ODEProblem(hamiltonian!, permutedims(dtwa_state(Ψ)), (saveat[1], saveat[end]), p,
        dense=false,
        dt=1/maximum(maximum(J, dims=1))/4,
        force_dtmin=true,
        adaptive=true,
        saveat=saveat
    )

    EnsembleProblem(
        prob,
        prob_func = (prob, i, repeat) -> dtwa_prob_func(prob, Ψ),
        output_func = (sol, i) -> (vcat(mean.(sol.u, dims=1)...), false),
        reduction = reduction,
        u_init = zeros(Float64, (length(saveat), 3)),
    )
end

DTWAProblem(hamiltonian!, p_0, tspan, p, s::Sequence) = DTWAProblem(
    hamiltonian!, p_0, tspan, p, s.durations
)


function solve_dtwa(ensemble_prob, trajectories; kwargs...)
    tspan = ensemble_prob.prob.tspan
    nsteps = size(ensemble_prob.u_init, 1)
    solve(
        ensemble_prob,
        Tsit5(),
        # AutoTsit5(Rosenbrock23()),
        # EnsembleSerial(),
        EnsembleDistributed(),
        trajectories=trajectories,
        batch=8;
        #saveat=collect(LinRange(tspan[1], tspan[2], nsteps));
        kwargs...
    )
end

function solve_dtwa(ensemble_prob, tracectories, s::Sequence; kwargs...)
    cb = IterativeCallback(s)
    solve_dtwa(ensemble_prob, tracectories; callback=cb, kwargs...)
end
