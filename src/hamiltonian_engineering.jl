abstract type PulseBlock end

struct Sequence{T} <: PulseBlock
    J::Matrix{T}
    fields::Vector{Vector{T}}
    durations::Vector{T}
    anisotropy::Vector{T}
    state::Matrix{T}
end

function Base.repeat(s::Sequence, n::Integer)
    fields = repeat(s.fields, n) #, outer=(1, n))
    durations = repeat(s.durations, n)
    Sequence(s.J, fields, durations, s.anisotropy, s.state)
end

function Base.iterate(s::Sequence, state=1)
    state > length(s.durations) ? nothing : s.durations[state], state + 1
end

Base.getindex(s::Sequence, i::Int) = (s.fields[i], s.durations[i])

function Base.setindex!(s::Sequence, v::Tuple{Vector{Float64}, Float64}, i::Int)
    s.fields[i] = v[1]
    s.durations[i] = v[2]
end

function Base.insert!(s::Sequence, i::Int, v::Tuple{Vector{Float64}, Float64})
    insert!(s.fields, i, v[1])
    insert!(s.durations, i, v[2])
end

absolute_times(s::Sequence) = [sum(s.durations[1:(i-1)]) for i in 1:length(s.durations)]

function clostest_index(s::Sequence, t)
    abs_times = absolute_times(s)
    argmin(abs.(t .- abs_times))
end

function create_time_choice(s::Sequence)
    f = function(integrator)
        if integrator.t == sum(s.durations[1:end-1])
            return nothing
        end
        index = clostest_index(s, integrator.t)
        integrator.t +  s.durations[index]
    end
end


function create_affect!(s::Sequence)
    f = function(integrator)
        J = s.J
        index = clostest_index(s, integrator.t)
        #h = s.fields[index]
        # integrator.p = ArrayPartition(J, h)
        system = XYZSystem(J, [0.0, 0.0, 0.0], [1.0, 1.0, 0])
        integrator.f = build_hamiltonian!(system)
    end
    f
end

using DifferentialEquations


DiffEqCallbacks.IterativeCallback(s::Sequence) = IterativeCallback(
    create_time_choice(s),
    create_affect!(s),
    save_positions=(false, false)
)


function wahuha(J, h, T, anisotropy, state)
    X = h .* [1., 0., 0.]
    Y = h .* [0., 1., 0.]
    Null = [0., 0., 0.]
    fields = [Null, X, Null, -Y, Null, Y, Null, -X, Null]

    t = π/2/h
    durations = [T, t, T, t, 2*T, t, T, t, T]
    return Sequence(J, fields, durations, anisotropy, state)
end

wahuha(J, h, T, anisotropy, state, n) = repeat(wahuha(J, h, T, anisotropy, state), n)
