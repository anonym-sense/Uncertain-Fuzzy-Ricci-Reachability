# JuliaReach (ReachabilityAnalysis.jl, TMJets Taylor models) baseline and UTR-TM engine.
#
# usage: julia +lts juliareach.jl <benchmark> <mode> [budget]
#   mode = none     plain JuliaReach on the whole initial set (no partitioning)
#          uniform  JuliaReach + uniform bisection of every dimension (level by level)
#          utr      JuliaReach as the engine inside UTR: curvature-guided refinement
# Same benchmarks, alpha-path semantics (parameters and Liu drifts as constant states),
# Liu central alpha bands, and alpha* bisection (tol 1e-2) as the Python framework.
# Writes ../experiments/results/<benchmark>__jr_<mode>__curvature__path.json

using ReachabilityAnalysis, LazySets, JSON, LinearAlgebra

const SQ3PI = sqrt(3) / pi
halfwidth(s, a) = SQ3PI * s * log((1 + a) / (1 - a))
const ALPHA_MAX = 0.999

struct Bench
    name::String
    n::Int                      # physical state dim
    f!::Function                # f!(dz, z, p, t) on augmented z = (x, theta)
    th_e::Vector{Float64}       # parameter centres (theta and Liu drifts)
    th_s::Vector{Float64}       # Liu sigma
    x0lo::Vector{Float64}
    x0hi::Vector{Float64}
    ulo::Vector{Float64}        # unsafe box (+-Inf allowed), physical coords
    uhi::Vector{Float64}
    T::Float64
end

# ------------------------------------------------------------------ models
function s_curve(pocket=false)
    om, ph, A, b = 2.24, -1.12, 1.75, 1.0
    f! = function (dz, z, p, t)
        dz[1] = z[3]
        dz[2] = -z[4] * (z[2] - A * sin(om * z[1] + ph)) + b * z[5]
        dz[3] = zero(z[1]); dz[4] = zero(z[1]); dz[5] = zero(z[1])
        nothing
    end
    ulo, uhi = pocket ? ([2.3, 0.4], [3.2, 1.2]) : ([5.3, -2.3], [6.0, -1.72])
    Bench(pocket ? "s_curve_pocket" : "s_curve", 2, f!, [0.6, 3.0, 0.0], [0.006, 0.15, 0.4],
          [-0.5, -0.6], [0.0, 0.0], ulo, uhi, 10.0)
end

function van_der_pol()
    f! = function (dz, z, p, t)
        dz[1] = z[2]
        dz[2] = z[3] * (1 - z[1]^2) * z[2] - z[1] + z[4]
        dz[3] = zero(z[1]); dz[4] = zero(z[1])
        nothing
    end
    Bench("van_der_pol", 2, f!, [1.0, 0.0], [0.02, 0.05], [1.25, 2.25], [1.55, 2.35],
          [-Inf, 2.75], [Inf, Inf], 7.0)
end

function pendulum()
    g, l, m0, b0 = 9.81, 1.0, 1.0, 0.1
    # LQR gain identical to frs/systems.py (computed there; copied for exact parity)
    K = JSON.parsefile(joinpath(@__DIR__, "gains.json"))["pendulum"]
    f! = function (dz, z, p, t)
        u = -(K[1] * z[1] + K[2] * z[2])
        ml2 = z[3] * (l * l)
        dz[1] = z[2]
        dz[2] = (g / l) * sin(z[1]) - (z[4] * z[2] - u - z[5]) / ml2
        dz[3] = zero(z[1]); dz[4] = zero(z[1]); dz[5] = zero(z[1])
        nothing
    end
    Bench("pendulum", 2, f!, [1.0, 0.1, 0.0], [0.08, 0.03, 0.4], [0.3, -0.1], [0.4, 0.1],
          [-Inf, -Inf], [-0.095, Inf], 3.0)
end

function acc()
    mass, f0, f1, h, kD, kv = 1650.0, 0.1, 5.0, 1.4, 0.35, 0.9
    f! = function (dz, z, p, t)
        D, v, vl = z[1], z[2], z[3]
        u = mass * (kD * (D - h * v) + kv * (vl - v))
        drag = f0 + f1 * v + z[4] * v^2
        dz[1] = vl - v
        dz[2] = (u - drag) / mass
        dz[3] = z[5] + z[6]
        dz[4] = zero(z[1]); dz[5] = zero(z[1]); dz[6] = zero(z[1])
        nothing
    end
    Bench("acc", 3, f!, [0.25, -1.2, 0.0], [0.03, 0.15, 0.25], [38.0, 26.0, 21.5], [42.0, 27.0, 22.5],
          [-Inf, -Inf, -Inf], [12.0, Inf, Inf], 8.0)
end

function planar_quad()
    g, Jz, arm, m0 = 9.81, 0.01, 0.15, 0.5
    K = reduce(vcat, permutedims.(JSON.parsefile(joinpath(@__DIR__, "gains.json"))["planar_quad"]))
    ue = m0 * g / 2
    f! = function (dz, z, p, t)
        u1 = ue - sum(K[1, j] * z[j] for j in 1:6)
        u2 = ue - sum(K[2, j] * z[j] for j in 1:6)
        T = u1 + u2
        dz[1] = z[4]; dz[2] = z[5]; dz[3] = z[6]
        dz[4] = -(T / z[7]) * sin(z[3]) + z[8]
        dz[5] = (T / z[7]) * cos(z[3]) - g + z[9]
        dz[6] = arm * (u2 - u1) / Jz
        dz[7] = zero(z[1]); dz[8] = zero(z[1]); dz[9] = zero(z[1])
        nothing
    end
    Bench("planar_quad", 6, f!, [0.5, 0.0, 0.0], [0.02, 0.3, 0.3],
          [-1.55, -0.05, -0.02, -0.05, -0.05, -0.02], [-1.45, 0.05, 0.02, 0.05, 0.05, 0.02],
          [-0.9, -Inf, -Inf, -Inf, -Inf, -Inf], [-0.5, -0.19, Inf, Inf, Inf, Inf], 4.0)
end

function chain(N)
    k, c = 2.0, 0.8
    n = 2N
    f! = function (dz, z, p, t)
        k3, w = z[n + 1], z[n + 2]
        for i in 1:N
            left = i > 1 ? z[i] - z[i - 1] : z[i]
            a = -k * left - k3 * left^3 - c * z[N + i]
            if i < N
                right = z[i] - z[i + 1]
                a = a - k * right - k3 * right^3
            else
                a = a + w
            end
            dz[i] = z[N + i]
            dz[N + i] = a
        end
        dz[n + 1] = zero(z[1]); dz[n + 2] = zero(z[1])
        nothing
    end
    lo = zeros(n); hi = zeros(n)
    lo[N], hi[N] = 0.8, 1.0
    lo[N+1:end] .= -0.05; hi[N+1:end] .= 0.05
    U = JSON.parsefile(joinpath(@__DIR__, "gains.json"))["chain_unsafe"][string(N)]
    ulo = fill(-Inf, n); uhi = fill(Inf, n); uhi[N] = U
    Bench("chain$N", n, f!, [0.5, 0.0], [0.05, 0.1], lo, hi, ulo, uhi, 4.0)
end

function make(name)
    name == "s_curve" && return s_curve(false)
    name == "s_curve_pocket" && return s_curve(true)
    name == "van_der_pol" && return van_der_pol()
    name == "pendulum" && return pendulum()
    name == "acc" && return acc()
    name == "planar_quad" && return planar_quad()
    startswith(name, "chain") && return chain(parse(Int, name[6:end]))
    error("unknown $name")
end

# ------------------------------------------------------------------ cells
# a cell = (lo, hi) over augmented coordinates; theta coords are absolute values
# zero-width coordinates are padded by 1e-9 (TMJets normalises by the radius; a sound,
# negligible enlargement)
function root(b::Bench, a)
    lo = vcat(b.x0lo, b.th_e .- halfwidth.(b.th_s, a))
    hi = vcat(b.x0hi, b.th_e .+ halfwidth.(b.th_s, a))
    hi = max.(hi, lo .+ 1e-9)
    lo, hi
end

function unsafe_set(b::Bench, N)
    hs = HalfSpace{Float64, Vector{Float64}}[]
    for i in 1:b.n
        if isfinite(b.uhi[i])
            a = zeros(N); a[i] = 1.0; push!(hs, HalfSpace(a, b.uhi[i]))
        end
        if isfinite(b.ulo[i])
            a = zeros(N); a[i] = -1.0; push!(hs, HalfSpace(a, -b.ulo[i]))
        end
    end
    HPolyhedron(hs)
end

"Run TMJets on one cell: :safe, :hit (flowpipe meets U or is incomplete) or :error."
function status(b::Bench, lo, hi, U)
    N = length(lo)
    X0 = Hyperrectangle(low=lo, high=hi)
    prob = @ivp(z' = b.f!(z), dim: N, z(0) ∈ X0)
    try
        sol = solve(prob, tspan=(0.0, b.T), alg=TMJets(abstol=1e-10, orderT=6, orderQ=2, maxsteps=20000))
        # TMJets may stop early (step-size underflow): an incomplete flowpipe proves nothing
        tend(sol) < b.T - 1e-9 && return :hit
        for R in sol
            Z = overapproximate(R, Zonotope)
            !isdisjoint(set(Z), U) && return :hit
        end
        return :safe
    catch e
        return :error   # TMJets internal failure: proves nothing (treated as unsafe-possible)
    end
end

hits(b::Bench, lo, hi, U) = status(b, lo, hi, U) != :safe

# finite-difference directional stretching toward the unsafe coordinates (UTR split rule)
function rk4_path(b::Bench, z0; dt=0.01)
    z = copy(z0); dz = similar(z); out = [copy(z)]
    k1 = similar(z); k2 = similar(z); k3 = similar(z); k4 = similar(z)
    for _ in 1:round(Int, b.T / dt)
        b.f!(k1, z, nothing, 0.0); b.f!(k2, z .+ 0.5dt .* k1, nothing, 0.0)
        b.f!(k3, z .+ 0.5dt .* k2, nothing, 0.0); b.f!(k4, z .+ dt .* k3, nothing, 0.0)
        z .+= dt / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4); push!(out, copy(z))
    end
    out
end

function split_dim(b::Bench, lo, hi)
    c = (lo .+ hi) ./ 2; h = (hi .- lo) ./ 2
    B = [i for i in 1:b.n if isfinite(b.ulo[i]) || isfinite(b.uhi[i])]
    best, bi = -1.0, 1
    for i in eachindex(c)
        h[i] <= 0 && continue
        d = zeros(length(c)); d[i] = 1e-4 * h[i]
        P1, P2 = rk4_path(b, c .+ d), rk4_path(b, c .- d)
        st = maximum(norm((P1[k] .- P2[k])[B]) for k in eachindex(P1)) / 2e-4   # stretch per unit half-width
        if st > best
            best, bi = st, i
        end
    end
    bi
end

const TIME_CAP = 1200.0   # seconds per verify call (timeout = unsafe-possible)

function verify(b::Bench, a, mode, budget)
    lo, hi = root(b, a)
    U = unsafe_set(b, length(lo))
    todo = [(lo, hi)]
    certified = 0
    t0 = time()
    while !isempty(todo)
        certified + length(todo) > budget && return false, certified + length(todo)
        time() - t0 > TIME_CAP && return false, certified + length(todo)
        st = [status(b, l, h, U) for (l, h) in todo]
        bad = [(todo[i]..., st[i]) for i in eachindex(todo) if st[i] != :safe]
        certified += length(todo) - length(bad)
        isempty(bad) && break
        mode == "none" && return false, certified + length(bad)
        new = Tuple{Vector{Float64},Vector{Float64}}[]
        for (l, h, s) in bad
            # TMJets errors get the same widest-relative split in every mode (fair fallback)
            if mode == "uniform" || s == :error   # bisect the widest relative dimension
                r = (h .- l) ./ max.(hi .- lo, 1e-12)
                d = argmax(r)
            else
                d = split_dim(b, l, h)
            end
            m = (l[d] + h[d]) / 2
            h1 = copy(h); h1[d] = m; l2 = copy(l); l2[d] = m
            push!(new, (l, h1)); push!(new, (l2, h))
        end
        todo = new
    end
    true, certified
end

function alpha_star(b::Bench, mode, budget; tol=1e-2)
    stats = Any[]
    safe(a) = (t = @elapsed (ok, c) = verify(b, a, mode, budget);
               push!(stats, Dict("alpha" => a, "safe" => ok, "cells" => c, "time" => t));
               println("  $(b.name) $mode alpha=$(round(a, digits=4)) safe=$ok cells=$c $(round(t, digits=1))s");
               flush(stdout); ok)
    lo, hi = 1e-4, ALPHA_MAX
    safe(lo) || return 0.0, stats
    # cheap probe at the top level: a single cell (no refinement)
    (verify(b, hi, "none", 1)[1]) && return 1.0, stats
    while hi - lo > tol
        m = (lo + hi) / 2
        safe(m) ? (lo = m) : (hi = m)
    end
    lo, stats
end

function main()
    name, mode = ARGS[1], ARGS[2]
    budget = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 512
    lock = joinpath(@__DIR__, "..", "experiments", "logs", "$(name)__jr_$(mode).lock")
    if isfile(lock)  # another queue already runs this job
        println("SKIP $name $mode (locked)"); return
    end
    touch(lock)
    b = make(name)
    hits(b, root(b, 1e-4)..., unsafe_set(b, b.n + length(b.th_e)))  # compile warm-up (not timed)
    t = @elapsed (a, stats) = alpha_star(b, mode, budget)
    res = Dict("alpha_star" => a, "engine" => "JuliaReach-TMJets/$mode", "budget" => budget,
               "stats" => stats, "benchmark" => name, "wall_time" => t, "n" => b.n, "p" => length(b.th_e))
    out = joinpath(@__DIR__, "..", "experiments", "results", "$(name)__jr_$(mode)__curvature__path.json")
    open(out, "w") do io; JSON.print(io, res, 2); end
    println("DONE $name jr_$mode alpha*=$(round(a, digits=4)) $(round(t, digits=1))s")
end

main()
