using VeloxQchaotic, VeloxQIO, VeloxQtoolbox
using KernelAbstractions
using Sockets

function _env_bool(key::String, default::Bool)
    val = get(ENV, key, "")
    isempty(val) && return default
    v = lowercase(strip(val))
    return v in ("1", "true", "yes", "on")
end

function _env_int(key::String, default::Int)
    val = get(ENV, key, "")
    isempty(val) && return default
    return parse(Int, val)
end

function _env_float(key::String, default::Float64)
    val = get(ENV, key, "")
    isempty(val) && return default
    return parse(Float64, val)
end

# CPU-only build: no GPU is available on this host.
function build_comp_model()
    cpu = VeloxQtoolbox.KA.CPU()
    return ComputationModel(
        scale_model = _env_bool("LANGEVIN_SCALE_MODEL", true),
        compress = _env_bool("LANGEVIN_COMPRESS", false),
        simulation_backend = cpu,
        energy_backend = cpu,
        th_per_block = _env_int("LANGEVIN_TH_PER_BLOCK", 64),
        energy_precision = Float32,
    )
end

function write_states(out_path::AbstractString, states)
    open(out_path, "w") do io
        n = size(states, 1)
        m = size(states, 2)
        for col in 1:m
            for i in 1:n
                if i > 1
                    write(io, ' ')
                end
                print(io, states[i, col])
            end
            write(io, '\n')
        end
    end
end

function build_solver(::Type{T}, num_rep::Int, num_steps::Int, dt::Real, sigma::Real,
                     detuning::Real, scale::Real, verbose::Bool) where {T<:Real}
    noise = VeloxQchaotic.Kernels.langevin_sb_noise(σ = T(sigma))
    solver = VeloxQchaotic.VeloxQ{T}(;
        num_steps = num_steps,
        num_rep = num_rep,
        kernel = VeloxQchaotic.Kernels.main_kernel_langevin_sb_no_kerr_no_heating,
        boundry = VeloxQchaotic.Kernels.inelastic_wall,
        noise = noise,
        dt = T(dt),
        verbose = verbose,
        scaling_correction = false,
        discrete_version = false,
        ternary_version = false,
        auto_tune_dt = false,
        relaxed_solution = false,
    )
    # params = [kerr_coeff, detuning, scale, heating_rate]; only `scale` (ξ) is used
    # by the langevin SB kernel, but all four must be present.
    set_params!(solver, T[zero(T), T(detuning), T(scale), zero(T)])
    return solver
end

function handle_sample(parts::Vector{<:AbstractString}, comp_model::ComputationModel)
    length(parts) >= 9 || error("sample needs 8 args, got $(length(parts) - 1)")
    model_path = String(parts[2])
    out_path = String(parts[3])
    num_rep = parse(Int, parts[4])
    num_steps = parse(Int, parts[5])
    dt = parse(Float32, parts[6])
    sigma = parse(Float32, parts[7])
    detuning = parse(Float32, parts[8])
    scale = parse(Float32, parts[9])
    meta_path = length(parts) >= 10 ? String(parts[10]) : ""

    model = load_model(Float32, model_path)
    solver = build_solver(Float32, num_rep, num_steps, dt, sigma,
                          detuning, scale, _env_bool("LANGEVIN_VERBOSE", false))

    sp = solver(model, comp_model)
    write_states(out_path, sp.states)

    if !isempty(meta_path)
        total_time = get(sp.metadata, "total_time", NaN)
        open(meta_path, "w") do io
            print(io, total_time)
        end
    end
    return nothing
end

function main()
    socket_path = get(ENV, "LANGEVIN_SOCKET", "")
    isempty(socket_path) && error("LANGEVIN_SOCKET env var required")
    println("[server] preloading packages and building computation model (CPU)")
    flush(stdout)
    comp_model = build_comp_model()

    ispath(socket_path) && rm(socket_path; force = true)
    server = listen(socket_path)
    println("[server] READY $socket_path")
    flush(stdout)

    try
        while true
            conn = accept(server)
            try
                while !eof(conn)
                    line = readline(conn; keep = false)
                    isempty(line) && continue
                    parts = split(line, '\t')
                    cmd = parts[1]
                    if cmd == "shutdown"
                        try
                            write(conn, "ok\n")
                            flush(conn)
                        catch
                        end
                        return
                    elseif cmd == "sample"
                        try
                            handle_sample(parts, comp_model)
                            write(conn, "ok\n")
                        catch err
                            msg = sprint(showerror, err)
                            msg = replace(msg, '\n' => " | ", '\t' => " ")
                            write(conn, "err\t", msg, "\n")
                        end
                        flush(conn)
                    else
                        write(conn, "err\tunknown command: $cmd\n")
                        flush(conn)
                    end
                end
            finally
                close(conn)
            end
        end
    finally
        close(server)
        try
            rm(socket_path; force = true)
        catch
        end
        println("[server] shutdown")
        flush(stdout)
    end
end

main()
