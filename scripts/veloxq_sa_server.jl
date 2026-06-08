using VeloxQstandard, VeloxQIO, VeloxQtoolbox
using CUDA
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

function build_comp_model()
    backend_choice = lowercase(strip(get(ENV, "VELOXQ_BACKEND", "cuda")))
    sim_backend = if backend_choice == "cpu"
        VeloxQtoolbox.KA.CPU()
    elseif backend_choice in ("cuda", "gpu")
        CUDABackend()
    else
        error("Unknown VELOXQ_BACKEND=$(backend_choice). Expected one of: cuda, gpu, cpu.")
    end
    return ComputationModel(
        scale_model = _env_bool("VELOXQ_SCALE_MODEL", true),
        compress = _env_bool("VELOXQ_COMPRESS", true),
        simulation_backend = sim_backend,
        energy_backend = sim_backend,
        th_per_block = _env_int("VELOXQ_TH_PER_BLOCK", 128),
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

function handle_sample(parts::Vector{<:AbstractString}, comp_model::ComputationModel)
    length(parts) >= 10 || error("sample needs 9 args, got $(length(parts) - 1)")
    model_path = String(parts[2])
    out_path = String(parts[3])
    num_rep = parse(Int, parts[4])
    num_steps = parse(Int, parts[5])
    num_sweeps = parse(Int, parts[6])
    start_temp = parse(Float64, parts[7])
    stop_temp = parse(Float64, parts[8])
    schedule_type = String(parts[9])
    meta_path = String(parts[10])

    model = load_model(model_path)
    solver = SimulatedAnnealing{Float32}(;
        num_rep = num_rep,
        num_steps = num_steps,
        num_sweeps_per_step = num_sweeps,
        schedule_type = schedule_type,
        start_temp = start_temp,
        stop_temp = stop_temp,
        verbose = _env_bool("VELOXQ_VERBOSE", false),
    )
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
    socket_path = get(ENV, "VELOXQ_SOCKET", "")
    isempty(socket_path) && error("VELOXQ_SOCKET env var required")
    println("[server] preloading packages and building computation model")
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
