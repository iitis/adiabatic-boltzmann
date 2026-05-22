using VeloxQFPGA, VeloxQIO, VeloxQtoolbox
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

function build_static_kwargs()
    kwargs = Dict{Symbol,Any}(
        :transport_type => Symbol(get(ENV, "FPGA_TRANSPORT", "auto")),
        :bulk_load => _env_bool("FPGA_BULK_LOAD", true),
        :core_clock_hz => _env_float("FPGA_CORE_CLOCK_HZ", 100_000_000.0),
        :verbose => _env_bool("FPGA_VERBOSE", false),
    )

    syscon_path = get(ENV, "FPGA_SYSCON_PATH", "")
    isempty(syscon_path) || (kwargs[:syscon_path] = syscon_path)

    bulk_dir = get(ENV, "FPGA_BULK_DIR", "")
    isempty(bulk_dir) || (kwargs[:bulk_dir] = bulk_dir)

    pcie_device = get(ENV, "FPGA_PCIE_DEVICE", "")
    isempty(pcie_device) || (kwargs[:pcie_device] = pcie_device)

    pcie_bar_size = _env_int("FPGA_PCIE_BAR_SIZE", 0)
    pcie_bar_size == 0 || (kwargs[:pcie_bar_size] = pcie_bar_size)

    timeout_s = get(ENV, "FPGA_TIMEOUT_S", "")
    isempty(timeout_s) || (kwargs[:timeout] = parse(Float64, timeout_s))

    bitstream = get(ENV, "FPGA_BITSTREAM", "")
    isempty(bitstream) || (kwargs[:bitstream] = bitstream)

    quartus_root = get(ENV, "FPGA_QUARTUS_ROOT", "")
    isempty(quartus_root) || (kwargs[:quartus_root] = quartus_root)

    return kwargs
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

function handle_sample(parts::Vector{<:AbstractString}, static_kwargs::Dict{Symbol,Any})
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
    kwargs = copy(static_kwargs)
    kwargs[:num_rep] = num_rep
    kwargs[:num_steps] = num_steps
    kwargs[:num_sweeps_per_step] = num_sweeps
    kwargs[:schedule_type] = schedule_type
    kwargs[:start_temp] = start_temp
    kwargs[:stop_temp] = stop_temp

    solver = FPGASA{Float32}(; kwargs...)
    sp = solver(model, ComputationModel())
    write_states(out_path, sp.states)

    if !isempty(meta_path)
        fpga_time = get(sp.metadata, :fpga_time, NaN)
        open(meta_path, "w") do io
            print(io, fpga_time)
        end
    end
    return nothing
end

function main()
    socket_path = get(ENV, "FPGA_SOCKET", "")
    isempty(socket_path) && error("FPGA_SOCKET env var required")
    println("[server] preloading packages and reading FPGA env")
    flush(stdout)
    static_kwargs = build_static_kwargs()

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
                            handle_sample(parts, static_kwargs)
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
