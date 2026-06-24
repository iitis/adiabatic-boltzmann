using VeloxQFPGA, VeloxQIO, VeloxQtoolbox
using Sockets

# Internal FPGASA helpers used to inline the body of (solver::FPGASA)(model, comp_model)
# minus its surrounding fpga_connect!/fpga_disconnect!. Keeping the connection open
# across VMC iterations saves ~0.05–0.2 s of PCIe bring-up per call.
using VeloxQFPGA: _model_to_csr, _load_graph!, _run_fpga_sa!, _read_spins_matrix,
                  _read_simd_width, _read_core_clock_hz, _fpga_time_seconds,
                  _compute_beta_range, fpga_connect!, fpga_disconnect!, is_connected

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

"""
    run_one_sample!(solver, comp_model, model, simd_width) -> (states::Matrix{Int8}, fpga_time::Float64)

Inline of the FPGASA call operator's body, minus connect/disconnect. The solver must
already be connected (the daemon does that once at startup). All schedule fields
(`num_rep`, `num_steps`, `num_sweeps_per_step`, `start_temp`, `stop_temp`,
`schedule_type`) must be set on `solver` before calling.
"""
function run_one_sample!(solver::FPGASA, comp_model::ComputationModel, model, simd_width::Int)
    N, h, row_ptr, col_val, weights = _model_to_csr(model)
    _load_graph!(solver, N, h, row_ptr, col_val, weights)

    beta_start, beta_stop = _compute_beta_range(solver, model)

    num_rep_v = solver.num_rep
    num_batches = cld(num_rep_v, simd_width)

    all_states = Matrix{Int8}(undef, N, num_rep_v)
    total_cycles = UInt64(0)
    clock_hz = _read_core_clock_hz(solver._transport, solver.core_clock_hz;
                                   verbose = solver.verbose)

    for batch = 1:num_batches
        rep_offset = (batch - 1) * simd_width
        batch_reps = min(simd_width, num_rep_v - rep_offset)
        cycles = _run_fpga_sa!(
            solver, N;
            beta_start = beta_start,
            beta_stop = beta_stop,
            steps = solver.num_steps,
            schedule_type = solver.schedule_type,
            sweeps = solver.num_sweeps_per_step,
        )
        total_cycles += cycles
        batch_states = _read_spins_matrix(solver._transport, N, batch_reps;
                                          verbose = solver.verbose)
        all_states[:, (rep_offset+1):(rep_offset+batch_reps)] .= batch_states
    end

    # Energy sort so the returned states matrix matches the FPGASA call operator's
    # sortperm-by-energy convention. The Python sampler then randomly subsamples.
    full_states = Float32.(all_states)
    en, σ = get_spectrum(
        full_states, similar(full_states), similar(full_states), model, 1.0f0;
        energy_precision = comp_model.energy_precision,
        energy_backend   = comp_model.energy_backend,
    )
    perm = sortperm(en)
    σ_sorted = Matrix{Int8}(σ[:, perm])

    fpga_time = _fpga_time_seconds(total_cycles, clock_hz)
    return σ_sorted, fpga_time
end

function handle_sample(parts::Vector{<:AbstractString}, solver::FPGASA,
                      comp_model::ComputationModel, simd_width::Int)
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

    # Mutate the persistent solver's schedule fields.
    solver.num_rep = num_rep
    solver.num_steps = num_steps
    solver.num_sweeps_per_step = num_sweeps
    solver.start_temp = start_temp
    solver.stop_temp = stop_temp
    solver.schedule_type = schedule_type

    model = load_model(model_path)
    states, fpga_time = run_one_sample!(solver, comp_model, model, simd_width)
    write_states(out_path, states)

    if !isempty(meta_path)
        open(meta_path, "w") do io
            print(io, fpga_time)
        end
    end
    return nothing
end

function try_reconnect!(solver::FPGASA)
    try
        fpga_disconnect!(solver)
    catch
    end
    fpga_connect!(solver)
    return _read_simd_width(solver._transport)
end

function main()
    socket_path = get(ENV, "FPGA_SOCKET", "")
    isempty(socket_path) && error("FPGA_SOCKET env var required")
    println("[server] preloading packages and reading FPGA env")
    flush(stdout)
    static_kwargs = build_static_kwargs()

    # Build solver once. Provide placeholder schedule fields — overwritten per call.
    placeholder_schedule = Dict{Symbol,Any}(
        :num_rep => 1024,
        :num_steps => 100,
        :num_sweeps_per_step => 1,
        :schedule_type => "geometric",
        :start_temp => 1.0,
        :stop_temp => 0.1,
    )
    solver_kwargs = merge(placeholder_schedule, static_kwargs)
    solver = FPGASA{Float32}(; solver_kwargs...)

    # Persistent connection: one connect at startup, one disconnect at shutdown.
    println("[server] connecting to FPGA")
    flush(stdout)
    fpga_connect!(solver)
    simd_width = _read_simd_width(solver._transport)
    println("[server] FPGA connected, simd_width=$simd_width")
    flush(stdout)

    comp_model = ComputationModel()

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
                            handle_sample(parts, solver, comp_model, simd_width)
                            write(conn, "ok\n")
                        catch err
                            msg = sprint(showerror, err)
                            msg = replace(msg, '\n' => " | ", '\t' => " ")
                            write(conn, "err\t", msg, "\n")
                            # PCIe MMIO may be in an undefined state after a failed sample.
                            # Try to recover the connection so subsequent samples don't all fail.
                            try
                                simd_width = try_reconnect!(solver)
                                @warn "[server] reconnected after sample error"
                            catch reconnect_err
                                @warn "[server] reconnect failed" reconnect_err
                            end
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
        try
            fpga_disconnect!(solver)
        catch
        end
        println("[server] shutdown")
        flush(stdout)
    end
end

main()
