using VeloxQFPGA, VeloxQIO, VeloxQtoolbox

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

function main()
    if length(ARGS) < 9
        println(
            stderr,
            "Usage: fpga_sa_bridge.jl <model_path> <out_path> <num_rep> <num_steps> <num_sweeps> <start_temp> <stop_temp> <schedule_type> <transport> [meta_path]",
        )
        exit(2)
    end

    model_path = ARGS[1]
    out_path = ARGS[2]
    num_rep = parse(Int, ARGS[3])
    num_steps = parse(Int, ARGS[4])
    num_sweeps = parse(Int, ARGS[5])
    start_temp = parse(Float64, ARGS[6])
    stop_temp = parse(Float64, ARGS[7])
    schedule_type = ARGS[8]
    transport = Symbol(ARGS[9])
    meta_path = length(ARGS) >= 10 ? ARGS[10] : ""

    println("[bridge] model=$model_path out=$out_path reps=$num_rep steps=$num_steps sweeps=$num_sweeps start=$start_temp stop=$stop_temp sched=$schedule_type transport=$transport meta=$meta_path")
    flush(stdout)

    kwargs = Dict{Symbol,Any}(
        :num_rep => num_rep,
        :num_steps => num_steps,
        :num_sweeps_per_step => num_sweeps,
        :schedule_type => schedule_type,
        :start_temp => start_temp,
        :stop_temp => stop_temp,
        :transport_type => transport,
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

    println("[bridge] loading model")
    flush(stdout)
    model = load_model(model_path)
    println("[bridge] building solver")
    flush(stdout)
    solver = FPGASA{Float32}(; kwargs...)
    println("[bridge] running solver")
    flush(stdout)
    sp = solver(model, ComputationModel())
    println("[bridge] solver done; writing output")
    flush(stdout)

    open(out_path, "w") do io
        n = size(sp.states, 1)
        m = size(sp.states, 2)
        for col in 1:m
            for i in 1:n
                if i > 1
                    write(io, ' ')
                end
                print(io, sp.states[i, col])
            end
            write(io, '\n')
        end
    end
    println("[bridge] output written")
    flush(stdout)

    if !isempty(meta_path)
        fpga_time = get(sp.metadata, :fpga_time, NaN)
        open(meta_path, "w") do io
            print(io, fpga_time)
        end
    end
end

main()
