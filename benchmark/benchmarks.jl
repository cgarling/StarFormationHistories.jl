import StarFormationHistories as SFH
using BenchmarkTools
using LinearAlgebra: BLAS
using PrettyTables: pretty_table
BLAS.set_num_threads(1)

function show_benchmarks(results)
    # Collect results
    sorted  = sort(collect(results), by=first)
    names   = [k for (k,_) in sorted]
    trials  = [v for (_,v) in sorted]

    # Pack into matrix
    data = hcat(
        names,
        [BenchmarkTools.prettytime(median(t).time) for t in trials],
        [BenchmarkTools.prettymemory(median(t).memory) for t in trials],
        [median(t).allocs for t in trials]
    )

    # Make pretty table
    pretty_table(data;
        column_labels = ["Benchmark", "Median Time", "Memory", "Allocs"],
        alignment     = [:l, :r, :r, :r]
    )
end

const SUITE = BenchmarkGroup()
SUITE["core"] = BenchmarkGroup()

const n_models = 2000
const hist_size = (150, 75)

# composite! call signatures
SUITE["core"]["composite!_vector_models_Float32"] = @benchmarkable SFH.composite!(C, A, B) setup=(C = zeros(Float32, hist_size); A = rand(Float32, n_models); B = [rand(Float32, hist_size) for _ in 1:n_models])
SUITE["core"]["composite!_vector_models_Float64"] = @benchmarkable SFH.composite!(C, A, B) setup=(C = zeros(Float64, hist_size); A = rand(Float64, n_models); B = [rand(Float64, hist_size) for _ in 1:n_models])
SUITE["core"]["composite!_flat_models_Float32"] = @benchmarkable SFH.composite!(C, A, B) setup=(C = zeros(Float32, prod(hist_size)); A = rand(Float32, n_models); B = rand(Float32, prod(hist_size), n_models))
SUITE["core"]["composite!_flat_models_Float64"] = @benchmarkable SFH.composite!(C, A, B) setup=(C = zeros(Float64, prod(hist_size)); A = rand(Float64, n_models); B = rand(Float64, prod(hist_size), n_models))

# loglikelihood call signatures
SUITE["core"]["loglikelihood_composite_data_Float32"] = @benchmarkable SFH.loglikelihood(C, D) setup=(C = rand(Float32, hist_size); D = rand(Float32, hist_size))
SUITE["core"]["loglikelihood_composite_data_Float64"] = @benchmarkable SFH.loglikelihood(C, D) setup=(C = rand(Float64, hist_size); D = rand(Float64, hist_size))
SUITE["core"]["loglikelihood_composite_data_Float64_Int64"] = @benchmarkable SFH.loglikelihood(C, D) setup=(C = rand(Float64, hist_size); D = rand(Int64, hist_size))
SUITE["core"]["loglikelihood_coeffs_vector_models_Float32"] = @benchmarkable SFH.loglikelihood(A, B, D) setup=(A = rand(Float32, n_models); B = [rand(Float32, hist_size) for _ in 1:n_models]; D = rand(Float32, hist_size))
SUITE["core"]["loglikelihood_coeffs_vector_models_Float64"] = @benchmarkable SFH.loglikelihood(A, B, D) setup=(A = rand(Float64, n_models); B = [rand(Float64, hist_size) for _ in 1:n_models]; D = rand(Float64, hist_size))
SUITE["core"]["loglikelihood_coeffs_flat_models_Float32"] = @benchmarkable SFH.loglikelihood(A, B, D) setup=(A = rand(Float32, n_models); B = rand(Float32, prod(hist_size), n_models); D = rand(Float32, prod(hist_size)))
SUITE["core"]["loglikelihood_coeffs_flat_models_Float64"] = @benchmarkable SFH.loglikelihood(A, B, D) setup=(A = rand(Float64, n_models); B = rand(Float64, prod(hist_size), n_models); D = rand(Float64, prod(hist_size)))

# ∇loglikelihood call signatures
SUITE["core"]["∇loglikelihood_single_model_Float32"] = @benchmarkable SFH.∇loglikelihood(M, C, D) setup=(M = rand(Float32, hist_size); C = rand(Float32, hist_size); D = rand(Float32, hist_size))
SUITE["core"]["∇loglikelihood_single_model_Float64"] = @benchmarkable SFH.∇loglikelihood(M, C, D) setup=(M = rand(Float64, hist_size); C = rand(Float64, hist_size); D = rand(Float64, hist_size))
SUITE["core"]["∇loglikelihood_vector_models_Float32"] = @benchmarkable SFH.∇loglikelihood(B, C, D) setup=(B = [rand(Float32, hist_size) for _ in 1:n_models]; C = rand(Float32, hist_size); D = rand(Float32, hist_size))
SUITE["core"]["∇loglikelihood_vector_models_Float64"] = @benchmarkable SFH.∇loglikelihood(B, C, D) setup=(B = [rand(Float64, hist_size) for _ in 1:n_models]; C = rand(Float64, hist_size); D = rand(Float64, hist_size))
SUITE["core"]["∇loglikelihood_flat_models_Float32"] = @benchmarkable SFH.∇loglikelihood(B, C, D) setup=(B = rand(Float32, prod(hist_size), n_models); C = rand(Float32, prod(hist_size)); D = rand(Float32, prod(hist_size)))
SUITE["core"]["∇loglikelihood_flat_models_Float64"] = @benchmarkable SFH.∇loglikelihood(B, C, D) setup=(B = rand(Float64, prod(hist_size), n_models); C = rand(Float64, prod(hist_size)); D = rand(Float64, prod(hist_size)))
SUITE["core"]["∇loglikelihood_coeffs_vector_models_Float32"] = @benchmarkable SFH.∇loglikelihood(A, B, D) setup=(A = rand(Float32, n_models); B = [rand(Float32, hist_size) for _ in 1:n_models]; D = rand(Float32, hist_size))
SUITE["core"]["∇loglikelihood_coeffs_vector_models_Float64"] = @benchmarkable SFH.∇loglikelihood(A, B, D) setup=(A = rand(Float64, n_models); B = [rand(Float64, hist_size) for _ in 1:n_models]; D = rand(Float64, hist_size))
SUITE["core"]["∇loglikelihood_coeffs_flat_models_Float32"] = @benchmarkable SFH.∇loglikelihood(A, B, D) setup=(A = rand(Float32, n_models); B = rand(Float32, prod(hist_size), n_models); D = rand(Float32, prod(hist_size)))
SUITE["core"]["∇loglikelihood_coeffs_flat_models_Float64"] = @benchmarkable SFH.∇loglikelihood(A, B, D) setup=(A = rand(Float64, n_models); B = rand(Float64, prod(hist_size), n_models); D = rand(Float64, prod(hist_size)))

# ∇loglikelihood! call signatures
SUITE["core"]["∇loglikelihood!_vector_models_Float32"] = @benchmarkable SFH.∇loglikelihood!(G, C, B, D) setup=(G = zeros(Float32, n_models); C = rand(Float32, hist_size); B = [rand(Float32, hist_size) for _ in 1:n_models]; D = rand(Float32, hist_size))
SUITE["core"]["∇loglikelihood!_vector_models_Float64"] = @benchmarkable SFH.∇loglikelihood!(G, C, B, D) setup=(G = zeros(Float64, n_models); C = rand(Float64, hist_size); B = [rand(Float64, hist_size) for _ in 1:n_models]; D = rand(Float64, hist_size))
SUITE["core"]["∇loglikelihood!_flat_models_Float32"] = @benchmarkable SFH.∇loglikelihood!(G, C, B, D) setup=(G = zeros(Float32, n_models); C = rand(Float32, prod(hist_size)); B = rand(Float32, prod(hist_size), n_models); D = rand(Float32, prod(hist_size)))
SUITE["core"]["∇loglikelihood!_flat_models_Float64"] = @benchmarkable SFH.∇loglikelihood!(G, C, B, D) setup=(G = zeros(Float64, n_models); C = rand(Float64, prod(hist_size)); B = rand(Float64, prod(hist_size), n_models); D = rand(Float64, prod(hist_size)))

# fisher_information call signatures
SUITE["core"]["fisher_information_flat_models_Float32"] = @benchmarkable SFH.fisher_information(B, C) setup=(B = rand(Float32, prod(hist_size), n_models); C = rand(Float32, prod(hist_size)))
SUITE["core"]["fisher_information_flat_models_Float64"] = @benchmarkable SFH.fisher_information(B, C) setup=(B = rand(Float64, prod(hist_size), n_models); C = rand(Float64, prod(hist_size)))
SUITE["core"]["fisher_information_flat_models!_Float32"] = @benchmarkable SFH.fisher_information!(I_mat, W, B, C) setup=(I_mat = zeros(Float32, n_models, n_models); W = zeros(Float32, prod(hist_size), n_models); B = rand(Float32, prod(hist_size), n_models); C = rand(Float32, prod(hist_size)))
SUITE["core"]["fisher_information_flat_models!_Float64"] = @benchmarkable SFH.fisher_information!(I_mat, W, B, C) setup=(I_mat = zeros(Float64, n_models, n_models); W = zeros(Float64, prod(hist_size), n_models); B = rand(Float64, prod(hist_size), n_models); C = rand(Float64, prod(hist_size)))




# If not on CI, we'll show a nice table
if get(ENV, "CI", "false") == "false"
    # Run the benchmarks
    results = run(SUITE, verbose=true)
    println("⎯⎯⎯ Core Suite ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
    show_benchmarks(results["core"])
end