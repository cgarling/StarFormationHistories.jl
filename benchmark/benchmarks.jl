import StarFormationHistories as SFH
using BenchmarkTools

const SUITE = BenchmarkGroup()

SUITE["core"] = BenchmarkGroup()

# composite!
SUITE["core"]["composite!_Float32"] = @benchmarkable SFH.composite!(C, A, B) setup=(n_models=2000; hist_size = 150 * 75; C = zeros(Float32, hist_size); A = rand(Float32, n_models); B = rand(Float32, hist_size, n_models))
SUITE["core"]["composite!_Float64"] = @benchmarkable SFH.composite!(C, A, B) setup=(n_models=2000; hist_size = 150 * 75; C = zeros(hist_size); A = rand(n_models); B = rand(hist_size, n_models))