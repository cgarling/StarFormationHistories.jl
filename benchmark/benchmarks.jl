import StarFormationHistories as SFH
using BenchmarkTools

const SUITE = BenchmarkGroup()
SUITE["core"] = BenchmarkGroup()

const n_models = 2000
const hist_size = (150, 75)
# composite!
SUITE["core"]["composite!_Float32"] = @benchmarkable SFH.composite!(C, A, B) setup=(C = zeros(Float32, prod(hist_size)); A = rand(Float32, n_models); B = rand(Float32, prod(hist_size), n_models))
SUITE["core"]["composite!_Float64"] = @benchmarkable SFH.composite!(C, A, B) setup=(C = zeros(prod(hist_size)); A = rand(n_models); B = rand(prod(hist_size), n_models))

# ∇loglikelihood!
SUITE["core"]["∇loglikelihood_Float64"] = @benchmarkable SFH.∇loglikelihood!(G, composite, models, data) setup=(G=zeros(n_models); composite=rand(prod(hist_size)); models=rand(prod(hist_size), n_models); data = rand(prod(hist_size)))