# DynamicHMC.jl looks pretty good but it's API seems quite complicated; you need to follow the LogDensityProblems.jl API, but you can use TransformVariables.jl and TransformedLogDensities.jl to do variable transformations for you and .... just seems very complicated.
import LogDensityProblems
import StaticArrays: SVector
import DynamicHMC
import Random
import MCMCDiagnosticTools: ess_rhat
import Statistics
import Distributions
import SFH: bin_cmd, bin_cmd_smooth, generate_stars_mag, generate_stars_mass, generate_stars_mass_composite, generate_stars_mag_composite, ingest_mags, model_cmd, sort_ingested, mass_limits, Binaries, NoBinaries, partial_cmd_smooth, fit_templates, construct_x0, composite!, loglikelihood, ∇loglikelihood


struct SFHModel{T,S,V}
    models::T
    composite::S
    data::V
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:SFHModel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::SFHModel) = length(problem.models)

function LogDensityProblems.logdensity_and_gradient(problem::SFHModel, logx)
    composite = problem.composite
    models = problem.models
    data = problem.data
    dims = length(models)
    # Transform the provided x
    x = SVector{dims}(exp(i) for i in logx)
    # Update the composite model matrix
    composite!( composite, x, models )
    logL = loglikelihood(composite, data) + sum(logx) # sum(logx) is the log-Jacobian correction
    # ∇logL = SVector{dims}( ∇loglikelihood(models[i], composite, data) * x[i] for i in eachindex(models,x) ) # The `* x[i]` is the Jacobian correction
    ∇logL = [ ∇loglikelihood(models[i], composite, data) * x[i] for i in eachindex(models,x) ] # The `* x[i]` is the Jacobian correction
    return logL, ∇logL
end

# Test code

nmodels = 10
histsize = (99,99)

models = [rand(histsize...) for i in 1:nmodels]
composite = rand(histsize...)
coeffs = rand(10) .* 5
# data = sum(coeffs .* models) # This is easy
data = rand.( Distributions.Poisson.(sum(coeffs .* models)) ) # This is harder

inst = SFHModel( models, composite, data )

result = [DynamicHMC.mcmc_with_warmup(Random.default_rng(), inst, 1000) for i in 1:2]
# We can also change where the mcmc starts;
# result = DynamicHMC.mcmc_with_warmup(Random.default_rng(), inst, 1000; reporter = DynamicHMC.ProgressMeterReport, initialization = ( q = log.(coeffs),) )
# reporter=DynamicHMC.ProgressMeterReport()
# reporter=DynamicHMC.LogProgressReport(step_interval=1000, time_interval_s=1)
# reporter=DynamicHMC.NoProgressReport()
# @time DynamicHMC.mcmc_with_warmup(Random.default_rng(), inst, 100; reporter=DynamicHMC.NoProgressReport()) # 3.8s
# @time DynamicHMC.mcmc_with_warmup(Random.default_rng(), inst, 100; reporter=DynamicHMC.NoProgressReport(), initialization=(q=log.(coeffs),)) # not much change really


DynamicHMC.Diagnostics.summarize_tree_statistics(result[1].tree_statistics)
# ess_rhat(DynamicHMC.stack_posterior_matrices(result))
# ess_rhat(Statistics.mean, result.posterior_matrix)
result2 = DynamicHMC.stack_posterior_matrices(result) # dim=3
println("ess_rhat: ",ess_rhat(Statistics.mean, result2))
result3 = exp.( DynamicHMC.pool_posterior_matrices(result) ) # dim=2
println("MCMC Means: ",Statistics.mean.(eachrow(result3)))
println("MCMC STDs: ", Statistics.std.(eachrow(result3)))
println("Correct Coeffs: ", coeffs)
println( abs.(Statistics.mean.(eachrow(result3)) .- coeffs) ./ Statistics.std.(eachrow(result3)) )
