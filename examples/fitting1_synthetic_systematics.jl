using Distributed

@everywhere using StarFormationHistories: AbstractBinaryModel, NoBinaries, RandomBinaryPairs, model_cmd, generate_stars_mass_composite, bin_cmd, AbstractMetallicityModel, AbstractDispersionModel, fit_sfh, construct_x0_mdf
@everywhere using Distributions: Sampleable, Univariate, Continuous
@everywhere using ProgressMeter: Progress, next!, finish!
@everywhere using Random: AbstractRNG, default_rng
@everywhere using StatsBase: Histogram
@everywhere using StaticArrays: SVector

# Support passing NoBinaries
dispatch_binarymodel(bmodel::Type{<:AbstractBinaryModel}, bfrac::Number) = bmodel(bfrac)
dispatch_binarymodel(bmodel::Type{NoBinaries}, bfrac::Number) = NoBinaries()

@everywhere function synthetic_systematics(mini_vec::AbstractVector{<:AbstractVector{<:Number}},
                                           mags::AbstractVector,
                                           mag_names::AbstractVector{String},
                                           y_mag_name::String,
                                           limit::Number,
                                           massfrac::AbstractVector{<:Number},
                                           imf::Sampleable{Univariate,Continuous},
                                           A_λ::SVector, # Recommend this being an SVector
                                           errfuncs, completefuncs, edges,
                                           MH_model::AbstractMetallicityModel, disp_model::AbstractDispersionModel,
                                           models::AbstractVector{<:AbstractMatrix{S}},
                                           logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number};
                                           dist_mod::Number=0,
                                           rng::AbstractRNG=default_rng(),
                                           mag_lim::Number=Inf,
                                           mag_lim_name::String="V",
                                           binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3),
                                           x0::AbstractVector{<:Number} = construct_x0_mdf(logAge, convert(S, 13.7); normalize_value=1e6),
                                           kws...) where {S <: Number}

    @assert length(mag_names) == 2 # Need 2 mags to form 1 color
    @assert y_mag_name ∈ mag_names
    # Create simulated star catalog
    starcat = generate_stars_mass_composite(mini_vec, mags, mag_names, limit, massfrac, imf; 
                                            dist_mod=dist_mod, rng=rng, mag_lim=mag_lim,
                                            mag_lim_name=mag_lim_name, binary_model=binary_model)
    # Concatenate per-population samples into single Vector{SVector}
    starcat = reduce(vcat, starcat[2])
    # Redden catalog
    for i in eachindex(starcat)
        starcat[i] = starcat[i] .+ A_λ
    end
    # Mock observe catalog
    sim_mags = model_cmd(starcat, errfuncs, completefuncs; rng=rng, ret_idxs=false)
    # Concatenate into 2D matrix
    sim_mags = reduce(hcat, sim_mags)
    # Form simulated Hess diagram
    sim_hess = bin_cmd(view(sim_mags,1,:) .- view(sim_mags,2,:),
                       view(sim_mags, findfirst(==(y_mag_name), mag_names), :), edges=edges)
    # Fit simulated Hess diagram with provided templates
    result = fit_sfh(MH_model, disp_model, models, sim_hess.weights, logAge, metallicities; x0=x0, kws...)
    return result
end

function distributed_systematics(Av_dist::Sampleable{Univariate,Continuous},
                                 A_λ_A_v::SVector, # Ratio of A_λ / A_v
                                 binary_dist::Sampleable{Univariate,Continuous},
                                 binary_model,
                                 dmod_dist::Sampleable{Univariate,Continuous},
                                 Nsamples::Integer,
                                 mini_vec::AbstractVector{<:AbstractVector{<:Number}},
                                 mags::AbstractVector,
                                 mag_names::AbstractVector{String},
                                 y_mag_name::String,
                                 limit::Number,
                                 massfrac::AbstractVector{<:Number},
                                 imf::Sampleable{Univariate,Continuous},
                                 errfuncs, completefuncs, edges,
                                 MH_model::AbstractMetallicityModel, disp_model::AbstractDispersionModel,
                                 models::AbstractVector{<:AbstractMatrix{S}},
                                 logAge::AbstractVector{<:Number},
                                 metallicities::AbstractVector{<:Number},
                                 T_max::Number;
                                 dist_mod::Number=0,
                                 # rng::AbstractRNG=default_rng(),
                                 mag_lim::Number=Inf,
                                 mag_lim_name::String="V",
                                 x0::AbstractVector{<:Number} = construct_x0_mdf(logAge, convert(S, 13.7); normalize_value=1e6),
                                 kws...) where {S <: Number}

    result = @distributed (vcat) for i in 1:Nsamples
        A_v = rand(Av_dist)
        A_λ = A_v .* A_λ_A_v
        bfrac = rand(binary_dist)
        dist_mod = rand(dmod_dist)
        # println(A_λ, " ", bfrac, " ", dist_mod)
        # println(A_v, " ", bfrac, " ", dist_mod)
        # bmodel = binary_model(bfrac)
        bmodel = dispatch_binarymodel(binary_model, bfrac)
        synthetic_systematics(mini_vec, mags, mag_names, y_mag_name, limit, massfrac, imf,
                              A_λ, errfuncs, completefuncs, edges, MH_model, disp_model,
                              models, logAge, metallicities;
                              dist_mod=dist_mod, mag_lim=mag_lim, mag_lim_name=mag_lim_name,
                              binary_model=bmodel, x0=x0, kws...)
        end
    return result
end

function threaded_systematics(Av_dist::Sampleable{Univariate,Continuous},
                              A_λ_A_v::SVector, # Ratio of A_λ / A_v
                              binary_dist::Sampleable{Univariate}, # ,Continuous},
                              binary_model,
                              dmod_dist::Sampleable{Univariate}, # ,Continuous},
                              Nsamples::Integer,
                              mini_vec::AbstractVector{<:AbstractVector{<:Number}},
                              mags::AbstractVector,
                              mag_names::AbstractVector{String},
                              y_mag_name::String,
                              limit::Number,
                              massfrac::AbstractVector{<:Number},
                              imf::Sampleable{Univariate,Continuous},
                              errfuncs, completefuncs, edges,
                              MH_model::AbstractMetallicityModel, disp_model::AbstractDispersionModel,
                              models::AbstractVector{<:AbstractMatrix{S}},
                              logAge::AbstractVector{<:Number},
                              metallicities::AbstractVector{<:Number},
                              T_max::Number, normalize_value::Number;
                              dist_mod::Number=0,
                              # rng::AbstractRNG=default_rng(),
                              mag_lim::Number=Inf,
                              mag_lim_name::String="V",
                              x0::AbstractVector{<:Number} = construct_x0_mdf(logAge, convert(S, 13.7); normalize_value=1e6),
                              kws...) where {S <: Number}

    # my_lock = ReentrantLock()
    # result = []
    result = Array{Float64}(undef, Nsamples, 3, length(unique(logAge)))
    p = Progress(Nsamples)
    Threads.@threads for i in 1:Nsamples
        A_v = rand(Av_dist)
        A_λ = A_v .* A_λ_A_v
        bfrac = rand(binary_dist)
        dist_mod = rand(dmod_dist)
        # println(A_λ, " ", bfrac, " ", dist_mod)
        # println(A_v, " ", bfrac, " ", dist_mod)
        # bmodel = binary_model(bfrac)
        bmodel = dispatch_binarymodel(binary_model, bfrac)
        r = synthetic_systematics(mini_vec, mags, mag_names, y_mag_name, limit, massfrac, imf,
                                  A_λ, errfuncs, completefuncs, edges, MH_model, disp_model,
                                  models, logAge, metallicities;
                                  dist_mod=dist_mod, mag_lim=mag_lim, mag_lim_name=mag_lim_name,
                                  binary_model=bmodel, x0=x0, kws...)
        coeffs = calculate_coeffs(r.map, logAge, metallicities)
        coeffs *= normalize_value
        _, cum_sfr_arr, sfr_arr, mean_mh_arr =
            calculate_cum_sfr(coeffs, logAge, metallicities, T_max)
        result[i, 1, :] .= cum_sfr_arr
        result[i, 2, :] .= sfr_arr
        result[i, 3, :] .= mean_mh_arr
        next!(p)
        # @lock my_lock push!(result, r)
        # Just compute cumulative SFH and return
    end
    finish!(p)
    return result
end


# f(catalog, A_λ) = [i .+ A_λ for i in catalog]
# # This is better, no allocations
# f!(catalog, A_λ) = (for i in eachindex(catalog); catalog[i] = catalog[i] .+ A_λ; end; return catalog)
# A = [@SVector rand(2) for i in 1:10000]
# A_λ = SVector(0.05, 0.02)
# @benchmark f(A, $A_λ) setup=(A = [@SVector rand(2) for i in 1:10000])
# @benchmark f!(A, $A_λ) setup=(A = [@SVector rand(2) for i in 1:10000])
