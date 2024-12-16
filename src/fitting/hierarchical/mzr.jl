# This file contains types and methods implementing mass-metallicity relations
# for use with the SFH fitting routines in mzr_fitting.jl

# AbstractMZR API

""" `AbstractMZR{T <: Real} <: AbstractMetallicityModel{T}`: abstract supertype for all metallicity models that are mass-metallicity relations. Concrete subtypes `T <: AbstractMZR` should implement the following API: 
 - `(model::T)(Mstar::Real)` should be defined so that the struct is callable with a stellar mass `Mstar` in solar masses, returning the mean metallicity given the MZR model. This is ``\\mu_j \\left( \\text{M}_* \\right)`` in the derivations presented in the documentation.
 - `nparams(model::T)` should return the number of fittable parameters in the model.
 - `fittable_params(model::T)` should return the values of the fittable parameters in the model.
 - `gradient(model::T, Mstar::Real)` should return a tuple that contains the partial derivative of the mean metallicity ``\\mu_j`` with respect to each fittable model parameter, plus the partial derivative with respect to the stellar mass `Mstar` as the final element.
 - `update_params(model::T, newparams)` should return a new instance of `T` with the fittable parameters contained in `newparams` (which is typically a vector or tuple) and non-fittable parameters inherited from the provided `model`.
 - `transforms(model::T)` should return a tuple of length `nparams(model)` which indicates how the fittable variables should be transformed for optimization, if at all. Elements should be `1` for parameters that are constrained to always be positive, `0` for parameters that can be positive or negative, and `-1` for parameters that are constrained to always be negative.
 - `free_params(model::T)` should return an `NTuple{nparams(model), Bool}` that is `true` for fittable parameters that you want to optimize and `false` for fittable parameters that you want to stay fixed during optimization. """
abstract type AbstractMZR{T <: Real} <: AbstractMetallicityModel{T} end
Base.Broadcast.broadcastable(m::AbstractMZR) = Ref(m)

"""
    nparams(model::AbstractMZR)::Int
Returns the number of fittable parameters in the model. 
"""
nparams(model::AbstractMZR)
"""
    fittable_params(model::AbstractMZR{T})::NTuple{nparams(model), T}
Returns the values of the fittable parameters in the provided MZR `model`.
"""
fittable_params(model::AbstractMZR)
"""
    gradient(model::AbstractMZR{T}, Mstar::Real)::NTuple{nparams(model)+1, T}
 Returns a tuple containing the partial derivative of the mean metallicity with respect to all fittable parameters, plus the partial derivative with respect to the stellar mass `Mstar` as the final element. These partial derivatives are evaluated at stellar mass `Mstar`.
"""
gradient(model::AbstractMZR, Mstar::Real)
"""
    update_params(model::T, newparams)::T where {T <: AbstractMZR}
Returns a new instance of the model type `T` with the fittable parameters contained in `newparams` (which is typically a vector or tuple), with non-fittable parameters inherited from the provided `model`. 
"""
update_params(model::AbstractMZR, newparams::Any)
"""
    transforms(model::AbstractMZR)::NTuple{nparams(model), Int}
Returns a tuple of length `nparams(model)` which indicates how the fittable variables should be transformed for optimization, if at all. Elements should be `1` for parameters that are constrained to always be positive, `0` for parameters that can be positive or negative, and `-1` for parameters that are constrained to always be negative.
"""
transforms(model::AbstractMZR)
"""
    free_params(model::AbstractMZR)::NTuple{nparams(model), Bool}
Returns an tuple of length `nparams(model)` that is `true` for fittable parameters that you want to optimize and `false` for fittable parameters that you want to stay fixed during optimization.
 """
free_params(model::AbstractMZR)

##############################################
# Calculating per-SSP weights (r_{j,k}) from AbstractMZR

"""
    calculate_coeffs(mzr_model::AbstractMZR{T}, disp_model::AbstractDispersionModel{U},
                     mstars::AbstractVector{<:Number}, 
                     logAge::AbstractVector{<:Number},
                     metallicities::AbstractVector{<:Number}) where {T, U}

Returns per-SSP stellar mass coefficients (``r_{j,k}`` in the [derivation](@ref mzr_derivation)) using the provided MZR model `mzr_model` and metallicity dispersion model `disp_model` for the set of SSPs with logarithmic ages `logAge` and metallicities `metallicities`.

# Examples
```jldoctest; setup = :(import StarFormationHistories: calculate_coeffs, PowerLawMZR, GaussianDispersion)
julia> n_logage, n_mh = 10, 20; # Number of unique logAges, MHs

julia> coeffs = calculate_coeffs(PowerLawMZR(1.0, -1.0),
                                 GaussianDispersion(0.2),
                                 rand(n_logage),
                                 repeat(range(7.0, 10.0; length=n_logage); inner=n_mh),
                                 repeat(range(-2.0, 0.0; length=n_mh); outer=n_logage));

julia> coeffs isa Vector{Float64}
true

julia> length(coeffs) == n_logage * n_mh
true
```
"""
function calculate_coeffs(mzr_model::AbstractMZR{T}, disp_model::AbstractDispersionModel{U},
                          mstars::AbstractVector{<:Number}, 
                          logAge::AbstractVector{<:Number},
                          metallicities::AbstractVector{<:Number}) where {T, U}
    unique_logAge = unique(logAge)
    @assert(length(mstars) == length(unique_logAge),
            "Length of `mstars` must be the same as `unique_logAge`.")
    @assert length(logAge) == length(metallicities)
    S = promote_type(eltype(mstars), eltype(logAge), eltype(metallicities), T, U)

    # To calculate cumulative stellar mass, we need unique_logAge sorted in reverse order
    s_idxs = sortperm(unique_logAge; rev=true)

    # Set up to calculate coefficients
    coeffs = Vector{S}(undef, length(logAge))
    # norm_vals = Vector{S}(undef, length(unique_logAge))
    # Calculate cumulative stellar mass vector, properly sorted, then put back into original order
    cum_mstar = cumsum(mstars[s_idxs])[invperm(s_idxs)]# [reverse(s_idxs)]
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        # Find the mean metallicity of this age bin based on the cumulative stellar mass
        μ = mzr_model(cum_mstar[i])
        idxs = findall(==(la), logAge) # Find all entries that match this logAge
        tmp_coeffs = [disp_model(metallicities[ii], μ) for ii in idxs] # Calculate relative weights
        A = sum(tmp_coeffs)
        # norm_vals[i] = A
        # Make sure sum over tmp_coeffs equals 1 and write to coeffs
        coeffs[idxs] .= tmp_coeffs .* mstars[i] ./ A
    end
    return coeffs
end

###############################################
# Compute objective and gradient for MZR models

function fg!(F, G, Zmodel0::AbstractMZR{T}, dispmodel0::AbstractDispersionModel{U},
             variables::AbstractVector{<:Number},
             models::Union{AbstractMatrix{<:Number},
                           AbstractVector{<:AbstractMatrix{<:Number}}},
             data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
             composite::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
             logAge::AbstractVector{<:Number},
             metallicities::AbstractVector{<:Number}) where {T, U}

    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)),
                     eltype(composite), eltype(logAge), eltype(metallicities),
                     T, U)
    # Number of fittable parameters in MZR model;
    # in G, these come after the stellar mass coefficients R_j
    mzrpar = nparams(Zmodel0)
    # Number of fittable parameters in metallicity dispersion model;
    # in G, these come after the MZR parameters
    disppar = nparams(dispmodel0)

    # Construct new instance of Zmodel0 with updated parameters
    mzr_model = update_params(Zmodel0, @view(variables[end-(mzrpar+disppar)+1:(end-disppar)]))
    # Get indices of free parameters in MZR model
    mzr_free = BitVector(free_params(mzr_model))
    # Construct new instance of dispmodel0 with updated parameters
    disp_model = update_params(dispmodel0, @view(variables[end-(disppar)+1:end]))
    # Get indices of free parameters in metallicity dispersion model
    disp_free = BitVector(free_params(disp_model))
    # Calculate all coefficients r_{j,k} for each template
    coeffs = calculate_coeffs(mzr_model, disp_model,
                              @view(variables[begin:end-(mzrpar+disppar)]),
                              logAge, metallicities)

    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    # Need to do compute logL before ∇loglikelihood! because it will overwrite composite
    logL = loglikelihood(composite, data)

    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        Base.require_one_based_indexing(G)
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients
        fullG = Vector{eltype(G)}(undef, length(coeffs))
        ∇loglikelihood!(fullG, composite, models, data)

        unique_logAge = unique(logAge)
        # Cumulative stellar mass, from earliest time to most recent
        # cum_mstar = cumsum(@view(variables[begin:end-(mzrpar+disppar)]))
        s_idxs = sortperm(unique_logAge; rev=true)
        cum_mstar = cumsum(variables[s_idxs])[invperm(s_idxs)]
    
        # Find indices into unique_logAge for each entry in logAge
        jidx = [findfirst(==(logAge[i]), unique_logAge) for i in eachindex(logAge)]
         # Find indicies into logAge for each entry in unique_logAge
        jidx_inv = [findall(==(unique_logAge[i]), logAge) for i in eachindex(unique_logAge)]
        
        # Calculate quantities from MZR model
        μvec = mzr_model.(cum_mstar) # Find the mean metallicity of each time bin
        # Calculate full gradient of MZR model with respect to parameters and
        # cumulative stellar masses
        gradμ = tups_to_mat(values.(gradient.(mzr_model, cum_mstar)))
        # \frac{\partial μ_j}{\partial M_*}; This should always be the last row in gradμ
        dμ_dRj_vec = gradμ[end,:]
        
        # Calculate quantities from metallicity dispersion model
        # Relative weights A_{j,k} for *all* templates
        tmp_coeffs_vec = disp_model.(metallicities, μvec[jidx])
        # Full gradient of the dispersion model with respect to parameters
        # and mean metallicity μ_j
        grad_disp = tups_to_mat(values.(gradient.(disp_model, metallicities, μvec[jidx])))
        # # \frac{\partial A_{j,k}}{\partial \mu_j}
        dAjk_dμj = grad_disp[end,:]
        # Calculate sum_k A_{j,k} for all j
        A_vec = [sum(tmp_coeffs_vec[ii] for ii in idxs) for idxs in jidx_inv]

        # \frac{\partial A_{j,k}}{\partial R_j}; one entry for every template
        dAjk_dRj = dAjk_dμj .* dμ_dRj_vec[jidx]
        # sum_k dAjk_dRj; one entry per entry in unique(logAge)
        ksum_dAjk_dRj = [sum(dAjk_dRj[ii] for ii in idxs) for idxs in jidx_inv]
        # \frac{\partial F}{\partial r_{j,k}} * \frac{\partial r_{j,k}}{\partial R_j}
        drjk_dRj = fullG .* (variables[jidx] ./ A_vec[jidx] .*
            (dAjk_dRj .- (tmp_coeffs_vec .* ksum_dAjk_dRj[jidx] ./ A_vec[jidx]) ) )
        # \sum_k of above
        ksum_drjk_dRj = [sum(drjk_dRj[ii] for ii in idxs) for idxs in jidx_inv]
        # Calculate cumulative sum for sum_{j=0}^{j=j^\prime}, remembering to permute
        # by s_idxs to get the indices in order from earliest time to latest time
        cum_drjk_dRj = reverse!(cumsum(reverse!(ksum_drjk_dRj[s_idxs])))

        # Zero-out gradient vector in preparation to accumulate sums
        G .= zero(eltype(G))

        # Add in the j^\prime \neq j terms, remembering to permute G
        # by s_idxs to put the cum_drjk_dRj back in their original order
        for i in eachindex(cum_drjk_dRj)[begin+1:end]
            G[s_idxs[i-1]] -= cum_drjk_dRj[i]
        end

        # Loop over j
        for i in eachindex(unique_logAge)
            A = A_vec[i]
            idxs = jidx_inv[i]
            # Add the j^\prime == j term
            G[i] -= sum(fullG[idx] * (coeffs[idx] / variables[i] +
                (dAjk_dRj[idx] - (ksum_dAjk_dRj[i] * tmp_coeffs_vec[idx] / A)) *
                variables[i] / A) for idx in idxs)
            
            # Add MZR terms
            ksum_dAjk_dμj = sum(dAjk_dμj[j] for j in idxs)
            psum = -sum( fullG[j] * variables[i] / A *
                (dAjk_dμj[j] - tmp_coeffs_vec[j] / A * ksum_dAjk_dμj) for j in idxs)
            for par in (1:mzrpar)[mzr_free]
                G[end-(mzrpar+disppar-par)] += psum * gradμ[par,i]
            end

            # Add metallicity dispersion terms
            for par in (1:disppar)[disp_free]
                # View into grad_disp giving the partial derivative of A_jk
                # with respect to parameter P
                dAjk_dP = view(grad_disp, par, :)
                ksum_dAjk_dP = sum(dAjk_dP[j] for j in idxs)
                G[end-(disppar-par)] -= sum( fullG[j] * variables[i] / A *
                    (dAjk_dP[j] - tmp_coeffs_vec[j] / A * ksum_dAjk_dP) for j in idxs)
            end
        end
        return -logL
    elseif F != nothing # Optim.optimize wants only objective
        return -logL
    end
end

###################
# Concrete subtypes

"""
    PowerLawMZR(α::Real, MH0::Real, logMstar0::Real=6,
                free::NTuple{2, Bool}=(true, true)) <: AbstractMZR
Mass-metallicity model described by a single power law index `α > 0`, a metallicity normalization `MH0`, and the logarithm of the stellar mass `logMstar0 = log10(Mstar0 [M⊙])` at which the mean metallicity is `MH0`. Because `logMstar0` and `MH0` are degenerate, we treat `MH0` as a fittable parameter and `logMstar0` as a fixed parameter that will not be changed during optimizations. Such a power law MZR is often used when extrapolating literature results to low masses, e.g., ``\\text{M}_* < 10^8 \\; \\text{M}_\\odot.`` `α` will be fit freely during optimizations if `free[1] == true` and `MH0` will be fit freely if `free[2] == true`. The MZR is defined by

```math
\\begin{aligned}
[\\text{M} / \\text{H}] \\left( \\text{M}_* \\right) &= [\\text{M} / \\text{H}]_0 + \\text{log} \\left( \\left( \\frac{\\text{M}_*}{\\text{M}_{*,0}} \\right)^\\alpha \\right) \\\\
&= [\\text{M} / \\text{H}]_0 + \\alpha \\, \\left( \\text{log} \\left( \\text{M}_* \\right) - \\text{log} \\left( \\text{M}_{*,0} \\right) \\right) \\\\
\\end{aligned}
```

# Examples
```jldoctest; setup=:(using StarFormationHistories: nparams, gradient, update_params, transforms, free_params)
julia> PowerLawMZR(1.0, -1) isa PowerLawMZR{Float64}
true

julia> import Test

julia> Test.@test_throws(ArgumentError, PowerLawMZR(-1.0, -1)) isa Test.Pass
true

julia> nparams(PowerLawMZR(1.0, -1)) == 2
true

julia> PowerLawMZR(1.0, -1, 6)(1e7) ≈ 0
true

julia> all(values(gradient(PowerLawMZR(1.0, -1, 6), 1e8)) .≈
                (2.0, 1.0, 1 / 1e8 / log(10)))
true

julia> update_params(PowerLawMZR(1.0, -1, 7, (true, false)), (2.0, -2)) ==
         PowerLawMZR(2.0, -2, 7, (true, false))
true

julia> transforms(PowerLawMZR(1.0, -1)) == (1,0)
true

julia> free_params(PowerLawMZR(1.0, -1, 7, (true, false))) == (true, false)
true
```
"""
struct PowerLawMZR{T <: Real} <: AbstractMZR{T}
    α::T   # Power-law slope
    MH0::T # Normalization / intercept
    logMstar0::T # log10(Mstar) at which [M/H] = MH0
    free::NTuple{2, Bool}
    PowerLawMZR(α::T, MH0::T, logMstar0::T, free::NTuple{2, Bool}) where T <: Real =
        α ≤ zero(T) ? throw(ArgumentError("α must be > 0")) : new{T}(α, MH0, logMstar0, free)
end
PowerLawMZR(α::Real, MH0::Real, logMstar0::Real=6, free::NTuple{2, Bool}=(true, true)) =
    PowerLawMZR(promote(α, MH0, logMstar0)..., free)
nparams(d::PowerLawMZR) = 2
fittable_params(d::PowerLawMZR) = (α = d.α, β = d.MH0)
(mzr::PowerLawMZR)(Mstar::Real) = mzr.MH0 + mzr.α * (log10(Mstar) - mzr.logMstar0)
gradient(model::PowerLawMZR{T}, Mstar::S) where {T, S <: Real} =
    (α = log10(Mstar) - model.logMstar0,
     β = one(promote_type(T, S)),
     # \frac{\partial \mu_j}{\partial M_*} = \frac{\partial \mu_j}{\partial R_j}
     Mstar = model.α / Mstar / logten)
update_params(model::PowerLawMZR, newparams) =
    PowerLawMZR(newparams..., model.logMstar0, model.free)
transforms(::PowerLawMZR) = (1, 0)
free_params(model::PowerLawMZR) = model.free
