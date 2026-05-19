"""
    fit_sfh_newton(MH_model0::AbstractMetallicityModel,
                   disp_model0::AbstractDispersionModel,
                   models::AbstractMatrix{<:Number},
                   data::AbstractVector{<:Number},
                   logAge::AbstractVector{<:Number},
                   metallicities::AbstractVector{<:Number};
                   x0::AbstractVector{<:Number} = <...>
                   kws...)
    fit_sfh_newton(MH_model0::AbstractMetallicityModel,
                   disp_model0::AbstractDispersionModel,
                   models::AbstractVector{<:AbstractMatrix{<:Number}},
                   data::AbstractMatrix{<:Number},
                   logAge::AbstractVector{<:Number},
                   metallicities::AbstractVector{<:Number};
                   x0::AbstractVector{<:Number} = <...>
                   kws...)

Returns a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) instance
containing MAP and MLE estimates obtained by fitting SSP templates `models` to `data` using
a Newton Trust-Region method with the Fisher information matrix as the Hessian
approximation. The interface mirrors [`fit_sfh`](@ref) exactly; see that function for a
full description of arguments, keyword arguments, and the return type.

The Hessian supplied to [`Optim.NewtonTrustRegion`](https://julianlsolvers.github.io/Optim.jl/stable/)
is the Fisher information matrix in the optimization variable space, computed by [`fgh!`](@ref).
Because the Fisher information is the expected Hessian of the negative log-likelihood
under the Dolphin (2002) Poisson model, this is equivalent to Fisher scoring and provides
better-scaled steps than quasi-Newton methods near the solution.

# Performance Notes
 - Each iteration evaluates the full Fisher information matrix, which is ``O(n_\\text{opt}^2 \\, n_\\text{pix})``. For typical problem sizes this is more expensive per iteration than BFGS but usually requires fewer iterations to converge.
 - Consider [`fit_sfh`](@ref) when the number of optimization variables is large (≫ 100) or when a warm-start from a previous BFGS result is available.
"""
function fit_sfh_newton(MH_model0::AbstractMetallicityModel{T}, disp_model0::AbstractDispersionModel{U},
                        models::AbstractMatrix{S},
                        data::AbstractVector{<:Number},
                        logAge::AbstractVector{<:Number},
                        metallicities::AbstractVector{<:Number};
                        x0::AbstractVector{<:Number} = construct_x0_mdf(logAge, convert(S, 13.7); normalize_value=1e6),
                        allow_f_increases::Bool = true,
                        store_trace::Bool = true,
                        extended_trace::Bool = true,
                        kws...) where {T, U, S <: Number}

    unique_logAge = unique(logAge)
    unique_MH = unique(metallicities)
    Nbins = length(x0)
    @argcheck length(x0) == length(unique_logAge) "length(x0) != length(unique(logAge))"
    @argcheck size(models, 1) == length(data)
    @argcheck size(models, 2) == length(logAge) == length(metallicities)

    # Renormalize x0 (same logic as fit_sfh)
    let _full_x0 = calculate_coeffs(MH_model0, disp_model0, x0, logAge, metallicities)
        if any(isnan, _full_x0)
            if MH_model0 isa AbstractMZR
                @warn "Initial `x0` provided to `fit_sfh_newton` produced NaN coefficients. Attempting to renormalize `x0` by replicating across metallicity bins."
            end
            _full_x0 = repeat(x0, length(unique_MH))
            x0 = renormalize_x0(data, models, x0, _full_x0)
            for i in 1:5
                _full_x0 = calculate_coeffs(MH_model0, disp_model0, x0, logAge, metallicities)
                if any(isnan, _full_x0)
                    @warn "Renormalization of x0 failed to produce valid coefficients after $i iterations. Returning last valid x0." i
                    break
                end
                x0 = renormalize_x0(data, models, x0, _full_x0)
            end
        else
            x0 = renormalize_x0(data, models, x0, _full_x0)
        end
    end

    # Log-transform x0
    x0 = map(log, x0)
    par = (values(fittable_params(MH_model0))..., values(fittable_params(disp_model0))...)
    tf = (transforms(MH_model0)..., transforms(disp_model0)...)
    free = SVector(free_params(MH_model0)..., free_params(disp_model0)...)
    x0_mzrdisp = logtransform(par, tf)
    x0 = vcat(x0, x0_mzrdisp[free])

    newton_options = Optim.Options(; allow_f_increases, store_trace, extended_trace, kws...)

    # Helper: reconstruct full (free+fixed, untransformed) variable vector from x (log-space)
    function _expand_x(x)
        Nfixed = count(~, free)
        _Nbins = lastindex(x) - count(free)
        full_x = similar(x, _Nbins + nparams(MH_model0) + nparams(disp_model0))
        for i in 1:_Nbins; full_x[i] = exp(x[i]); end
        x_extra = @view(x[_Nbins+1:end])
        x_zdisp = exptransform(x_extra, SVector(tf)[free])
        full_x[(_Nbins+1:lastindex(full_x))[free]]   .= x_zdisp
        init_par = SVector(fittable_params(MH_model0)..., fittable_params(disp_model0)...)
        full_x[(_Nbins+1:lastindex(full_x))[.~free]] .= init_par[.~free]
        return full_x
    end

    ptf      = findall(==(1), tf)
    ptf      = ptf[free[ptf]]
    ptf_idx  = ptf .+ Nbins
    ntf      = findall(==(-1), tf)
    ntf      = ntf[free[ntf]]

    function fgh_map!(F, G, H_out, X)
        SS = promote_type(eltype(models), eltype(data))
        composite = similar(data, SS)
        full_x    = _expand_x(X)
        G2 = isnothing(G) ? nothing : similar(full_x)
        nlogL = fgh!(F, G2, H_out, MH_model0, disp_model0, full_x, models, data,
                     composite, logAge, metallicities)
        # Transform-correct gradient and Jacobian corrections for MAP
        if !isnothing(H_out) || !isnothing(G2)
            ptf_pos = ptf_idx
            for i in vcat(eachindex(full_x)[begin:Nbins], ptf_pos)
                xi = full_x[i]
                if !isnothing(F); nlogL -= log(xi); end
                if !isnothing(G2); G2[i] = G2[i] * xi - 1; end
            end
            for i in ntf
                xi = full_x[i + Nbins]
                if !isnothing(F); nlogL += log(xi); end
                if !isnothing(G2); G2[i + Nbins] = -G2[i + Nbins] * xi + 1; end
            end
        end
        if !isnothing(G)
            @argcheck axes(G) == axes(X)
            for i in firstindex(G):Nbins; G[i] = G2[i]; end
            free_count = 1
            for i in 1:length(free)
                if free[i]
                    G[Nbins+free_count] = G2[Nbins+i]
                    free_count += 1
                end
            end
        end
        return isnothing(F) ? zero(SS) : nlogL
    end

    function fgh_mle!(F, G, H_out, X)
        SS = promote_type(eltype(models), eltype(data))
        composite = similar(data, SS)
        full_x    = _expand_x(X)
        G2 = isnothing(G) ? nothing : similar(full_x)
        nlogL = fgh!(F, G2, H_out, MH_model0, disp_model0, full_x, models, data,
                     composite, logAge, metallicities)
        # Transform-correct gradient (no Jacobian corrections for MLE)
        if !isnothing(G2)
            for i in vcat(eachindex(full_x)[begin:Nbins], ptf_idx)
                G2[i] = G2[i] * full_x[i]
            end
            for i in ntf
                G2[i + Nbins] = -G2[i + Nbins] * full_x[i + Nbins]
            end
        end
        if !isnothing(G)
            @argcheck axes(G) == axes(X)
            for i in firstindex(G):Nbins; G[i] = G2[i]; end
            free_count = 1
            for i in 1:length(free)
                if free[i]
                    G[Nbins+free_count] = G2[Nbins+i]
                    free_count += 1
                end
            end
        end
        return isnothing(F) ? zero(SS) : nlogL
    end

    result_map = Optim.optimize(NLSolversBase.only_fgh!(fgh_map!), x0,
                                Optim.NewtonTrustRegion(), newton_options)
    result_mle = Optim.optimize(NLSolversBase.only_fgh!(fgh_mle!),
                                Optim.minimizer(result_map),
                                Optim.NewtonTrustRegion(), newton_options)

    # Extract inverse Hessian from final Newton step trace (the Fisher information inverse)
    # NewtonTrustRegion does not accumulate an invH in the trace like BFGS; compute it from fgh!
    SS = promote_type(eltype(models), eltype(data))
    let μ_map_tmp = Optim.minimizer(result_map), μ_mle_tmp = Optim.minimizer(result_mle)
        composite_tmp = similar(data, SS)
        H_map = Matrix{SS}(undef, length(x0), length(x0))
        H_mle = Matrix{SS}(undef, length(x0), length(x0))
        fgh_map!(nothing, nothing, H_map, μ_map_tmp)
        fgh_mle!(nothing, nothing, H_mle, μ_mle_tmp)
        invH_map = PDMat(Hermitian(inv(Symmetric(H_map))))
        invH_mle = Hermitian(inv(Symmetric(H_mle)))
        try
            invH_mle = PDMat(invH_mle)
        catch
            @debug "Inverse Hessian matrix of MLE estimate is not positive definite" invH_mle
        end

        μ_map = similar(μ_map_tmp, Nbins + nparams(MH_model0) + nparams(disp_model0))
        μ_mle = similar(μ_map)
        σ_map = similar(invH_map, length(μ_map))
        σ_mle = similar(σ_map)
        σ_map_tmp = sqrt.(diag(invH_map))
        σ_mle_tmp = sqrt.(diag(invH_mle))
        for i in 1:Nbins
            μ_map[i] = exp(μ_map_tmp[i])
            μ_mle[i] = exp(μ_mle_tmp[i])
            σ_map[i] = μ_map[i] * σ_map_tmp[i]
            σ_mle[i] = μ_mle[i] * σ_mle_tmp[i]
        end
        free_count = 0
        for i in Nbins+1:length(μ_map)
            if free[i-Nbins]
                tfi = tf[i-Nbins]
                j = i - free_count
                if tfi == 1
                    μ_map[i] = exp(μ_map_tmp[j])
                    μ_mle[i] = exp(μ_mle_tmp[j])
                    σ_map[i] = μ_map[j] * σ_map_tmp[j]
                    σ_mle[i] = μ_map[j] * σ_mle_tmp[j]
                elseif tfi == 0
                    μ_map[i] = μ_map_tmp[j]
                    μ_mle[i] = μ_mle_tmp[j]
                    σ_map[i] = σ_map_tmp[j]
                    σ_mle[i] = σ_mle_tmp[j]
                elseif tfi == -1
                    μ_map[i] = -exp(μ_map_tmp[j])
                    μ_mle[i] = -exp(μ_mle_tmp[j])
                    σ_map[i] = -μ_map[j] * σ_map_tmp[j]
                    σ_mle[i] = -μ_map[j] * σ_mle_tmp[j]
                end
            else
                μ_map[i] = par[i-Nbins]
                μ_mle[i] = par[i-Nbins]
                σ_map[i] = 0
                σ_mle[i] = 0
                free_count += 1
            end
        end

        return CompositeBFGSResult(
            BFGSResult(μ_map, σ_map, invH_map, result_map,
                       update_params(MH_model0, @view(μ_map[Nbins+1:Nbins+nparams(MH_model0)])),
                       update_params(disp_model0, @view(μ_map[Nbins+nparams(MH_model0)+1:end]))),
            BFGSResult(μ_mle, σ_mle, invH_mle, result_mle,
                       update_params(MH_model0, @view(μ_mle[Nbins+1:Nbins+nparams(MH_model0)])),
                       update_params(disp_model0, @view(μ_mle[Nbins+nparams(MH_model0)+1:end]))))
    end
end

fit_sfh_newton(MH_model0::AbstractMetallicityModel, disp_model0::AbstractDispersionModel,
               models::AbstractVector{<:AbstractMatrix{<:Number}},
               data::AbstractMatrix{<:Number},
               logAge::AbstractVector{<:Number},
               metallicities::AbstractVector{<:Number}; kws...) =
    fit_sfh_newton(MH_model0, disp_model0, stack_models(models), vec(data),
                   logAge, metallicities; kws...)
