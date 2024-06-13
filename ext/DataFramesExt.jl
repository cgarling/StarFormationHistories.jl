module DataFramesExt

import StarFormationHistories: process_ASTs
using Printf: @sprintf
using DataFrames: DataFrame
using StatsBase: median, mean

function process_ASTs(ASTs::DataFrame, inmag::Symbol, outmag::Symbol,
                      bins::AbstractVector{<:Real}, selectfunc;
                      statistic=median)
    @assert length(bins) > 1
    !issorted(bins) && sort!(bins)

    completeness = Vector{Float64}(undef, length(bins)-1)
    bias = similar(completeness)
    error = similar(completeness)
    bin_centers = similar(completeness)

    input_mags = getproperty(ASTs, inmag)

    Threads.@threads for i in eachindex(completeness)
        # Get the stars in the current bin
        inbin = findall((input_mags .>= bins[i]) .&
            (input_mags .< bins[i+1]))
        tmp_asts = ASTs[inbin,:]
        if size(tmp_asts,1) == 0
            @warn(@sprintf("No input magnitudes found in bin ranging from %.6f => %.6f \
                               in `ASTs.inmag`, please revise `bins` argument.", bins[i],
                           bins[i+1]))
            completeness[i] = NaN
            bias[i] = NaN
            error[i] = NaN
            bin_centers[i] = bins[i] + (bins[i+1] - bins[i])/2
        end
        # Let selectfunc determine which ASTs are properly detected
        good = [selectfunc(row) for row in eachrow(tmp_asts)]
        completeness[i] = count(good) / size(tmp_asts,1)
        if count(good) > 0
            inmags = getproperty(tmp_asts, inmag)[good]
            outmags = getproperty(tmp_asts, outmag)[good]
            diff = outmags .- inmags # This makes bias relative to input
            bias[i] = statistic(diff)
            error[i] = statistic(abs.(diff))
            bin_centers[i] = mean(inmags)
        else
            if i != firstindex(completeness)
                bias[i] = bias[i-1]
                error[i] = error[i-1]
            else
                bias[i] = NaN
                error[i] = NaN
            end
            bin_centers[i] = bins[i] + (bins[i+1] - bins[i])/2
        end
    end
    return bin_centers, completeness, bias, error
end


end
