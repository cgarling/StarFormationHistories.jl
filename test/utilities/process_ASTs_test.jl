using StableRNGs: StableRNG
using StatsBase: median, mean
using Test
using TypedTables: Table
using DataFrames: DataFrame
using StarFormationHistories: process_ASTs

const seedval = 58392 # Seed to use when instantiating new StableRNG objects

@testset "process_ASTs" begin
    rng = StableRNG(seedval)
    nstars = 100_000
    bins = 20.0:0.1:21.0
    bin_centers = bins[begin:end-1] .+ step(bins)/2
    # Increasing error as a function of input mags
    error = [0.01 + 0.1 * (b-first(bins)) for b in bin_centers]
    # Constant bias
    bias = [0.01 for b in bin_centers]
    inmags = Vector{Float64}(undef, nstars)
    outmags = similar(inmags)
    # Flag that indicates whether an AST is good or not; all true for now
    flag = trues(length(inmags))
    # Since we are using low nstars to test fast, the sample bias and error
    # will not be exactly equal to the values in `bias` and `error` above.
    # We will record the actual sample bias and error in these vectors
    # and compare against these.
    a_bias = Vector{Float64}(undef, length(bin_centers)) # Store the actual sample bias
    a_error = similar(a_bias)                            # Store the actual sample error
    nstars_iter = nstars ÷ length(bin_centers)
    for i in eachindex(bin_centers)
        # Divide total nstars equally between bins
        idxs = ((i-1)*nstars_iter+1):(i*nstars_iter)
        # Create input magnitudes properly scaled to be inside bin
        curr_in = (rand(rng, nstars_iter) .* (bins[i+1] - bins[i])) .+ bins[i]
        inmags[idxs] .= curr_in
        curr_out = curr_in .+ (randn(rng, length(curr_in)) .* error[i]) .+ bias[i]
        outmags[idxs] .= curr_out
        curr_diff = curr_out .- curr_in
        a_bias[i] = median(curr_diff)
        a_error[i] = 1.482602218505602 * median(abs.(curr_diff .- a_bias[i]))
    end
    # Loop over all supported table types
    for ttype in (Table, DataFrame)
        # Basic call
        result = process_ASTs(ttype(in=inmags, out=outmags, flag=flag),
                              :in, :out, bins, x->x.flag==true)
        @test result isa NTuple{4, Vector{Float64}}
        @test all(result[2] .== 1)
        @test result[3] ≈ a_bias
        @test result[4] ≈ a_error
        # The error is the Gaussian σ used to generate the output magnitudes
        @test result[4] ≈ error rtol=0.05
        
        # Make sure selectfunc is properly utilized to filter input ASTs
        outmags2 = copy(outmags)
        outmags2[1:100] .= 99.999
        result3 = process_ASTs(ttype(in=inmags, out=outmags2, flag=flag),
                               :in, :out, bins,
                               x-> (x.flag==true) & (x.out != 99.999))
        @test result3 isa NTuple{4, Vector{Float64}}
        # Make sure completeness is properly affected by bad stars
        @test all(result3[2] .== vcat((nstars_iter-100)/nstars_iter,
                                      ones(length(bin_centers)-1)))
        # Bias and error wont be exactly the same but close
        @test result3[3] ≈ a_bias rtol=1e-2
        @test result3[4] ≈ a_error rtol=1e-2

        # Test case where first bin has input stars but
        # none pass `selectfunc` criteria
        result4 = process_ASTs(ttype(in=inmags, out=outmags, flag=flag),
                              :in, :out, bins,
                              x -> (x.flag == true) & (x.in > bins[2]))
        @test result4 isa NTuple{4, Vector{Float64}}
        @test result4[2][1] == 0.0
        @test result4[2][2:end] == ones(length(bin_centers)-1)
        @test isnan(result4[3][1])
        @test result4[3][2:end] ≈ a_bias[2:end]
        @test isnan(result4[4][1])
        @test result4[4][2:end] ≈ a_error[2:end]

        # Test case where second bin has input stars but
        # none pass `selectfunc` criteria
        # This case used to set error and bias
        # to the previous bin value.
        # The argument was that this would make the output
        # continuous, but it is not strictly correct. Now we return NaN
        # for bias and error whenever the number of detected stars in
        # a bin is 0.
        result5 = process_ASTs(ttype(in=inmags, out=outmags, flag=flag),
                              :in, :out, bins,
                              x -> (x.flag == true) & ((x.in < bins[2]) | (x.in >= bins[3])))
        @test result5 isa NTuple{4, Vector{Float64}}
        @test result5[2][1] == 1.0
        @test result5[2][2] == 0.0
        @test result5[2][3:end] == ones(length(bin_centers)-2)
        @test result5[3][1] ≈ a_bias[1]
        @test isnan(result5[3][2])
        @test result5[3][3:end] ≈ a_bias[3:end]
        @test result5[4][1] ≈ a_error[1]
        @test isnan(result5[4][2])
        @test result5[4][3:end] ≈ a_error[3:end]

        # Test case where first bin has no input stars
        result6 = process_ASTs(ttype(in=inmags, out=outmags, flag=flag),
                               :in, :out, vcat(bins, last(bins) + step(bins)),
                               x -> (x.flag == true))
        @test result6 isa NTuple{4, Vector{Float64}}
        @test isnan(result6[2][end])
        @test all(result6[2][begin:end-1] .== 1)
        @test isnan(result6[3][end])
        @test result6[3][begin:end-1] ≈ a_bias
        @test isnan(result6[4][end])
        @test result6[4][begin:end-1] ≈ a_error
    end
end

@testset "process_ASTs joint" begin
    rng = StableRNG(seedval)
    nstars = 200_000
    bins = (20.0:0.25:22.0, 20.0:0.25:22.0)
    in1, in2 = 20 .+ 2 .* rand(rng, nstars), 20 .+ 2 .* rand(rng, nstars)
    # Leave the red corner of the grid without artificial stars
    keep = in1 .- in2 .< 1.5
    in1, in2 = in1[keep], in2[keep]
    out1 = in1 .+ 0.02 .* randn(rng, length(in1)) .+ 0.01
    out2 = in2 .+ 0.05 .* randn(rng, length(in2)) .- 0.02
    # Diagonal selection that depends on both filters
    selectfunc(x) = (x.in1 + x.in2) / 2 < 21.25
    σ = (0.02, 0.05)
    for ttype in (Table, DataFrame)
        table = ttype(in1=in1, in2=in2, out1=out1, out2=out2)
        # Direct calculation of (completeness, bias, error) over the stars `idx`, with counts
        function direct(idx)
            det = [k for k in idx if (in1[k] + in2[k]) / 2 < 21.25]
            isempty(det) && return (isempty(idx) ? NaN : 0.0, (NaN, NaN), (NaN, NaN)), length(idx), 0
            be = [(median(d), 1.482602218505602 * median(abs.(d .- median(d)))) for d in (out1[det] .- in1[det], out2[det] .- in2[det])]
            return (length(det) / length(idx), first.(Tuple(be)), last.(Tuple(be))), length(idx), length(det)
        end
        inbin(m, b, i) = b[i] .<= m .< b[i+1]
        ingrid = inbin(in1, [bins[1][1], bins[1][end]], 1) .& inbin(in2, [bins[2][1], bins[2][end]], 1)
        for n0 in (0, 10)
            centers, completeness, bias, error = process_ASTs(table, (:in1, :in2), (:out1, :out2), bins, selectfunc; n0)
            @test centers == Tuple(b[begin:end-1] .+ step(b) / 2 for b in bins)
            @test size(completeness) == (length(bins[1]) - 1, length(bins[2]) - 1)
            @test all(x -> size(x) == size(completeness), (bias..., error...))
            # Every row and column contains detected stars, so the 1-D fallback leaves no NaN
            @test !any(isnan, completeness) && !any(x -> any(isnan, x), (bias..., error...))
            for i in axes(completeness, 1), j in axes(completeness, 2)
                (c, b, e), n, ndet = direct(findall(inbin(in1, bins[1], i) .& inbin(in2, bins[2], j)))
                row, _, _ = direct(findall(inbin(in1, bins[1], i) .& ingrid))
                col, _, _ = direct(findall(inbin(in2, bins[2], j) .& ingrid))
                blend(x, n, f) = (n == 0 || isnan(x)) ? f : (n * x + n0 * f) / (n + n0)
                @test completeness[i, j] ≈ blend(c, n, min(row[1], col[1]))
                @test bias[1][i, j] ≈ blend(b[1], ndet, row[2][1])
                @test bias[2][i, j] ≈ blend(b[2], ndet, col[2][2])
                @test error[1][i, j] ≈ blend(e[1], ndet, row[3][1])
                @test error[2][i, j] ≈ blend(e[2], ndet, col[3][2])
            end
            # Fully selected bins recover the input bias and Gaussian σ
            @test completeness[1, 1] == 1
            @test bias[1][1, 1] ≈ 0.01 atol=0.005
            @test bias[2][1, 1] ≈ -0.02 atol=0.01
            @test error[1][1, 1] ≈ σ[1] rtol=0.1
            @test error[2][1, 1] ≈ σ[2] rtol=0.1
            # The red corner has no artificial stars, so it takes the 1-D values
            @test completeness[end, end] < 0.01 # Entirely past the diagonal selection; exactly 0 for n0 = 0
            @test completeness[end, 1] == min(direct(findall(inbin(in1, bins[1], length(bins[1]) - 1) .& ingrid))[1][1], direct(findall(inbin(in2, bins[2], 1) .& ingrid))[1][1])
        end
        # A row without input stars stays NaN
        c, b, e = process_ASTs(table, (:in1, :in2), (:out1, :out2), (vcat(bins[1], 22.5), bins[2]), selectfunc)[2:4]
        @test all(isnan, c[end, :]) && all(isnan, b[1][end, :]) && all(isnan, e[1][end, :])
        @test !any(isnan, c[begin:end-1, :])
    end
end
