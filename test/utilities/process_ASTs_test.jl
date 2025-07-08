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
        a_error[i] = median(abs.(curr_diff .- a_bias[i]))
    end
    # Loop over all supported table types
    for ttype in (Table, DataFrame)
        # Basic call
        result = process_ASTs(ttype(in=inmags, out=outmags, flag=flag),
                              :in, :out, bins, x->x.flag==true; statistic=median)
        @test result isa NTuple{4, Vector{Float64}}
        @test all(result[2] .== 1)
        @test result[3] ≈ a_bias
        @test result[4] ≈ a_error
        
        # Use different statistic and test the answer is different
        result2 = process_ASTs(ttype(in=inmags, out=outmags, flag=flag),
                               :in, :out, bins, x->x.flag==true;
                               statistic=mean)
        @test result2 isa NTuple{4, Vector{Float64}}
        @test all(result2[2] .== 1)
        @test !(result2[3] ≈ a_bias)
        @test !(result2[4] ≈ a_error)
        
        # Make sure selectfunc is properly utilized to filter input ASTs
        outmags2 = copy(outmags)
        outmags2[1:100] .= 99.999
        result3 = process_ASTs(ttype(in=inmags, out=outmags2, flag=flag),
                               :in, :out, bins,
                               x-> (x.flag==true) & (x.out != 99.999);
                               statistic=median)
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
                              x -> (x.flag == true) & (x.in > bins[2]);
                              statistic=median)
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
                              x -> (x.flag == true) & ((x.in < bins[2]) | (x.in >= bins[3]));
                              statistic=median)
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
                               x -> (x.flag == true);
                               statistic=median)
        @test result6 isa NTuple{4, Vector{Float64}}
        @test isnan(result6[2][end])
        @test all(result6[2][begin:end-1] .== 1)
        @test isnan(result6[3][end])
        @test result6[3][begin:end-1] ≈ a_bias
        @test isnan(result6[4][end])
        @test result6[4][begin:end-1] ≈ a_error
    end
end
