# import DataFrames: DataFrame
using StarFormationHistories: process_ASTs
using StableRNGs: StableRNG
using StatsBase: median, mean
using Test
using TypedTables: Table
using DataFrames: DataFrame

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
        a_error[i] = median(abs.(curr_diff))
    end
    result = process_ASTs(Table(in=inmags, out=outmags, flag=flag),
                          :in, :out, bins, x->x.flag==true)
    @test result isa NTuple{4, Vector{Float64}}
    @test all(result[2] .== 1)
    @test result[3] ≈ a_bias
    @test result[4] ≈ a_error
    # Use different statistic and test the answer is different
    result2 = process_ASTs(Table(in=inmags, out=outmags, flag=flag),
                           :in, :out, bins, x->x.flag==true;
                           statistic=mean)
    @test result2 isa NTuple{4, Vector{Float64}}
    @test all(result2[2] .== 1)
    @test !(result2[3] ≈ a_bias)
    @test !(result2[4] ≈ a_error)
    # Make sure selectfunc is properly utilized to filter input ASTs
    outmags2 = copy(outmags)
    outmags2[1:100] .= 99.999
    result3 = process_ASTs(Table(in=inmags, out=outmags2, flag=flag),
                           :in, :out, bins, x-> (x.flag==true) & (x.out != 99.999))
    @test result3 isa NTuple{4, Vector{Float64}}
    # Make sure completeness is properly affected by bad stars
    @test all(result3[2] .== vcat((nstars_iter-100)/nstars_iter, ones(length(bin_centers)-1)))
    # Bias and error wont be exactly the same but close
    @test result3[3] ≈ a_bias rtol=1e-2
    @test result3[4] ≈ a_error rtol=1e-2
    # Test on dataframe
    result_df = process_ASTs(DataFrame(in=inmags, out=outmags, flag=flag),
                             :in, :out, bins, x->x.flag==true)
    @test result_df isa NTuple{4, Vector{Float64}}
    @test all(result_df[2] .== 1)
    @test result_df[3] ≈ a_bias
    @test result_df[4] ≈ a_error
end
