# This example demonstrates the different types of Gaussian error kernels
# available for Hess diagram modelling.

import StarFormationHistories as SFH
import StaticArrays: SMatrix
import StatsBase: cov, mean, AnalyticWeights # import Statistics: cov, mean
# import Distributions: Gaussian # Normal 1-D Gaussian for sampling
using Plots
gr()

# Define filter sets
# obs_filters = ["F090W", "F150W", "F277W"]
obs_filters = (:F090W, :F150W, :F277W)
# Define photometric error functions
F090W_error(m) = min( SFH.exp_photerr(m, 1.03, 15.0, 36.0, 0.02), 0.4 )
F150W_error(m) = min( SFH.exp_photerr(m, 1.03, 15.0, 35.0, 0.02), 0.4 )
F277W_error(m) = min( SFH.exp_photerr(m, 1.03, 15.0, 34.0, 0.02), 0.4 )
err_funcs = NamedTuple{obs_filters}((F090W_error, F150W_error, F277W_error))

# Evaluate and sample error distribution at particular mags
mags = NamedTuple{obs_filters}((27.0, 26.5, 26.0))

# Number of MC points to sample to compare to each kernel
n_points = 20_000_000

###########################################################
# Sample for case where 2-D error distribution is separable
# between x and y -- y-axis magnitude does not appear
# in x axis color.

# Color is obs_filters[1] - obs_filters[2] = F090W - F150W
separable_color_indices = (1,2)
# Y-axis magnitude is obs_filters[3] = F277W
separable_y_index = 3
x0_sep = mags[obs_filters[separable_color_indices[1]]] - mags[obs_filters[separable_color_indices[2]]]
y0_sep = mags[obs_filters[separable_y_index]]
# Add errors in quadrature for x-axis 
x0_err_sep = sqrt( err_funcs[obs_filters[separable_color_indices[1]]]( mags[obs_filters[separable_color_indices[1]]] )^2 + err_funcs[obs_filters[separable_color_indices[2]]]( mags[obs_filters[separable_color_indices[2]]] )^2 )
y0_err_sep = err_funcs[obs_filters[separable_y_index]]( mags[obs_filters[separable_y_index]] )
# Sample random MC points given the above errors
x1_mags_sep = randn(n_points) .* err_funcs[obs_filters[separable_color_indices[1]]]( mags[obs_filters[separable_color_indices[1]]] ) .+ mags[obs_filters[separable_color_indices[1]]]
x2_mags_sep = randn(n_points) .* err_funcs[obs_filters[separable_color_indices[2]]]( mags[obs_filters[separable_color_indices[2]]] ) .+ mags[obs_filters[separable_color_indices[2]]]
y_mags_sep = randn(n_points) .* err_funcs[obs_filters[separable_y_index]]( mags[obs_filters[separable_y_index]] ) .+ mags[obs_filters[separable_y_index]]
x_mags_sep = x1_mags_sep .- x2_mags_sep

sep_bins = (range(start=x0_sep - 3*x0_err_sep,stop=x0_sep + 3*x0_err_sep,length=100),
            range(start=y0_sep - 3*y0_err_sep,stop=y0_sep + 3*y0_err_sep,length=100))
# Construct the separable kernel with all variables in units of pixels or bins
separable_kernel = SFH.GaussianPSFAsymmetric( length(sep_bins[1])/2, length(sep_bins[2])/2, x0_err_sep / step(sep_bins[1]), y0_err_sep / step(sep_bins[2]))
# The above x0 and y0 are correct, not the ones below
# separable_kernel = SFH.GaussianPSFAsymmetric( mean(eachindex(sep_bins[1])), mean(eachindex(sep_bins[1])), x0_err / step(sep_bins[1]), y0_err / step(sep_bins[2]))
sep_kernel_img = zeros( length(sep_bins[1])-1, length(sep_bins[2])-1)
# Mutate sep_kernel_img in place to hold the kernel normalized to sum to 1.
SFH.addstar!(sep_kernel_img, separable_kernel)

# Bin the sampled magnitudes 
sep_data_hist = SFH.bin_cmd(x_mags_sep, y_mags_sep; edges=sep_bins)
# Normalize bins to sum to 1, same as kernel
sep_data_hist.weights ./= sum(sep_data_hist.weights)


######################
# Sample for case where 2-D error distribution is
# covariant of the form x=(F090W - F150W), y=F090W.

# Color is obs_filters[1] - obs_filters[2] = F090W - F150W
covar1_color_indices = (1,2)
# Y-axis magnitude is obs_filters[1] = F090W
covar1_y_index = 1
x0_covar1 = mags[obs_filters[covar1_color_indices[1]]] - mags[obs_filters[covar1_color_indices[2]]]
y0_covar1 = mags[obs_filters[covar1_y_index]]
# Add errors in quadrature for x-axis 
x0_err_covar1 = sqrt( err_funcs[obs_filters[covar1_color_indices[1]]]( mags[obs_filters[covar1_color_indices[1]]] )^2 + err_funcs[obs_filters[covar1_color_indices[2]]]( mags[obs_filters[covar1_color_indices[2]]] )^2 )
y0_err_covar1 = err_funcs[obs_filters[covar1_y_index]]( mags[obs_filters[covar1_y_index]] )
# Sample random MC points given the above errors
x1_mags_covar1 = randn(n_points) .* err_funcs[obs_filters[covar1_color_indices[1]]]( mags[obs_filters[covar1_color_indices[1]]] ) .+ mags[obs_filters[covar1_color_indices[1]]]
x2_mags_covar1 = randn(n_points) .* err_funcs[obs_filters[covar1_color_indices[2]]]( mags[obs_filters[covar1_color_indices[2]]] ) .+ mags[obs_filters[covar1_color_indices[2]]]
x_mags_covar1 = [x1_mags_covar1, x2_mags_covar1]
# This defines the covariance pattern
y_mags_covar1 = copy(x_mags_covar1[findfirst(x -> x == covar1_y_index, covar1_color_indices)]) 
x_mags_covar1 = x1_mags_covar1 .- x2_mags_covar1

# covar1_bins = (range(start=x0_covar1 - 3*x0_err_covar1,stop=x0_covar1 + 3*x0_err_covar1,length=100),
#                range(start=y0_covar1 - 3*y0_err_covar1,stop=y0_covar1 + 3*y0_err_covar1,length=100))
covar1_bins = (range(start=x0_covar1 - 3*x0_err_covar1,stop=x0_covar1 + 3*x0_err_covar1, step=0.003),
               range(start=y0_covar1 - 3*y0_err_covar1,stop=y0_covar1 + 3*y0_err_covar1, step=0.003))
# Construct the covariant kernel with all variables in units of pixels or bins
# covar1_matrix = SMatrix{2,2}(σx^2,σy^2,σy^2,σy^2)
covar1_matrix = SMatrix{2,2}( (x0_err_covar1 / step(covar1_bins[1]))^2,
                              (y0_err_covar1 / step(covar1_bins[2]))^2,
                              (y0_err_covar1 / step(covar1_bins[2]))^2,
                              (y0_err_covar1 / step(covar1_bins[2]))^2 )
# Determine the covariance matrix from the random samples numerically
# covar1_matrix = cov([x_mags_covar1 y_mags_covar1]) ./ step(covar1_bins[1])^2
covar1_kernel = SFH.Gaussian2D( length(covar1_bins[1])/2 + 0.25, length(covar1_bins[2])/2 + 0.5, covar1_matrix)

covar1_kernel_img = zeros( length(covar1_bins[1])-1, length(covar1_bins[2])-1)
# Mutate sep_kernel_img in place to hold the kernel normalized to sum to 1.
SFH.addstar!(covar1_kernel_img, covar1_kernel)

# Bin the sampled magnitudes 
covar1_data_hist = SFH.bin_cmd(x_mags_covar1, y_mags_covar1; edges=covar1_bins)
# Normalize bins to sum to 1, same as kernel
covar1_data_hist.weights ./= sum(covar1_data_hist.weights)

# Calculate means of the matrices
function matrix_mean(x_bins, y_bins, matrix)
    # Assume that size(matrix) == (length(x_bins)-1, length(y_bins)-1)
    x_bin_centers = minimum(x_bins) .+ cumsum(diff(x_bins))
    y_bin_centers = minimum(y_bins) .+ cumsum(diff(y_bins))
    weights = AnalyticWeights(vec(matrix))
    x_mean = mean( repeat(x_bin_centers, outer=length(y_bin_centers)), weights)
    y_mean = mean( repeat(y_bin_centers, outer=length(x_bin_centers)), weights)
    return (x_mean, y_mean)
end
covar1_data_means = matrix_mean(covar1_bins[1], covar1_bins[2], covar1_data_hist.weights)
covar1_kernel_means = matrix_mean(covar1_bins[1], covar1_bins[2], covar1_kernel_img)
# println("Pixel difference in covar1 x means: ", (first(covar1_data_means) - first(covar1_kernel_means)) / step(first(covar1_bins)) )
# println("Pixel difference in covar1 y means: ", (last(covar1_data_means) - last(covar1_kernel_means)) / step(last(covar1_bins)) )

######################
# Make plot

# This SHOULD work, but for some reason with Plots.jl there is some scaling problem between histogram2d and heatmap that makes the two seem differently scaled. Safer just sticking with heatmap for both MC-sampled data and the smooth kernel.
# r1 = histogram2d( x_mags, y_mags, sep_bins; xticks=true, yticks=true, show_empty_bins=true, colorbar=true, normalize=:probability, clim=(0,0.00125))

# Arguments for heatmap:
# heatmap(x, y, z)
# ERROR: ArgumentError: Length of x & y does not match the size of z.
# Must be either `size(z) == (length(y), length(x))` (x & y define midpoints)
# or `size(z) == (length(y)+1, length(x)+1))` (x & y define edges).

l = @layout [ grid(3,3) ]
# Separable kernel
r1 = heatmap(sep_data_hist)
k1 = heatmap(sep_bins[2], sep_bins[1], sep_kernel_img)
diff1 = heatmap(sep_bins[2], sep_bins[1], sep_data_hist.weights .- sep_kernel_img)
# First covariant kernel
r2 = heatmap(covar1_data_hist.edges[2], covar1_data_hist.edges[1], covar1_data_hist.weights)
k2 = heatmap(covar1_bins[2], covar1_bins[1], covar1_kernel_img)
diff2 = heatmap(covar1_bins[2], covar1_bins[1], covar1_data_hist.weights .- covar1_kernel_img)
plot(r1, k1, diff1, r2, k2, diff2; layout=l, size=(700,700), left_margin=(2,:mm), right_margin=(2,:mm), xticks=false, yticks=false, show_empty_bins=true, colorbar=false)
# end
