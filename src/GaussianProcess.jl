module GaussianProcess

export exp_cov_fn, sqexp_cov_fn
export GP, marginal_lkl, latent_func, conditional, posterior_predict

using Distributions, Distances, LinearAlgebra

include("kernels.jl")
include("gp.jl")

end
