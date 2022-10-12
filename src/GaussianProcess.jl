module GaussianProcess

export const_cov_fn, lin_cov_fn, noise_cov_fn, ratquad_cov_fn
export sin_cov_fn, exp_cov_fn, sqexp_cov_fn, sqexp_cov_grad
export marginal_lkl, latent_GP, conditional, posterior_predict

using Distributions, Distances, LinearAlgebra

include("kernels.jl")
include("gp.jl")

end
