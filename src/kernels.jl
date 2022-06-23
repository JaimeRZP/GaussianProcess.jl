# Squared-exponential covariance function
function sqexp_cov_fn(X, eta, l)
    D = pairwise(Distances.Euclidean(), X, dims=1)
    return @.(eta * exp(-D^2 / (2*l))) + 0.0005 * LinearAlgebra.I
end 

# Exponential covariance function
function exp_cov_fn(X, eta, l)
    D = pairwise(Distances.Euclidean(), X, dims=1)
    return   @.(eta * exp(-D / l)) + 0.0005 * LinearAlgebra.I
end