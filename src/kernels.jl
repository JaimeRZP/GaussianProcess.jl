# Squared-exponential covariance function
function sqexp_cov_fn(eta, l; delta=0.005)
    return (X) -> let
        D = pairwise(Distances.Euclidean(), X, dims=1)
        K =  @.(eta * exp(-D^2 / (2*l))) + delta * LinearAlgebra.I
    end
end 

# Exponential covariance function
function exp_cov_fn(eta, l; delta=0.005)
    return (X) -> let
        D = pairwise(Distances.Euclidean(), X, dims=1)
        K = @.(eta * exp(-D / l)) + delta * LinearAlgebra.I
    end
end