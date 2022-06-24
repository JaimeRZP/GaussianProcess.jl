# Squared-exponential covariance function
function sqexp_cov_fn(X; delta=0.0005, kwargs...)
    X_mat = _turn_into_mat(X)
    D = pairwise(Distances.Euclidean(), X_mat, dims=1)
    return @.(kwargs[:eta] * exp(-D^2 / (2*kwargs[:l]))) + delta * I
end 

# Exponential covariance function
function exp_cov_fn(X; delta=0.0005, kwargs...)
    X_mat = _turn_into_mat(X)
    D = pairwise(Distances.Euclidean(), X_mat, dims=1)
    return   @.(kwargs[:eta] * exp(-D / kwargs[:l])) + delta * I
end

function _turn_into_mat(X)
    N = length(X)
    X_mat = zeros(N, 1)
    X_mat[:,1] = X
    return X_mat
end