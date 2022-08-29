"""
Kernel functions used to produce the \
the covariance matrix of the Gaussian Processes. \
See https://en.wikipedia.org/wiki/Gaussian_process \
for details. \
"""
function sqexp_cov_fn(X; delta=0.0005, kwargs...)
    """
    Implementation of the square exponential kernel. \
    Args:
        X (Vector{Float}): node's position vector \
    kwargs: 
        delta = 0.0005: strength of white noise \
                        component for \
                        for numerical stability. 
        eta (Number): cov. mat. amplitude.
        l (Number): cov. mat. correlation length.
    Returns:
        cov_mat (Matrix (X,X)): GP's cov. mat.
    """
    # Squared-exponential covariance function
    X_mat = _turn_into_mat(X)
    D = pairwise(Distances.Euclidean(), X_mat, dims=1)
    return @.(kwargs[:eta] * exp(-D^2 / (2*kwargs[:l]))) + delta * I
end 

function exp_cov_fn(X; delta=0.0005, kwargs...)
    """
    Implementation of the exponential kernel \
    also known as Ornstein–Uhlenbeck. \
    Args:
        X (Vector{Float}): node's position vector \
    kwargs: 
        delta = 0.0005: strength of white noise \
                        component for \
                        for numerical stability. 
        eta (Number): cov. mat. amplitude.
        l (Number): cov. mat. correlation length.
    Returns:
        cov_mat (Matrix (X,X)): GP's cov. mat.
    """
    # Exponential covariance function
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