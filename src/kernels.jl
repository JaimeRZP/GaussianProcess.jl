"""
    sqexp_cov_fn(X; delta=0.0005, kwargs...)

Implementation of the square exponential kernel. 

Arguments:

- `X::Vector{Float}` : node's position vector 
- `delta::Float` : strength of white noise component for numerical stability. 
- `eta::Number` : covariance matrix amplitude.
- `l::Number` : covariance matrix correlation length.

Returns:

- `cov_mat::Matrix` : GP's covariance matrix

Usage:

```julia
sqexp_cov_fn(x; eta=0.05, l=1.0)
```
"""
function sqexp_cov_fn(X; delta=0.0005, kwargs...)
    X_mat = _turn_into_mat(X)
    D = pairwise(Distances.Euclidean(), X_mat, dims=1)
    return @.(kwargs[:eta] * exp(-D^2 / (2*kwargs[:l]))) + delta * I
end 

"""
    exp_cov_fn(X; delta=0.0005, kwargs...)

Implementation of the exponential kernel also known as Ornstein–Uhlenbeck.

Arguments:

- `X::Vector{Float}` : node's position vector 
- `delta::Float` : strength of white noise component for numerical stability. 
- `eta::Number` : covariance matrix amplitude.
- `l::Number` : covariance matrix correlation length.

Returns:

- `cov_mat::Matrix` : GP's covariance matrix

Usage:

```julia
exp_cov_fn(x; eta=0.05, l=1.0)
```
"""
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