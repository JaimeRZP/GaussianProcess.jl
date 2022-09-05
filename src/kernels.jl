"""
    const_cov_fn(X; delta=0.0005, kwargs...)

Implementation of the constant kernel. 

Arguments:

- `X::Vector{Float}` : node's position vector 
- `delta::Float` : strength of white noise component for numerical stability. 
- `C::Number` : constant covariance matrix.

Returns:

- `cov_mat::Matrix` : GP's covariance matrix

Usage:

```julia
const_cov_fn(x; C=1.0)
```
"""
function const_cov_fn(X; delta=0.0005, kwargs...)
    return kwargs[:C] + delta * I
end 

"""
    lin_cov_fn(X; delta=0.0005, kwargs...)

Implementation of the linear kernel. 

Arguments:

- `X::Vector{Float}` : node's position vector 
- `delta::Float` : strength of white noise component for numerical stability. 
- `C::Number` : amplitude of costant term.
- `a::Number` : amplitude of linear term.

Returns:

- `cov_mat::Matrix` : GP's covariance matrix

Usage:

```julia
lin_cov_fn(x; C=0.05, a=1.0)
```
"""
function lin_cov_fn(X; delta=0.0005, kwargs...)
    return  kwargs[:C] + kwargs[:a].*(X' .* X) + delta * I
end

"""
    noise_cov_fn(X; delta=0.0005, kwargs...)

Implementation of the white noise kernel. 

Arguments:

- `X::Vector{Float}` : node's position vector 
- `delta::Float` : strength of white noise component for numerical stability. 
- `a::Number` : noise's amplitude.

Returns:

- `cov_mat::Matrix` : GP's covariance matrix

Usage:

```julia
noise_cov_fn(x; d=1.0)
```
"""
function noise_cov_fn(X; delta=0.0005, kwargs...)
    return  (delta + kwargs[:d]) * I
end

"""
    ratquad_cov_fn(X; delta=0.0005, kwargs...)

Implementation of the rational quadratic kernel. 

Arguments:

- `X::Vector{Float}` : node's position vector 
- `delta::Float` : strength of white noise component for numerical stability. 
- `eta::Number` : covariance matrix amplitude.
- `alpha::Number` : covariance matrix spectral index.

Returns:

- `cov_mat::Matrix` : GP's covariance matrix

Usage:

```julia
ratquad_cov_fn(x; eta=0.05, alpha=1.0)
```
"""
function ratquad_cov_fn(X; delta=0.0005, kwargs...)
    X_mat = _turn_into_mat(X)
    D = pairwise(Distances.Euclidean(), X_mat, dims=1)
    return @.(kwargs[:eta] * (1 + abs(D)^2)^(-kwargs[:alpha])) + delta * I
end

"""
    sin_cov_fn(X; delta=0.0005, kwargs...)

Implementation of the (sinusoidal) periodic kernel. 

Arguments:

- `X::Vector{Float}` : node's position vector 
- `delta::Float` : strength of white noise component for numerical stability. 
- `eta::Number` : covariance matrix amplitude.
- `l::Number` : covariance matrix correlation length.

Returns:

- `cov_mat::Matrix` : GP's covariance matrix

Usage:

```julia
sin_cov_fn(x; eta=0.05, l=1.0)
```
"""
function sin_cov_fn(X; delta=0.0005, kwargs...)
    X_mat = _turn_into_mat(X)
    D = pairwise(Distances.Euclidean(), X_mat, dims=1)
    return @.(kwargs[:eta] * exp(-2*sin(D/2)^2 / (kwargs[:l]^2))) + delta * I) + delta * I
end

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