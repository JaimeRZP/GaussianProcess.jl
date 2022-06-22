# Squared-exponential covariance function
function sqexp_cov_fn(eta, l; delta=0.005)
    return (X) -> let
        if length(size(X)) == 1 
            X_mat = zeros(length(X), 1)
            X_mat[:, 1] = X
        else 
            X_mat = X
        end
        D = pairwise(Distances.Euclidean(), X_mat, dims=1)
        K =  @.(eta * exp(-D^2 / (2*l))) + delta * LinearAlgebra.I
    end
end 

# Exponential covariance function
function exp_cov_fn(eta, l; delta=0.005)
    return (X) -> let
        if length(size(X)) == 1 
            X_mat = zeros(length(X), 1)
            X_mat[:, 1] = X
        else 
            X_mat = X
        end
        D = pairwise(Distances.Euclidean(), X_mat, dims=1)
        K = @.(eta * exp(-D / l)) + delta * LinearAlgebra.I
    end
end