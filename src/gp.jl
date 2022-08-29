function marginal_lkl(mean, kernel; data_cov=nothing)
    """
    Marginal likelihood implementation of a GP's. \
    This is equivalent to analytically marginalizing \
    over the GP's nodes. \
    Used when GP is linearly related to data. \
    Args:
        mean (Vector{Number}): GP's mean.
        kernel (Matrix{Number}): GP's cov mat.
    Kwargs:
        data_cov (Matrix{Float}): data cov. mat.
    Returns:
        GP (MvNormal): instance of MvNormal.
    """
    if data_cov != nothing
        kernel .+= data_cov
    end
    return MvNormal(mean, kernel)
end

function posterior_predict(X_new, X_old, mean_new, mean_old, data, cov_fn;
                            data_cov=nothing)
    """
    Returns a function that transforms the GP \
    from an old parameter space to a new parameter space, \
    given a particular set of values for the GP's covariance \
    matrix hyperparameters. \
    Args:
        new_X (Vector{Float}): target position vector.
        old_X (Vector{Float}): position vector of the \
                               latent GP.
        new_mean (Vector{Number}): expected GP mean \
                                   for new position vector. 
        old_mean (Vector{Number}): GP's mean in old \
                                   position vector. 
        data (Vector{Float}): data vector.
        cov_fn (function): covariance function used to generate \
                           the GP's kernel. 
    Kwargs:
        data_cov (Matrix{Float}): data cov. mat.
    Returns:
        predict (function): predicts GP in target space.
            Kwargs:
                a=a, b=b... : Cov mat hyperparameters
            Returns:
                GP (MvNormal): instance of MvNormal.
    """
    if data_cov == nothing
        data_cov = zeros(length(X_new), length(X_new))
    end
    M = size(X_new, 1)
    Z = [X_new; X_old]
    return (;kwargs...) -> let
        K = cov_fn(Z; kwargs...)
        Koo = K[(M+1):end, (M+1):end] + data_cov
        Knn = K[1:M, 1:M]
        Kno = K[1:M, (M+1):end]
        Koo_inv = inv(Koo)
        C = Kno * Koo_inv
        m = C * (data - mean_old) + mean_new
        S = Matrix(LinearAlgebra.Hermitian(Knn - C * Kno'))
        return MvNormal(m, S)
    end
end

function latent_GP(mean, nodes, kernel)
    """
    Latent variable implementation of a GP's. \
    Rotates the GP ndodes by the covariance \
    matrix and adds them to the mean vector. \
    Used when GP is not linearly related to data. \
    Args:
        mean (Vector{Number}): GP's mean.
        nodes (Vector{Number}): GP's nodes.
        kernel (Matrix{Number}): GP's cov mat.
    Returns:
        GP (Vector{Number}): Gaussian process realization.
    """
    return mean .+ cholesky(kernel).U' * nodes
end

function conditional(old_X, new_X, latent_gp, cov_fn; kwargs...)
    """
    Given the GP's covariance matrix, applies a \
    Wiener filter to transform the latent GP \
    N-dimensional parameter space to M-dimensional \
    target parameter space. \
    Args:
        old_X (Vector{Float}): position vector of the \
                               latent GP.
        new_X (Vector{Float}): target position vector.
        latent_gp (Vector{Number}): latent GP realization. 
    Kwargs:
        a=a, b=b... : Cov mat hyperparameters 
    Returns:
        gp (Vector{Number}): GP in target space. 
    """
    N = length(new_X)
    Z = [new_X; old_X]
    K = cov_fn(Z; kwargs...)
    Koo = K[(N+1):end, (N+1):end]    
    Kno = K[1:N, (N+1):end]          
    
    Koo_inv = inv(Koo)
    C = Kno * Koo_inv
    return C * latent_gp
end
