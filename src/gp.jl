function marginal_lkl(mean, kernel; data_cov=nothing)
    if data_cov != nothing
        kernel .+= data_cov
    end
    return MvNormal(mean, kernel)
end

function posterior_predict(X_new, X_old, mean_new, mean_old, data, cov_fn;
                            data_cov=data_cov)
    M = size(X_new, 1)
    Z = [X_new; X_old]
    return (;kwargs...) -> let
        K = sqexp_cov_fn(Z, kwargs[:eta], kwargs[:l])
        Koo = K[(M+1):end, (M+1):end] + data_cov
        Knn = K[1:M, 1:M]
        Kno = K[1:M, (M+1):end]
        Koo_inv = inv(Koo)
        C = Kno * Koo_inv
        m = C * (data - mean_old) + mean_new
        S = Matrix(LinearAlgebra.Hermitian(Knn - C * Kno'))
        mvn = MvNormal(m, S)
        rand(mvn)
    end
end

function latent_GP(mean, nodes, kernel)
    return mean .+ cholesky(kernel).U' * nodes
end

function conditional(old_X, new_X, latent_gp, cov_fn; kwargs...)
    N, P = size(new_X)
    Z = [new_X; old_X]
    K = cov_fn(Z, kwargs[:eta], kwargs[:l])
    Koo = K[(N+1):end, (N+1):end]    
    Kno = K[1:N, (N+1):end]          
    
    Koo_inv = inv(Koo)
    C = Kno * Koo_inv
    return C * latent_gp
end
