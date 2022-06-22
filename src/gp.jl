abstract type GP_type end

struct GP <: GP_type
    X::Vector{Float64}
    mean::Vector{Float64}
    cov_func
    data_cov::Matrix{Float64}
end

GP(X, mean, cov_func; data_cov=nothing) = begin
    if data_cov == nothing
        N = length(mean)
        data_cov = zeros(N,N)
    end
    GP(X, mean, cov_fuc, data_cov)
end

function marginal_lkl(GP::GP_type)
    kernel = GP.cov_func(GP.X)
    return MvNormal(GP.mean, kernel + GP.data_cov)
end

function latent_func(nodes, GP::GP_type)
    kernel = GP.cov_func(GP.X)
    latent_gp = GP.mean .+ cholesky(kernel).U' * nodes
    gp = conditional(latent_gp)
end

function conditional(new_X, nodes, GP::GP_type)
    N = length(GP.X)
    Z = [GP.X; new_X]
    Kernel = GP.cov_func(Z)
    Koo = K[(N+1):end, (N+1):end]    
    Kno = K[1:N, (N+1):end]          
    
    Koo_inv = inv(Koo)
    C = Kno * Koo_inv
    return C * nodes
end

function posterior_predict(old_X, new_X, old_mean, 
                           new_mean, data, cov_func;
                           data_cov=nothing)
    
    if data_cov == nothing
        N = length(data)
        data_cov = zeros(N,N)
    end
    
    M = size(new_X, 1)
    Z = [new_X; old_X]
    
    return (;kwargs...) -> let
        cov_fn = cov_func(kwargs[:eta], kwargs[:l])
        K = cov_fn(Z)
        Koo = K[(M+1):end, (M+1):end] + data_cov
        Knn = K[1:M, 1:M]
        Kno = K[1:M, (M+1):end]

        Koo_inv = inv(Koo)
        C = Kno * Koo_inv

        m = C * (data - old_mean) + new_mean
        S = Matrix(LinearAlgebra.Hermitian(Knn - C * Kno'))
        return MvNormal(m, S)
    end
end
