abstract type GP_type end

struct GP <: GP_type
    X::Matrix{Float64}
    mean::Vector{Float64}
    cov_func
end

GP(X, mean, cov_func) = begin
    GP(X, mean, cov_fuc)
end

struct latent_GP <: GP_type
    X::Matrix{Float64}
    mean::Vector{Float64}
    nodes
    latent_gp
    cov_func
end

latent_GP(X, mean, nodes, cov_func) = begin
    kernel = cov_func(X)
    N = length(X)
    latent_gp = mean .+ cholesky(kernel).U' * nodes
    latent_GP(X, mean, nodes, latent_gp, cov_func)
end


function marginal_lkl(GP::GP_type; data_cov=nothing)
    kernel = GP.cov_func(GP.X)
    if data_cov != nothing
        kernel .+= data_cov
    end
    return MvNormal(GP.mean, kernel)
end

function conditional(new_X, latent_GP::GP_type)
    N, P = size(new_X)
    Z = [new_X; latent_GP.X]
    K = latent_GP.cov_func(Z)
    Koo = K[(N+1):end, (N+1):end]    
    Kno = K[1:N, (N+1):end]          
    
    Koo_inv = inv(Koo)
    C = Kno * Koo_inv
    return C * latent_GP.latent_gp
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
