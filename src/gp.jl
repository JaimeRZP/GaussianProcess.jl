function latent_GP(mean, nodes, kernel)
    return mean .+ cholesky(kernel).U' * nodes
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
