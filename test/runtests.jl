using Test
using GaussianProcess
using Distances
using LinearAlgebra

@testset "All tests" begin
    
    @testset "kernels" begin
        X = Vector(1:10)
        N = length(X)
        X_mat = zeros(N, 1)
        X_mat[:,1] = X
        D = pairwise(Distances.Euclidean(), X_mat, dims=1)

        const_K_bm = 1.0
        lin_K_bm = 1.0 .+ 0.5.* (X' .* X)
        noise_K_bm = 0.5 * I
        ratquad_K_bm =  @.(0.5*(1 + abs(D)^2)^0.5)
        sin_K_bm = @.(0.5 * exp(-2*sin(D/2)^2 / (0.5^2)))
        sqexp_K_bm = @. exp(-D^2)
        exp_K_bm = @. exp(-D)

        const_K = const_cov_fn(X; C=1)
        lin_K = lin_cov_fn(X; C=1, a=0.5)
        noise_K = noise_cov_fn(X; d=0.5)
        ratquad_K = ratquad_cov_fn(X; eta=0.5, alpha=0.5)
        sin_K = sin_cov_fn(x; eta=0.5, l=0.5)
        sqexp_K = sqexp_cov_fn(X; eta=1, l=0.5)
        exp_K = exp_cov_fn(X; eta=1, l=1)
            
        @test all(@. (abs(const_K/const_K_bm-1.0) < 0.00051))
        @test all(@. (abs(lin_K/lin_K_bm-1.0) < 0.00051))
        @test all(@. (abs(noise_K/noise_K_bm-1.0) < 0.00051))
        @test all(@. (abs(ratquad_K/ratquad_K_bm-1.0) < 0.00051))
        @test all(@. (abs(sin_K/sin_K_bm-1.0) < 0.00051))
        @test all(@. (abs(sqexp_K/sqexp_K_bm-1.0) < 0.00051))
        @test all(@. (abs(exp_K/exp_K_bm-1.0) < 0.00051))
    end
    
    @testset "marginal_lkl" begin   
        X = Vector(1:10)
        sqexp = sqexp_cov_fn(X; eta=1, l=0.5)

        @test marginal_lkl(X, sqexp).μ == X
        @test marginal_lkl(X, sqexp).Σ == sqexp
    end
    
    @testset "latent_GP" begin   
        X = Vector(ones(10))
        kernel = Diagonal(X)
        @test latent_GP(X, X, kernel) == 2.0 * X
    end
    
    @testset "conditional" begin
        Y = Vector(ones(10))
        X = Vector(1:10)
        cond = conditional(X, X, Y, sqexp_cov_fn; eta=1, l=0.5)
        @test abs.(cond ./ Y .- 1) < 0.0005 * Y
    end
    
    @testset "posterior_predict" begin   
        Y = Vector(ones(10))
        Z = Vector(zeros(10))
        gp_predict = posterior_predict(Y, Y, Z, Z, Y, sqexp_cov_fn)
        dist = gp_predict(delta=0.0005, eta=1, l=0.5)
        
        K0 = sqexp_cov_fn(Y, eta=1, l=0.5, delta=0.0005)
        K1 = sqexp_cov_fn(Y, eta=1, l=0.5, delta=0.0)
        C = K1 * inv(K0) 
        

        @test dist.μ == C * (Y - Z) + Z
        @test dist.Σ == K0 - C * K1'
    end

end