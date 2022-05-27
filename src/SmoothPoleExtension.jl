#This file expands on the VFit implementation and provides a least square pole extension
#using VFitApproximation

"""
Build regression parameter for psi
"""
function Γ_regrr(λ_φ,λ_ψ,m)
    iden1 =Diagonal([λ_φ,λ_ψ])
    iden2 = Matrix{ComplexF64}(I,m,m)
    return kron(iden1,iden2) 
end

"""
Get Phi and Psi 
"""
function update_φ_ψ(f_vec::Vector, l::AbstractVector,xi::AbstractVector,weights::Vector = Float64[])
    m = length(xi)
    C1 = zeros(ComplexF64,length(l),length(xi))
    C2 = zeros(ComplexF64,length(l),length(xi))
    Y = f_vec
    for j in 1:length(l)                                        #Build the initial matrix in RKFIT paper eq (A.4) to be solved
        C1[j,:] = 1 ./(l[j] .- xi)
        C2[j,:] = -Y[j] ./(l[j] .- xi)
    end
    C = hcat(C1,C2)
    if isempty(weights)
        weights = zeros(Float64,length(l))                          #If there are a lot of samples near a singularity then they will dominate the error, these weights help reweight the uneven sampling
        weights[2:end] = abs.(l[2:end] - l[1:end-1])
        weights[1]= weights[2]
    end

    W = Diagonal(sqrt.(weights))
    A = W*C
    b = W*Y

    #Build Γ
    Γ = Γ_regrr(0,1,m)
    P = (A'A + Γ'Γ)\(A'b)

    return P[1:m], P[m+1:2m]
end

"""
Loss Function for ξ update
"""
function loss_calc(r::VFitApproximation.VFit,f_df::DataFrame,weights::Vector = Float64[])
    f_df = f_df[sortperm(f_df[:,1]),:]
    l = VFitApproximation.rapprox.((r,),f_df[:,1]) .- f_df[:,2]
    test_vec = f_df[:,1]
    if isempty(weights)
        weights = zeros(Float64,length(test_vec))
        weights[2:end] = diff()
        weights[1]= weights[2]
    end


    loss = sum(weights .* abs2.(l)) + sum(abs2.(r.ψ))
    return loss
end

"""
Build Least squares differential for updating xi
"""
function grad_L(r::VFitApproximation.VFit,f_df::DataFrame,weights::Vector = Float64[])
    l = VFitApproximation.rapprox.((r,),f_df[:,1]) .- f_df[:,2]

    test_vec = f_df[:,1]
    if isempty(weights)
        weights = zeros(Float64,length(test_vec))
        weights[2:end] = diff(weights)
        weights[1]= weights[2]
    end
    l= sqrt.(weights) .* l
    dA = zeros(ComplexF64,length(l),length(r.φ)+length(r.ψ))
    x = vcat(r.φ,r.ψ)
    g2 = zeros(ComplexF64,length(l),length(r.poles))
    #grad = 0.0
    λ = f_df[:,1]
    Y = f_df[:,2]
    for j in 1:length(l)
        # p  = sum(r.φ ./ (λ[j] .- r.poles))
        # q  = sum(r.ψ ./ (λ[j] .- r.poles))
        # dp = r.φ ./ ((λ[j] .- r.poles).^2)
        # dq = r.ψ ./ ((λ[j] .- r.poles).^2) 
        
        # g2[j,:] = l[j].*(dp ./(1+q) .- (p/((1+q)^2) .* dq))
        
        dA[j,1:length(r.φ)] = -1.0./((λ[j] .- r.poles).^2)
        dA[j,length(r.φ)+1:end] = Y[j]./((λ[j] .- r.poles).^2) 
        
    end
    #grad = sum(g2,dims=1)
    
    grad = x'*dA'*l
    return grad
end


"""
Build least squares with Gradient Descent for ξ
"""
function GD_pole_update(f_df,ξ,w::Vector=Float64[],tol=1e-10)

    F_vec = f_df[:,2]
    λ = f_df[:,1]

    ∇L= zeros(ComplexF64,length(ξ))
    prev_ξ = zeros(ComplexF64,length(ξ))
    max_count = 500
    cnt = 0

    φ,ψ = update_φ_ψ(F_vec,λ,ξ,w)
    r = VFitApproximation.VFit(φ,ψ,ξ)

    Err_vec = []
    Loss = loss_calc(r,f_df,w)
    push!(Err_vec,Loss)
    #Update poles via GD
    while (Loss>=tol) && (cnt<=max_count)
        print("At iteration ",cnt)
        print(" The Squared Loss is ",Loss,"\n")
        r.φ, r.ψ = update_φ_ψ(F_vec,λ,r.poles,w)
        gl = grad_L(r,f_df,w)'
        γ = dot(r.poles .- prev_ξ, gl .- ∇L)./dot(gl .- ∇L, gl .- ∇L)
        ∇L = gl
        prev_ξ = r.poles
        r.poles = vec(r.poles .- γ.*∇L)
        
        Loss = loss_calc(r,f_df,w)
        cnt+=1
        push!(Err_vec,Loss)
    end 

    gd_error_plot(Err_vec,length(ξ),length(λ),cnt)
    return r,Err_vec
end