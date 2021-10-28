#This file expands on the VFit implementation and provides a least square pole extension



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
function loss_calc(r::VFit,f_df::DataFrame,weights::Vector = Float64[])
    f_df = f_df[sortperm(f_df[:,1]),:]
    l = rapprox.((r,),f_df[:,1]) .- f_df[:,2]
    test_vec = f_df[:,1]
    if isempty(weights)
        weights = zeros(Float64,length(test_vec))
        weights[2:end] = diff(weights)
        weights[1]= weights[2]
    end
    loss = (0.5/length(l))*sum(weights .* abs2.(l))
    return loss
end

"""
Build Least squares differential for updating xi
"""
function grad_L(r::VFit,f_df::DataFrame,weights::Vector = Float64[])
    l = rapprox.((r,),f_df[:,1]) .- f_df[:,2]

    test_vec = f_df[:,1]
    if isempty(weights)
        weights = zeros(Float64,length(test_vec))
        weights[2:end] = diff(weights)
        weights[1]= weights[2]
    end
    l= sqrt.(weights) .* l
    p = zeros(ComplexF64,length(l),length(r.φ))
    q = zeros(ComplexF64,length(l),length(r.ψ))
    dp = zeros(ComplexF64,length(l),length(r.φ))
    dq = zeros(ComplexF64,length(l),length(r.ψ))
    #grad = 0.0
    Y = f_df[:,2]
    for j in 1:length(l)
        p[j,:]  = r.φ ./ (Y[j] .- r.poles)
        q[j,:]  = r.ψ ./ (Y[j] .- r.poles)
        dp[j,:] = r.φ ./ ((Y[j] .- r.poles).^2)
        dq[j,:] = r.ψ ./ ((Y[j] .- r.poles).^2) 
        
        

    end


end


"""
Build least squares for ξ
"""
function pole_update()

end