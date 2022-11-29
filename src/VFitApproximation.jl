module VFitApproximation

# Write your package code here.

using BaryRational
using Random
using Distributions
using LinearAlgebra
using PyPlot
#using MathLink
using DataFrames
using Polynomials

import StatsBase

#include("MathematicaPipelining.jl")
include("PlottingFuncs.jl")
#include("SmoothPoleExtension.jl")


export VFit, rapprox

mutable struct VFit
    φ::AbstractVector
    ψ::AbstractVector
    poles::AbstractVector
end

function rapprox(self::VFit,x::Union{Float64,Complex})
    p = self.φ ./ (x .- self.poles .+ 1e-10)
    q = self.ψ ./ (x .- self.poles .+ 1e-10)
    r = sum(p)/(1+sum(q))
    return r
end

"""
Function that samples points from an exponential distribution
Inputs  theta of exponential distribution,
        num is number of points
Output  Returns num number of points sampled from an exponential distribution in (0,1)
"""
function create_sample_points(theta::Float64,num::Int)
    dist = Exponential(theta)
    pts = rand(dist,num)
    while any(x->x>=1,pts)
        pts[pts .>=1] =  rand(dist, length(pts[pts .>= 1]))
    end
    return pts
end

"""
Function that samples points from a mixed distribution--- an exponential distribution (0,1) and a Uniform distribution [0.2,1)
Input   θ of the exponential distribution (θ*exp[-θ*x]),
        prior for the mixed distribution and
        num = number of points to sample
Output  Returns num number of points sampled from the mixed distribution with prior
"""
function create_mixed_sample_points(theta::Float64,prior::Float64, num::Int)
    dist1 = Exponential(theta)
    dist2 = Uniform(0.2,1)
    mixmode = MixtureModel([dist1,dist2],[prior,1-prior])
    pts = rand(mixmode,num)
    while any(x->x>=1,pts)
        pts[pts.>=1] = rand(mixmode,length(pts[pts .>= 1]))
    end
    return pts
end

"""
This function lets us obtain AAA approximation, if one chooses to start with an initial guess for poles as the AAA support points
Inputs  f  function we are approximating,
        mm number of support points

Output Returns a AAA approximation of the function.
"""
function firstAAA_step(f::Function,mm::Int)
    Random.seed!(427)
    Z = create_sample_points(0.14,10000)
    aaa_out = aaa(Z,f.(Z),mmax=mm)
    return aaa_out
end

"""
This function is used to calculate the barycentric weights by running a least square approximation to a matrix equation
given by equation (A.4) in the RKFit paper
Inputs  f      the function that is being approximated,
        lambda the sample points,
        xi     the barycentric poles of p(x) and q(x)

Outputs φ The barycentric weights of the numerator p(x)
        ψ The barycentric weights of the denominator q(x)
"""
function get_phi_psi(f::Function,lambda::AbstractVector,xi::AbstractVector)
    m = length(xi)
    l = sort(lambda)
    #l=lambda
    C1 = zeros(ComplexF64,length(lambda),length(xi))
    C2 = zeros(ComplexF64,length(lambda),length(xi))
    Y = f.(l)
    for j in 1:length(l)                                        #Build the initial matrix in RKFIT paper eq (A.4) to be solved
        C1[j,:] = 1 ./(l[j] .- xi)
        C2[j,:] = -Y[j] ./(l[j] .- xi)
    end
    C = hcat(C1,C2)

    weights = zeros(Float64,length(l))                          #If there are a lot of samples near a singularity then they will dominate the error, these weights help reweight the uneven sampling
    weights[2:end] = abs.(l[2:end] - l[1:end-1])
    weights[1]= weights[2]
    W = Diagonal(sqrt.(weights))
    A = W*C
    P = A \ (W*Y)
    #P = C\Y                                                    #Solve Cx=y matrix equation and get th phi's and psi's
    return P[1:m], P[m+1:2m]

end


function get_phi_psi(f_vec::Vector, l::AbstractVector,xi::AbstractVector;weights::Vector = Float64[],α::Number=0.0)
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
    if α==0
        P = A\(W*Y)
    else
        Γ =  Γ_regrr(α,0,m)
        P = (A'A+ Γ)/(A'(W*Y))
    end

    #P = C\Y                                                    #Solve Cx=y matrix equation and get th phi's and psi's
    return P[1:m], P[m+1:2m]

end


"""
This function returns the roots and poles of q(x), the Barycentric denominator by pipelining it through Mathematica
Inputs r_ = The rational function at the current step
Outputs   = Roots of the barycentric denominator
"""
# function Find_roots_using_Mathematica(r_ :: VFit)
#     Darrexp  = build_denom.(r_.ψ,r_.poles)                                  #Write rational function denominator in correct form in MathLink expression
#     Dexp = weval(addemup(Darrexp,length(Darrexp)))                          #Evaluate the built expression in MathLink form
#     #Dfactored = weval(W"Factor"(Dexp))                                      #Run Factor from Mathematica to turn expression into (x-z_1)(x-z_2)....(x-z_m)/(x-p_1)(x-p_2)...(x-p_m)
#     Dfactored = weval(W"Roots"(W"Equal"(Dexp,0),W"x"))                     #Run Reduce[] from Mathematica to get output x=z_1 || x=z_2 ... etc 
#     #print("\n",math2Expr(Dfactored))
#     #print("\n",math2Expr(Dexp))
#     roots = ExtractExpr(Dfactored,ComplexF64[])                                #Extract the roots (z_1,z_2,...z_m) from the denominator polynomial q(x)
#     if any(isnan,roots)
#         Dfactored = weval(W"Reduce"(W"Equal"(Dexp,0),W"x"))                 #In the tricky case where Reduce does not provide roots we run Roots function hoping that it would
#         roots = ExtractExpr(Dfactored,ComplexF64[])
#         if any(isnan,roots)
#             print("\n\nNo roots can be found! Exiting with same poles\n\n")
#             return r_.poles
#         end
#     end

#     return roots 
# end

"""
This function returns the roots of the barycentric denominator q(x). The method below uses Julia's Polynomials.jl package. Thus keeping the procedure native
Inputs r_ = The rational function at the current step
Outputs   = Roots of the barycentric denominator
"""
function Find_roots_julia(r_ :: VFit)
    m = length(r_.poles)
    p = fromroots(r_.poles)
    for j=1:m
        p += r_.ψ[j]*fromroots(deleteat!(copy(r_.poles),j))
    end
    return roots(p)
end


"""
Returns the square error of a rational function approximation on a random test sample.
Inputs r_ = The rational function, f:= the function to appoximate
Outputs Least square error
"""
function get_sqrerr(r::VFit, f::Function, test::AbstractVector)
    #test = rand(Uniform(0.01,1),500)
    #Random.seed!(1234)                                      #This seed prevents the test dataset from changing                                      
    #test = sort(rand(Uniform(0.01,1),500))
    #test = collect(range(inps[1],inps[end],length=5000))
    weights = zeros(Float64,length(test))
    weights[2:end] = abs.(test[2:end] - test[1:end-1])
    weights[1]= weights[2]

    residue = real(rapprox.((r,),test)) .- f.(test)
    return sum(weights .* (abs.(residue).^2))
end


"""
Returns the square error of a rational function approximation on a test sample provided by the user.
    Inputs  r_ = The rational function, 
            f:= the function to appoximate
    Outputs Least square error

"""
function get_sqrerr(r::VFit, f_testdf::DataFrame, weights::Vector = Float64[])
    f_testdf = f_testdf[sortperm(f_testdf[:,1]),:]
    residue = rapprox.((r,),f_testdf[:,1]) .- (f_testdf[:,2])
    
    test_vec = f_testdf[:,1]
    if isempty(weights)
        weights = zeros(Float64,length(test_vec))
        weights[2:end] = abs.(test_vec[2:end] - test_vec[1:end-1])
        weights[1]= weights[2]
    end

    
    return sum(weights .* ((abs.(residue)).^2))
end


"""
The main part of the program that combines all the functions and runs the VFit Algorithm
Inputs  f:=               The function to approximate
        m:=               The degree of the polynomials p(x) and q(x) that make up the rational function
        ξ:=               The initial guess of starting support points for the barycentric forms
        λ:=               The training sample points over which the VFit approximation will be made
        tol:=             The tolerance of the least square error, defaults to 1e-10
        force_conjugacy:= Set to true if you want to enforce the approximation to be real. Use for approximating real valued functions

Outputs r     := The rational function
        errors:= The training and testing errors faced
"""
function vfitting(f::Function, m::Int, ξ::AbstractVector, λ::AbstractVector; tol::Float64 =1e-10,iterations::Int=21,force_conjugacy::Bool=false,regression_param::Number=0.0)

    cnt = 1                                                                 #Initialization of the count of iteration
    phi,psi = get_phi_psi(f,λ,ξ;α=regression_param)                                            #Get the first φ and ψ
    r = VFit(phi,psi,ξ)                                                     #Initialize the rational approximation r

    num= length(λ)
    inps = sort(λ)
    test_λ = collect(range(inps[1],inps[end],length=5000))

    sqres =  get_sqrerr(r, f, test_λ)
    self_err = get_sqrerr(r,f,λ)
    trainerrarr = Float64[]
    testerrarr = Float64[]
    print("The initial square error is: $(sqres)\n")
    while (self_err>tol || sqres>10*tol) && cnt<iterations                          #Convergence criteria of 20 used-- if iteration is >20 times it is probably stuck in some local minima
        print("At iteration ",cnt)
        sqres = get_sqrerr(r,f,test_λ)
        self_err = get_sqrerr(r,f,λ)
        print(" The self error is: ",self_err,"\n")
        print(" The square error is: ",sqres,"\n")
        push!(testerrarr,sqres)
        push!(trainerrarr,self_err)
        #r.poles = Find_roots_using_Mathematica(r)                           #Update the poles of the rational function
        new_poles = Find_roots_julia(r)
        if force_conjugacy
            _poles_ = [new_poles[1:2:end]; conj(new_poles[1:2:end])]
            sort!(_poles_,by=x->(real(x),imag(x)))
            r.poles = _poles_
        else
            r.poles = new_poles
        end
         
        r.φ, r.ψ = get_phi_psi(f,λ,r.poles;α=regression_param)                                 #Get the next φ and ψ and update the rational function
        #print("\nPoles\n",r.poles)
        cnt=cnt+1
    end
    #errors = plot_iters(trainerrarr,testerrarr,m,num,cnt)   #Plots the training and testing errors incurred at
    errors = (trainerrarr,testerrarr)
    return r,errors

end

"""
Same as the last function but allows a discritized/sampled function to be fed in (like a Nx2 array of points and function value at those points). 
    This allows us to perform a VFit Approximation on sampled or non continuous functions. 
    You would want to make sure the sample points correspond appropriately to the function values when fed in.
Inputs  f_df:=  The function to approximate in the form of an Array which includes sample points and the function values that correspond to those points
        m:=     The degree of the polynomials p(x) and q(x) that make up the rational function
        ξ:=     The initial guess of starting support points for the barycentric forms
        tol:=   The tolerance of the least square error, defaults to 1e-10
        weightvec:= A vector of weights for each data point. The length of this vector should be same as the number of datapoints
        force_conjugacy:= Set to true if you want to enforce the approximation to be real. Use for approximating real valued functions

Outputs r     := The rational function
        errors:= The training and testing errors faced
"""
function vfitting(f_df::DataFrame, m::Int, ξ::AbstractVector; tol::Float64 =1e-10,  iterations::Int=21, weightvec::AbstractVector = Float64[], force_conjugacy::Bool=false,regression_param::Number=0.0)
    cnt = 1                                                                 #Initialization of the count of iteration
    
    #Split Training and testing into 80 percent train and 20 percent test
    # ixs = randperm(length(f_df[:,1]))
    # tr_num = Int(floor(0.8*length(ixs)))
    # f_train = f_df[ixs[1:tr_num],:]
    # f_test = f_df[ixs[tr_num+1:end],:]
    # if !isempty(weightvec)
    #     weights_train = weightvec[ixs[1:tr_num],:]
    #     weights_test  = weighvec[ixs[tr_num+1:end],:]
    # end
    
    # #Odd-Even splitting
    # f_df = f_df[sortperm(f_df[:,1]),:]
    # f_train = f_df[2:2:length(f_df[:,1]),:]                                 #extracts all the even elements out
    # f_test = f_df[1:2:length(f_df[:,1]),:]
    # if !isempty(weightvec)
    #     weights_train = weightvec[2:2:length(f_df[:,1])]
    #     weights_test  = weightvec[1:2:length(f_df[:,1])]
    # else
    #     weights_train = Float64[]
    #     weights_test  = Float64[]
    # end
 

    #All points
    f_train = f_df
    f_test  = f_df
    if !isempty(weightvec)
        weights_train = weightvec
        weights_test  = weightvec
    else
        weights_train = Float64[]
        weights_test  = Float64[]
    end

    f_train = f_train[sortperm(f_train[:,1]),:]                              #Sort the data
    f_vec = f_train[:,2]
    λ = f_train[:,1]



    #Start the training
    phi,psi = get_phi_psi(f_vec,λ,ξ;weights=weights_train,α=regression_param)                                     #Get the first φ and ψ
    r = VFit(phi,psi,ξ)                                                      #Initialize the rational approximation r

    num= length(λ)
    test_sqres =  get_sqrerr(r,f_test)
    self_err = get_sqrerr(r,f_train)
    trainerrarr = Float64[]
    testerrarr = Float64[]
    print("The initial square error is: $(test_sqres)\n")
    while (test_sqres>tol) && (cnt<iterations)                                             #Convergence criteria of 50 used-- if iteration is >50 times it is probably stuck in some local minima
        print("At iteration ",cnt)
        test_sqres = get_sqrerr(r,f_test)
        self_err = get_sqrerr(r,f_train)
        print(" The self error is: ",self_err,"\n")
        print(" The square error is: ",test_sqres,"\n")
        push!(testerrarr,test_sqres)
        push!(trainerrarr,self_err)
        #r.poles = Find_roots_using_Mathematica(r)                                         #Update the poles of the rational function
        new_poles = Find_roots_julia(r)
        if force_conjugacy
            _poles_ = [new_poles[1:2:end]; conj(new_poles[1:2:end])]
            sort!(_poles_,by=x->(real(x),imag(x)))
            r.poles = _poles_
        else
            r.poles = new_poles
        end
        
        r.φ, r.ψ = get_phi_psi(f_vec,λ,r.poles,weights=weights_train,α=regression_param)                             #Get the next φ and ψ and update the rational function

        cnt=cnt+1
    end
    #errors = plot_iters(trainerrarr,testerrarr,m,num,cnt)                       #Plots the training and testing errors incurred at
    errors = (trainerrarr,testerrarr)
    return r,errors

end

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
Get Phi and Psi and Xi
"""
function update_φ_ψ(f_vec::Vector, λ::AbstractVector,ξ::AbstractVector,weights::Vector = Float64[],α=1.0)
    m = length(ξ)
    C1 = zeros(ComplexF64,length(λ),length(ξ))
    C2 = zeros(ComplexF64,length(λ),length(ξ))
    Y = f_vec
    for j in 1:length(λ)                                        #Build the initial matrix in RKFIT paper eq (A.4) to be solved
        C1[j,:] = 1 ./(λ[j] .- ξ)
        C2[j,:] = -Y[j] ./(λ[j] .- ξ)
    end
    C = hcat(C1,C2)
    if isempty(weights)
        weights = zeros(Float64,length(λ))                          #If there are a lot of samples near a singularity then they will dominate the error, these weights help reweight the uneven sampling
        weights[2:end] = diff(λ)
        weights[1]= weights[2]
    end

    W = Diagonal(sqrt.(weights))
    A = W*C
    b = W*Y

    #Build Γ
    Γ = Γ_regrr(0,α,m)
    P = (A'A + Γ + 1e-7I)\(A'b)

    v = A*P - b
    loss = v'*v + P'*Γ*P
    
    φ = P[1:m]
    ψ = P[m+1:end]

    #dAxv = zeros(ComplexF64,length(λ),length(r.poles))
    #x = vcat(r.φ,r.ψ)
    #g2 = zeros(ComplexF64,length(λ),length(r.poles))
    grad = zeros(ComplexF64,m)

    for i in 1:m
        g = 0.0
        for j in 1:length(λ)
            # p  = sum(r.φ ./ (λ[j] .- r.poles))
            # q  = sum(r.ψ ./ (λ[j] .- r.poles))
            # dp = r.φ ./ ((λ[j] .- r.poles).^2)
            # dq = r.ψ ./ ((λ[j] .- r.poles).^2) 
            
            # g2[j,:] = l[j].*(dp ./(1+q) .- (p/((1+q)^2) .* dq))
            
            # dA[j,1:length(r.φ)] = -1.0./((λ[j] .- r.poles).^2)
            # dA[j,length(r.φ)+1:end] = Y[j]./((λ[j] .- r.poles).^2) 

            dA_dξ_x = (1.0 / ((λ[j] - ξ[i])^2)) * (φ[i] - Y[j]*ψ[i])
            g = g + dA_dξ_x' * v[j]
        end
        grad[i] = g
    end


    return φ,ψ, grad, real(loss)
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
        weights[2:end] = diff(test_vec)
        weights[1]= weights[2]
    end


    #loss = sum(weights .* abs2.(l)) + sum(abs2.(r.ψ))
    loss = sum(weights .* abs2.(l))
    return loss
end

"""
Build Least squares differential for updating xi
"""
function grad_L(r::VFit,f_df::DataFrame,l::Vector=Float64[],weights::Vector = Float64[])
    f_df = f_df[sortperm(f_df[:,1]),:]

    if isempty(l)
        l = rapprox.((r,),f_df[:,1]) .- f_df[:,2]
    end

    test_vec = f_df[:,1]
    if isempty(weights)
        weights = zeros(Float64,length(test_vec))
        weights[2:end] = diff(test_vec)
        weights[1]= weights[2]
    end

    l = sqrt.(weights) .* l
    #dAxv = zeros(ComplexF64,length(l),length(r.poles))
    #x = vcat(r.φ,r.ψ)
    #g2 = zeros(ComplexF64,length(l),length(r.poles))
    grad = zeros(ComplexF64,length(r.poles))
    λ = f_df[:,1]
    Y = f_df[:,2]
    

    for i in 1:length(r.poles)
        g = 0.0
        for j in 1:length(l)
            # p  = sum(r.φ ./ (λ[j] .- r.poles))
            # q  = sum(r.ψ ./ (λ[j] .- r.poles))
            # dp = r.φ ./ ((λ[j] .- r.poles).^2)
            # dq = r.ψ ./ ((λ[j] .- r.poles).^2) 
            
            # g2[j,:] = l[j].*(dp ./(1+q) .- (p/((1+q)^2) .* dq))
            
            # dA[j,1:length(r.φ)] = -1.0./((λ[j] .- r.poles).^2)
            # dA[j,length(r.φ)+1:end] = Y[j]./((λ[j] .- r.poles).^2) 

            dA_dξ_x = (1.0 / ((λ[j] - r.poles[i])^2)) * (r.φ[i] - Y[j]*r.ψ[i])
            g = g + dA_dξ_x' * l[j]
        end
        grad[i] = g
    end
    
    return grad
end


"""
Build least squares with Gradient Descent for ξ
"""
function GD_pole_update(f_df,ξ,w::Vector=Float64[],α=1.0,tol=1e-10)

    F_vec = f_df[:,2]
    λ = f_df[:,1]

    ∇L= zeros(ComplexF64,length(ξ))
    prev_ξ = zeros(ComplexF64,length(ξ))
    max_count = 500
    cnt = 0

    φ,ψ,gl,loss = update_φ_ψ(F_vec,λ,ξ,w,α)
    r = VFit(φ,ψ,ξ)

    Err_vec = []
    Loss_vec = []
    Gammas = []
    err = loss_calc(r,f_df,w)
    push!(Err_vec,err)
    #Update poles via GD
    while (err>=tol) && (cnt<=max_count)
        print("At iteration ",cnt)
        print(" The Squared Error is ",err,"\n")
        r.φ, r.ψ, gl, loss = update_φ_ψ(F_vec,λ,r.poles,w,α)
        print(" The Squared Loss is ",loss,"\n")
        push!(Loss_vec,loss)
        err = loss_calc(r,f_df,w)
        # gl = grad_L(r,f_df,loss,w)
        γ = dot(r.poles .- prev_ξ, gl .- ∇L)./dot(gl .- ∇L, gl .- ∇L)
        #γ = 1e-2
        ∇L .= gl
        prev_ξ .= r.poles
        r.poles .= r.poles .- γ.*∇L
        #r.φ, r.ψ = update_φ_ψ(F_vec,λ,r.poles,w)
        cnt+=1
        push!(Err_vec,err)
        push!(Gammas,abs(γ))
    end 
    
    r.φ, r.ψ, gl, loss = update_φ_ψ(F_vec,λ,r.poles,w,α)
    push!(Err_vec,loss_calc(r,f_df,w))
    push!(Loss_vec,loss)

    
    #PyPlot.show()
    print("\n The alpha value is $(α) \n")
    gd_error_plot(Err_vec,length(ξ),length(λ),cnt,α)
    gd_loss_plot(Loss_vec,length(ξ),length(λ),cnt,Gammas,α)
    return r,Err_vec,Loss_vec
end




end 


# #    print("\n",math2Expr(Dfactored),"\n")
#     # filter!(x->x!=1,roots)                                                  #Mathematica outputs have a lot of 1's as coefficients of x that seep through. This filters them out
#     # if length(roots)==length(r_.poles)+1                                    #Sometimes there is a multiplyer to the whole factored polynomial, this removes the multiplier which appears in the beginning of the expression
#     #     roots = roots[2:end]
#     # end
#     #@assert(length(roots)==length(r_.poles))
    
#     if length(roots)==length(r_.poles)
#         (return (-1.0)*roots)
#     else
#         print(math2Expr(Dfactored),"\n")
#         print(roots,"\n")
#         print("Quitting and resending old roots as Factor[] reduced the order")
#         #throw(DomainError(roots, "roots should be of length $(length(r_.poles))"))
#         return r_.poles
#     #return (-1.0)*roots
#     end 
#     return (-1.0)*roots