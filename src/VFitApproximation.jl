module VFitApproximation

# Write your package code here.

using BaryRational
using Random
using Distributions
using LinearAlgebra
using PyPlot
using MathLink

import StatsBase

include("MathematicaPipelining.jl")
include("PlottingFuncs.jl")

export VFit, rapprox

mutable struct VFit
    φ::AbstractVector
    ψ::AbstractVector
    poles::AbstractVector
end

function rapprox(self::VFit,x::Float64)
    p = self.φ ./ (x .- self.poles)
    q = self.ψ ./ (x .- self.poles)
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
    C1 = zeros(Complex64,length(lambda),length(xi))
    C2 = zeros(Complex64,length(lambda),length(xi))
    Y = f.(l)
    for j in 1:length(l)                                        #Build the initial matrix in RKFIT paper eq (A.4) to be solved
        C1[j,:] = 1 ./(l[j] .- xi)
        C2[j,:] = -Y[j] ./(l[j] .- xi)
    end
    C = hcat(C1,C2)

    weights = zeros(Float64,length(l))                          #If there are a lot of samples near a singularity then they will dominate the error, these weights help reweight the uneven sampling
    weights[2:end] = abs.(l[2:end] - l[1:end-1])
    weights[1]= abs(l[1])
    W = Diagonal(sqrt.(weights))
    A = W*C
    P = A\(W*Y)
    #P = C\Y                                                    #Solve Cx=y matrix equation and get th phi's and psi's
    return P[1:m], P[m+1:2m]

end

function get_phi_psi(f_vec::Vector, l::Vector,xi::AbstractVector)
    m = length(xi)
    C1 = zeros(Complex64,length(l),length(xi))
    C2 = zeros(Complex64,length(l),length(xi))
    Y = f_vec
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
    P = A\(W*Y)
    #P = C\Y                                                    #Solve Cx=y matrix equation and get th phi's and psi's
    return P[1:m], P[m+1:2m]

end


"""
This function returns the roots and poles of q(x), the Barycentric denominator by pipelining it through Mathematica
Inputs r_ = The rational function at the current step
Outputs   = Roots of the barycentric denominator
"""
function Find_roots_using_Mathematica(r_ :: VFit)
    Darrexp  = build_denom.(r_.ψ,r_.poles)                                  #Write rational function denominator in correct form in MathLink expression
    Dexp = weval(addemup(Darrexp,length(Darrexp)))                          #Evaluate the built expression in MathLink form
    #Dfactored = weval(W"Factor"(Dexp))                                      #Run Factor from Mathematica to turn expression into (x-z_1)(x-z_2)....(x-z_m)/(x-p_1)(x-p_2)...(x-p_m)
    Dfactored = weval(W"Reduce"(W"Equal"(Dexp,0),W"x"))                     #Run Reduce[] from Mathematica to get output x=z_1 || x=z_2 ... etc 
    roots = ExtractExpr(Dfactored,Complex[])                                #Extract the roots (z_1,z_2,...z_m) from the denominator polynomial q(x)
    if any(isnan,roots)
        Dfactored = weval(W"Reduce"(W"Equal"(Dexp,0),W"x"))                 #In the tricky case where Reduce does not provide roots we run Roots function hoping that it would
        roots = ExtractExpr(Dfactored,Complex[])
        if any(isnan,roots)
            print("\n\nNo roots can be found! Exiting with same poles\n\n")
            return r_.poles
        end
    end

    return roots 
end


"""
Returns the square error of a rational function approximation on a random test sample.
Inputs r_ = The rational function, f:= the function to appoximate
Outputs Least square error
"""
function get_sqrerr(r::VFit, f::Function, test::Vector{Float64})
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
function get_sqrerr(r::VFit, f_testdf::Matrix)
    f_testdf = f_testdf[sortperm(f_testdf[:,1]),:]
    residue = rapprox.((r,),f_testdf[:,1]) .- (f_testdf[:,2])
    
    test_vec = f_testdf[:,1]
    weights = zeros(Float64,length(test_vec))
    weights[2:end] = abs.(test_vec[2:end] - test_vec[1:end-1])
    weights[1]= weights[2]

    
    return sum(weights .* ((abs.(residue)).^2))
end


"""
The main part of the program that combines all the functions and runs the VFit Algorithm
Inputs  f:=   The function to approximate
        m:=   The degree of the polynomials p(x) and q(x) that make up the rational function
        ξ:=   The initial guess of starting support points for the barycentric forms
        λ:=   The training sample points over which the VFit approximation will be made
        tol:= The tolerance of the least square error, defaults to 1e-10

Outputs r     := The rational function
        errors:= The training and testing errors faced
"""
function vfitting(f::Function, m::Int, ξ::Vector{Float64}, λ::Vector{Float64}, tol::Float64 =1e-10)

    cnt = 1                                                                 #Initialization of the count of iteration
    phi,psi = get_phi_psi(f,λ,ξ)                                            #Get the first φ and ψ
    r = VFit(phi,psi,ξ)                                                     #Initialize the rational approximation r

    num= length(λ)
    inps = sort(λ)
    test_λ = collect(range(inps[1],inps[end],length=5000))

    sqres =  get_sqrerr(r, f, test_λ)
    self_err = get_sqrerr(r,f,λ)
    trainerrarr = Float64[]
    testerrarr = Float64[]
    print("The initial square error is: $(sqres)\n")
    while (self_err>tol || sqres>10*tol) && cnt<51                          #Convergence criteria of 10 used-- if iteration is >10 times it is probably stuck in some local minima
        print("At iteration ",cnt)
        sqres = get_sqrerr(r,f,test_λ)
        self_err = get_sqrerr(r,f,λ)
        print(" The self error is: ",self_err,"\n")
        print(" The square error is: ",sqres,"\n")
        push!(testerrarr,sqres)
        push!(trainerrarr,self_err)
        r.poles = Find_roots_using_Mathematica(r)                           #Update the poles of the rational function
        r.φ, r.ψ = get_phi_psi(f,λ,r.poles)                                 #Get the next φ and ψ and update the rational function

        cnt=cnt+1
    end
    errors = plot_iters(trainerrarr,testerrarr,m,num)   #Plots the training and testing errors incurred at

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

Outputs r     := The rational function
        errors:= The training and testing errors faced
"""
function vfitting(f_df::Matrix, m::Int, ξ::Vector{Float64}, tol::Float64 =1e-10)
    cnt = 1                                                                 #Initialization of the count of iteration
    
    #Split Training and testing into 80 percent train and 20 percent test
    # ixs = randperm(length(f_df[:,1]))
    # tr_num = Int(floor(0.8*length(ixs)))
    # f_train = f_df[ixs[1:tr_num],:]
    # f_test = f_df[ixs[tr_num+1:end],:]
    
    #Odd-Even splitting
    f_df = f_df[sortperm(f_df[:,1]),:]
    f_train = f_df[2:2:length(f_df[:,1]),:]                                 #extracts all the even elements out
    f_test = f_df[1:2:length(f_df[:,1]),:]

    #All points
    # f_train = f_df
    # f_test = f_df

    f_train = f_train[sortperm(f_train[:,1]),:]                              #Sort the data
    f_vec = f_train[:,2]
    λ = f_train[:,1]



    #Start the training
    phi,psi = get_phi_psi(f_vec,λ,ξ)                                     #Get the first φ and ψ
    r = VFit(phi,psi,ξ)                                                      #Initialize the rational approximation r

    num= length(λ)
    test_sqres =  get_sqrerr(r,f_test)
    self_err = get_sqrerr(r,f_train)
    trainerrarr = Float64[]
    testerrarr = Float64[]
    print("The initial square error is: $(test_sqres)\n")
    while (test_sqres>tol) && (cnt<51)                                             #Convergence criteria of 50 used-- if iteration is >50 times it is probably stuck in some local minima
        print("At iteration ",cnt)
        test_sqres = get_sqrerr(r,f_test)
        self_err = get_sqrerr(r,f_train)
        print(" The self error is: ",self_err,"\n")
        print(" The square error is: ",test_sqres,"\n")
        push!(testerrarr,test_sqres)
        push!(trainerrarr,self_err)
        r.poles = Find_roots_using_Mathematica(r)                           #Update the poles of the rational function
        r.φ, r.ψ = get_phi_psi(f_vec,λ,r.poles)                             #Get the next φ and ψ and update the rational function

        cnt=cnt+1
    end
    errors = plot_iters(trainerrarr,testerrarr,m,num)                       #Plots the training and testing errors incurred at

    return r,errors

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