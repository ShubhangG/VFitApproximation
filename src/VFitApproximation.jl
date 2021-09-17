module VFitApproximation

# Write your package code here.

using BaryRational
using Random
using Distributions
using LinearAlgebra

include("MathematicaPipelining.jl")
include("PlottingFuncs.jl")


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

```
Function that samples points from an exponential distribution
Inputs: theta of exponential distribution,
        num is number of points
Output: Returns num number of points sampled from an exponential distribution in (0,1)
```
function create_sample_points(theta::Float64,num::Int)
    dist = Exponential(theta)
    pts = rand(dist,num)
    while any(x->x>=1,pts)
        pts[pts .>=1] =  rand(dist, length(pts[pts .>= 1]))
    end
    return pts
end

```
Function that samples points from a mixed distribution--- an exponential distribution (0,1) and a Uniform distribution [0.2,1)
Input:  θ of the exponential distribution (θ*exp[-θ*x]),
        prior for the mixed distribution and
        num = number of points to sample
Output: Returns num number of points sampled from the mixed distribution with prior
```
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

```
This function lets us obtain AAA approximation, if one chooses to start with an initial guess for poles as the AAA support points
Inputs: f: function we are approximating,
        mm: number of support points

Output: Returns a AAA approximation of the function.
```
function firstAAA_step(f::Function,mm::Int)
    Random.seed!(427)
    Z = create_sample_points(0.14,10000)
    aaa_out = aaa(Z,f.(Z),mmax=mm)
    return aaa_out
end

```
This function is used to calculate the barycentric weights by running a least square approximation to a matrix equation
given by equation (A.4) in the RKFit paper
Inputs  f:      the function that is being approximated,
        lambda: the sample points,
        xi:     the barycentric poles of p(x) and q(x)

Outputs φ: The barycentric weights of the numerator p(x)
        ψ: The barycentric weights of the denominator q(x)
```
function get_phi_psi(f::Function,lambda::AbstractVector,xi::AbstractVector)
    m = length(xi)
    l = sort(lambda)
    #l=lambda
    C1 = zeros(Float64,length(lambda),length(xi))
    C2 = zeros(Float64,length(lambda),length(xi))
    Y = f.(l)
    for j in 1:length(l)                                        #Build the initial matrix in RKFIT paper eq (A.4) to be solved
        C1[j,:] = 1 ./(l[j] .- xi)
        C2[j,:] = -Y[j] ./(l[j] .- xi)
    end
    C = hcat(C1,C2)

    weights = zeros(Float64,length(l))                          #If there are a lot of samples near a singularity then they will dominate the error, these weights help reweight the uneven sampling
    weights[2:end] = l[2:end] - l[1:end-1]
    weights[1]= l[1]
    W = Diagonal(sqrt.(weights))
    A = W*C
    P = A\(W*Y)
    #P = C\Y                                                    #Solve Cx=y matrix equation and get th phi's and psi's
    return P[1:m], P[m+1:2m]

end

```
This function returns the roots and poles of q(x), the Barycentric denominator by pipelining it through Mathematica
Inputs r_ := The rational function at the current step
Outputs := Roots of the barycentric denominator
```
function Find_roots_using_Mathematica(r_ :: VFit)
    Darrexp  = build_denom.(r_.ψ,r_.poles)                                  #Write rational function denominator in correct form in MathLink expression
    Dexp = weval(addemup(Darrexp,length(Darrexp)))                          #Evaluate the built expression in MathLink form
    Dfactored = weval(W"Factor"(Dexp))                                      #Run Factor from Mathematica to turn expression into (x-z_1)(x-z_2)....(x-z_m)/(x-p_1)(x-p_2)...(x-p_m)
    roots, poles = ExtractExpr(Dfactored,Float64[],Float64[],0)             #Extract the roots (z_1,z_2,...z_m) from the denominator polynomial q(x)
    filter!(x->x!=1,roots)                                                  #Mathematica outputs have a lot of 1's as coefficients of x that seep through. This filters them out
    if length(roots)==length(r_.poles)+1                                    #Sometimes there is a multiplyer to the whole factored polynomial, this removes the multiplier which appears in the beginning of the expression
        roots = roots[2:end]
    end
    #@assert(length(roots)==length(r_.poles))
    (length(roots)==length(r_.poles)) ? (return (-1.0)*roots) : begin print(math2exp(Dfactored),"\n"); print(roots,"\n"); print("Quitting"); throw(DomainError(roots, "roots should be of length $(length(r_.poles))")) end
    return (-1.0)*roots
end


```
Returns the square error of a rational function approximation on a test sample.
Inputs r_:= The rational function, f:= the function to appoximate
Outputs Least square error
```
function get_sqrerr(r::VFit, f::Function)
    #test = rand(Uniform(0.01,1),500)
    Random.seed!(1234)                                      #This seed prevents the test dataset from changing
    #test = sort(rand(Uniform(0.01,1),500))
    test = collect(range(0.01,1,length=5000))
    weights = zeros(Float64,length(test))
    weights[2:end] = test[2:end] - test[1:end-1]
    weights[1]= test[1]

    residue = rapprox.((r,),test) .- f.(test)
    return sum(weights .* (residue.^2))
end

```
The main part of the program that combines all the functions and runs the VFit Algorithm
Inputs  f:=   The function to approximate
        m:=   The degree of the polynomials p(x) and q(x) that make up the rational function
        ξ:=   The initial guess of starting support points for the barycentric forms
        λ:=   The training sample points over which the VFit approximation will be made
        tol:= The tolerance of the least square error, defaults to 1e-10

Outputs r     := The rational function
        errors:= The training and testing errors faced
```
function vfitting(f::Function, m::Int, ξ::AbstractArray, λ::AbstractArray, tol::Float64 =1e-10)

    cnt = 1                                                                 #Initialization of the count of iteration
    phi,psi = get_phi_psi(f,λ,ξ)                                            #Get the first φ and ψ
    r = VFit(phi,psi,ξ)                                                     #Initialize the rational approximation r

    num= length(λ)
    sqres =  get_sqrerr(r, f)
    weights = zeros(Float64,length(λ))
    weights[2:end] = λ[2:end] - λ[1:end-1]
    weights[1]= λ[1]
    self_err = sum(weights .* ((rapprox.((r,), λ) .- f.(λ)).^2))
    trainerrarr = Float64[]
    testerrarr = Float64[]
    print("The initial square error is: $(sqres)\n")
    while (self_err>tol || sqres>10*tol) && cnt<51                          #Convergence criteria of 10 used-- if iteration is >10 times it is probably stuck in some local minima
        print("At iteration ",cnt)
        sqres = get_sqrerr(r,f)
        self_err = sum((rapprox.((r,), λ) .- f.(λ)).^2)
        print(" The self error is: ",self_err,"\n")
        print(" The square error is: ",sqres,"\n")
        push!(testerrarr,sqres)
        push!(trainerrarr,self_err)
        r.poles = Find_roots_using_Mathematica(r)                           #Update the poles of the rational function
        r.φ, r.ψ = get_phi_psi(f,λ,r.poles)                                 #Get the next φ and ψ and update the rational function

        cnt=cnt+1
    end
    errors = plot_iters(trainerrarr,testerrarr,m,num,guessxi,guesslambda)   #Plots the training and testing errors incurred at

    return r,errors

end



end
