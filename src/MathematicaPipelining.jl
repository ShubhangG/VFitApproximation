"""
This function creates the main denominator q(x) in MathLink expression for the VFit rational approximation.
Inputs  psi weights,
        z support points
Outputs MathLink.WExpr The input expression of the denominator q(x) to Mathematica that is written in barycentric form
"""
function build_denom(psi,z)
    # if imag(z)!=0
    #     expz = W"Complex"(real(z),imag(z))
    # else
    #     expz = real(z)
    # end

    # if imag(psi)!=0
    #     exp_ps = W"Complex"(real(psi),imag(psi))
    # else
    #     exp_ps = real(psi) 
    # end 
    expz = W"Complex"(real(z),imag(z))
    exp_ps = W"Complex"(real(psi),imag(psi))
    return W"Times"(exp_ps,W"Power"(W"Plus"(W"Times"(expz,-1),W"x"),-1))
end

"""
This is a recursive function that comes after running build_denom on all the support points thus building each term of the denominator q(x).
This sums it all up into one nice Mathematica expression which we shall use Mathematica to get its roots.

Inputs: wexpr: Array of Mathematica expressions, idx: index number of the array, starts from length(wexpr) to 1
Outputs: Combined expression represented as a sum
"""
function addemup(wexpr,idx)
    if idx==0
        return 1
    else
        return W"Plus"(wexpr[idx],addemup(wexpr,idx-1))
    end
end
"""
After using Mathematica to do the root finding, these set of recursive functions extract the roots from the syntax tree.
These roots are passed into a Vector of complex numbers P that is returned as the output to Julia 
"""

function ExtractExpr(num::Number,P::AbstractVector)
    push!(P,num)
end

function ExtractExpr(symb::MathLink.WSymbol,P::AbstractVector)
    #print(Symbol(symb.name))
end

function ExtractExpr(cmplxvec::Vector{Float64}, P::AbstractVector)
    push!(P,complex(cmplxvec[1],cmplxvec[2]))
end 

"""
This function extracts the numbers from the output of the Mathematica syntax tree and finds the numbers indicating the roots of q(x)  
"""
function ExtractExpr(expr::MathLink.WExpr,P::AbstractVector)
    if expr.head.name=="Equal"
        ExtractExpr.(expr.args,(P,))
    elseif expr.head.name=="Complex"
        ExtractExpr(expr.args,P)
    elseif expr.head.name=="Or"
        ExtractExpr.(expr.args,(P,))
    elseif expr.head.name=="Roots" || expr.head.name=="Root"
        print("Can't find roots! Try again")
        push!(P,NaN)
        return P
    else
        push!(P,eval(math2Expr(expr)))

    end
    return P
end

"""
A function that computes the output provided by mathematica if it is not in numeric form but still contains symbols and rationals.
These set of functions do essentially what eval(math2Expr(WExpr)) does. 

"""
function ComputeExpr(expr::MathLink.WExpr)
    if expr.head.name=="Times"
        return prod(map(ComputeExpr,expr.args))
    elseif expr.head.name=="Plus"
        return sum(map(ComputeExpr,expr.args))
    elseif expr.head.name=="Power"
        return ComputeExpr(expr.args[1])^(ComputeExpr(expr.args[2]))
    elseif expr.head.name=="Rational"
        return expr.args[1]/expr.args[2]
    elseif expr.head.name=="Complex"
        return complex(expr.args[1],expr.args[2])
    end
end

function ComputeExpr(num::Number)
    return num
end 

function ComputeExpr(symb::MathLink.WSymbol)
    print("The symbol $(Symbol(symb.name)) shouldn't exist here!!")
    exit()
end

function math2Expr(symb::MathLink.WSymbol)
    Symbol(symb.name)
end

function math2Expr(num::Number)
    num
end

function math2Expr(cmplx::Vector)
    Meta.parse("$(complex(cmplx[1],cmplx[2]))")
end
"""
Inspired from UsingMathLink.jl by Github user gangchern
# `https://github.com/gangchern/usingMathLink`
I changed it to suite my needs for extracting roots 

"""
function math2Expr(expr::MathLink.WExpr)
    if expr.head.name=="Times"
        return Expr(:call, :*, map(math2Expr,expr.args)...)
    elseif expr.head.name=="Plus"
        return Expr(:call, :+,map(math2Expr,expr.args)...)
    elseif expr.head.name=="Power"
        return Expr(:call, :^, map(math2Expr,expr.args)...)
    elseif expr.head.name=="Rational"
        return  Expr(:call, ://, map(math2Expr,expr.args)...)
    elseif expr.head.name=="Equal"
        return Expr(:call, Symbol("="), map(math2Expr, expr.args)...)
    elseif expr.head.name=="Complex"
        return math2Expr(expr.args)
    elseif expr.head.name=="Or"
        return map(math2Expr,expr.args)
    else
        return Expr(:call, Symbol(expr.head.name), map(math2Expr,expr.args)...)
    end
end

# """
# This function is for extracting the factors/roots which Mathematica provides from Factor[]

# """
# function ExtractExpr(num::Number,P::AbstractVector,Q::AbstractVector,powflag)
#     if powflag==1
#          push!(Q,Float64(num))
#     else
#          push!(P,Float64(num))
#     end
#     #print(num)
# end

# function ExtractExpr(symb::MathLink.WSymbol,P::AbstractVector,Q::AbstractVector,powflag)
#     #print(Symbol(symb.name))
# end
# """
# This function checks each condition in the output Mathematica syntax tree and finds the numbers in the numerator and denominator
# and places them into two vectors P and Q indicating the roots of p(x) and q(x) where the rational function is r = p/q
# """
# function ExtractExpr(expr::MathLink.WExpr,P::AbstractVector,Q::AbstractVector,powflag)
#     if expr.head.name=="Times"
#         ExtractExpr.(expr.args,(P,),(Q,),powflag)
#     elseif expr.head.name=="Plus"
#         ExtractExpr.(expr.args,(P,),(Q,),powflag)
#     elseif expr.head.name=="Power"
#         ExtractExpr.(expr.args,(P,),(Q,),1)
#     end
#     return P,Q
# end


# """
# Taken from UsingMathLink.jl by Github user gangchern
# `https://github.com/gangchern/usingMathLink`
# I used the parts that were needed for me.
# These set of functions converts the Mathematica syntax tree to a Julia expression tree, thus making it easier to read.
# """
# function math2Expr(symb::MathLink.WSymbol)
#     Symbol(symb.name)
# end
# function math2Expr(num::Number)
#     num
# end
# function math2Expr(expr::MathLink.WExpr)
#     if expr.head.name=="Times"
#         return Expr(:call, :*, map(math2Expr,expr.args)...)
#     elseif expr.head.name=="Plus"
#         return Expr(:call, :+,map(math2Expr,expr.args)...)
#     elseif expr.head.name=="Power"
#         return Expr(:call, :^, map(math2Expr,expr.args)...)
#     elseif expr.head.name=="Rational"
#         return  Expr(:call, ://, map(math2Expr,expr.args)...)
#     else
#         return Expr(:call, Symbol(expr.head.name), map(math2Expr,expr.args)...)
#     end
# end
