using MathLink

"""
This function creates the main denominator q(x) in MathLink expression for the VFit rational approximation.
Inputs  psi weights,
        z support points
Outputs MathLink.WExpr The input expression of the denominator q(x) to Mathematica that is written in barycentric form
"""
function build_denom(psi,z)
    return W"Times"(psi,W"Power"(W"Plus"(-z,W"x"),-1))
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
This function is for extracting the factors/roots which Mathematica provides

"""
function ExtractExpr(num::Number,P::AbstractVector,Q::AbstractVector,powflag)
    if powflag==1
         push!(Q,Float64(num))
    else
         push!(P,Float64(num))
    end
    #print(num)
end

function ExtractExpr(symb::MathLink.WSymbol,P::AbstractVector,Q::AbstractVector,powflag)
    #print(Symbol(symb.name))
end
"""
This function checks each condition in the output Mathematica syntax tree and finds the numbers in the numerator and denominator
and places them into two vectors P and Q indicating the roots of p(x) and q(x) where the rational function is r = p/q
"""
function ExtractExpr(expr::MathLink.WExpr,P::AbstractVector,Q::AbstractVector,powflag)
    if expr.head.name=="Times"
        ExtractExpr.(expr.args,(P,),(Q,),powflag)
    elseif expr.head.name=="Plus"
        ExtractExpr.(expr.args,(P,),(Q,),powflag)
    elseif expr.head.name=="Power"
        ExtractExpr.(expr.args,(P,),(Q,),1)
    else
    end
    return P,Q
end


"""
Taken from UsingMathLink.jl by Github user gangchern
`https://github.com/gangchern/usingMathLink`
I used the parts that were needed for me.
These set of functions converts the Mathematica syntax tree to a Julia expression tree, thus making it easier to read.
"""
function math2Expr(symb::MathLink.WSymbol)
    Symbol(symb.name)
end
function math2Expr(num::Number)
    num
end
function math2Expr(expr::MathLink.WExpr)
    if expr.head.name=="Times"
        return Expr(:call, :*, map(math2Expr,expr.args)...)
    elseif expr.head.name=="Plus"
        return Expr(:call, :+,map(math2Expr,expr.args)...)
    elseif expr.head.name=="Power"
        return Expr(:call, :^, map(math2Expr,expr.args)...)
    elseif expr.head.name=="Rational"
        return  Expr(:call, ://, map(math2Expr,expr.args)...)
    else
        return Expr(:call, Symbol(expr.head.name), map(math2Expr,expr.args)...)
    end
end
