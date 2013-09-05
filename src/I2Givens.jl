type CivecmI2Givens <: AbstractCivecm
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	rank::(Int64,Int64)
	pars::Vector{Float64}
	llConvCrit::Float64
	maxiter::Int64
	Z0::Matrix{Float64}
	Z1::Matrix{Float64}
	Z2::Matrix{Float64}
	Z3::Matrix{Float64}
	R0::Matrix{Float64}
	R1::Matrix{Float64}
	R2::Matrix{Float64}
end

function civecmI2Givens(endogenous::Matrix{Float64}, exogenous::Matrix{Float64}, lags::Int64)
	mDX = diff(endogenous)
	mDDX = diff(diff(endogenous))
	mLDDX = lagmatrix(mDDX, 1:lags - 2)
	mDU = diff(exogenous)
	mDDU = diff(diff(exogenous))
	mLDDU = lagmatrix(mDDU, 0:lags - 2)
	
	Z0 = mDDX[lags - 1:, :]
	Z1 = [mDX[lags - 1:end - 1, :] mDU[lags - 1:end - 1, :]]
	Z2 = [endogenous[lags:end - 1, :] exogenous[lags:end - 1, :]]
	Z3 = copy(mLDDX)
	if size(mLDDU, 1) > 0
	    Z3 = [Z3 mLDDU]
	end
	if size(Z3, 2) > 0
		R0 = mreg(Z0, Z3)[2]
		R1 = mreg(Z1, Z3)[2]
		R2 = mreg(Z2, Z3)[2]
	else
		R0 = Z0
		R1 = Z1
		R2 = Z2
	end

	return CivecmI2Givens(endogenous, float64(exogenous), lags, (size(endogenous, 2), 0), Array(Float64, npars(size(R1, 2), size(R0, 2), size(R0, 2), 0)), 1.0e-8, 50000, Z0, Z1, Z2, Z3, R0, R1, R2)
end
civecmI2Givens(endogenous::Matrix{Float64}, lags::Int64) = civecmI2Givens(endogenous, zeros(size(endogenous, 1), 0), lags)
civecmI2Givens(endogenous::Matrix{Float64}, exogenous::Range1, lags::Int64) = civecmI2Givens(endogenous, float64(reshape(exogenous, length(exogenous), 1)), lags)

function npars(p1::Integer,p::Integer,r::Integer,s::Integer)
	n = div(p*(p - 1),2) - div((p - r)*(p - r - 1),2) +
		div(p1*(p1 - 1),2) - div((p1 - r)*(p1 - r - 1),2) +
		r +
        p*r + r*(p1 - r) +
        div((p - r)*(p - r - 1),2) - div((p - r - s)*(p - r - s - 1),2) +
        div((p1 - r)*(p1 - r - 1),2) - div((p1 - r - s)*(p1 - r - s - 1),2) +
        s
    return n
end
npars(obj::CivecmI2Givens) = npars(size(obj.Z1, 2), size(obj.Z0, 2), obj.rank[1], obj.rank[2])

function setrank(obj::CivecmI2Givens, rank::(Int64, Int64))
	if sum(rank) > size(obj.endogenous, 2)
		error("Illegal choice of rank")
	else
		obj.rank = rank
	end
	return obj
end

residuals(obj::CivecmI2Givens) = obj.R0 - obj.R2*β(obj)*σ_α_β(obj)*α(obj)' - obj.R1*Γ(obj)'

function α(obj::CivecmI2Givens, full::Bool)
    p = size(obj.Z0, 2)
    count = 1
   	ans = eye(p)
    for i = 1:obj.rank[1]
        for j = i + 1:p
            Q = eye(p)
            Q[i, i] = cos(obj.pars[count])
			Q[j, j] = Q[i, i]
            Q[i, j] = sin(obj.pars[count])
            Q[j, i] = -Q[i, j]
            count = count + 1
            ans = Q*ans
        end
    end
    return (full ? ans : ans[:,1:obj.rank[1]])
end
α(obj::CivecmI2Givens) = α(obj, false)

function β(obj::CivecmI2Givens, full::Bool)
    p = size(obj.Z0, 2)
    p1 = size(obj.Z1, 2)
    count = div(p*(p-1),2) - div((p-obj.rank[1])*(p-obj.rank[1]-1),2) + 1
    ans = eye(p1)
    for i = 1:obj.rank[1]
        for j = i + 1:p1
            Q = eye(p1)
            Q[i, i] = cos(obj.pars[count])
            Q[j, j] = Q[i, i]
            Q[i, j] = sin(obj.pars[count])
            Q[j, i] = -Q[i, j]
            count = count + 1
            ans = Q*ans
        end
    end
    return (full ? ans : ans[:,1:obj.rank[1]])
end
β(obj::CivecmI2Givens) = β(obj, false)

function σ_α_β(obj::CivecmI2Givens)
	p = size(obj.Z0, 2)
	p1 = size(obj.Z1, 2)
	count = div(p*(p-1),2)-div((p-obj.rank[1])*(p-obj.rank[1]-1),2) + 
		div(p1*(p1-1),2)-div((p1-obj.rank[1])*(p1-obj.rank[1]-1),2) + 1
	return diagm(obj.pars[count:count + obj.rank[1] - 1])
end

function Γ(obj::CivecmI2Givens)
	p = size(obj.Z0, 2)
	p1 = size(obj.Z1, 2)
	count = div(p*(p-1),2)-div((p-obj.rank[1])*(p-obj.rank[1]-1),2) + 
		div(p1*(p1-1),2)-div((p1-obj.rank[1])*(p1-obj.rank[1]-1),2) + 
		obj.rank[1] + 1
	ans = Array(Float64, p, p1)
	ans[:,1:obj.rank[1]] = obj.pars[count:count + p*obj.rank[1] - 1]
	ans[1:obj.rank[1],obj.rank[1] + 1:end] = obj.pars[count + p*obj.rank[1]:count + p*obj.rank[1] + (p - obj.rank[1])*obj.rank[1] - 1]
	ans[obj.rank[1] + 1:,obj.rank[1] + 1:] = ξ(obj)*σ_ξ_η(obj)*ηpar(obj)'
	return α(obj, true)*ans*β(obj, true)'
end

function ξ(obj::CivecmI2Givens)
	p = size(obj.Z0, 2)
	p1 = size(obj.Z1, 2)
	count = div(p*(p-1),2)-div((p-obj.rank[1])*(p-obj.rank[1]-1),2) + 
		div(p1*(p1-1),2)-div((p1-obj.rank[1])*(p1-obj.rank[1]-1),2) + 
		obj.rank[1] + 
		p*obj.rank[1] + (p - obj.rank[1])*obj.rank[1] + 1
	ans = eye(p - obj.rank[1], obj.rank[2])
	for i = 1:obj.rank[2]
	    for j = i + 1:p - obj.rank[1]
	        Q = eye(p - obj.rank[1])
	        Q[i, i] = cos(obj.pars[count])
	        Q[j, j] = Q[i, i]
	        Q[i, j] = sin(obj.pars[count])
	        Q[j, i] = -Q[i, j]
	        count = count + 1
	        ans = Q*ans
	    end
	end
	return ans
end

function ηpar(obj::CivecmI2Givens)
	p = size(obj.Z0, 2)
	p1 = size(obj.Z1, 2)
	count = div(p*(p-1),2)-div((p-obj.rank[1])*(p-obj.rank[1]-1),2) + 
		div(p1*(p1-1),2)-div((p1-obj.rank[1])*(p1-obj.rank[1]-1),2) + 
		obj.rank[1] + 
		p*obj.rank[1] + (p - obj.rank[1])*obj.rank[1] + 
		div((p - obj.rank[1])*(p-obj.rank[1] - 1), 2) - div((p - sum(obj.rank))*(p - sum(obj.rank) - 1), 2) + 1
	ans = eye(p1 - obj.rank[1], obj.rank[2])
	for i = 1:obj.rank[2]
	    for j = i + 1:p1 - obj.rank[1]
	        Q = eye(p1 - obj.rank[1])
	        Q[i, i] = cos(obj.pars[count])
	        Q[j, j] = Q[i, i]
	        Q[i, j] = sin(obj.pars[count])
	        Q[j, i] = -Q[i, j]
	        count = count + 1
	        ans = Q*ans
	    end
	end
	return ans
end

function σ_ξ_η(obj::CivecmI2Givens)
	p = size(obj.Z0, 2)
	p1 = size(obj.Z1, 2)
	count = div(p*(p-1),2)-div((p-obj.rank[1])*(p-obj.rank[1]-1),2) + 
		div(p1*(p1-1),2)-div((p1-obj.rank[1])*(p1-obj.rank[1]-1),2) + 
		obj.rank[1] + 
		p*obj.rank[1] + (p - obj.rank[1])*obj.rank[1] + 
		div((p - obj.rank[1])*(p-obj.rank[1] - 1), 2) - div((p - sum(obj.rank))*(p - sum(obj.rank) - 1), 2) + 
		div((p1 - obj.rank[1])*(p1-obj.rank[1] - 1), 2) - div((p1 - sum(obj.rank))*(p1 - sum(obj.rank) - 1), 2) + 1
	return diagm(obj.pars[count:count + obj.rank[2] - 1])
end
		
function loglikelihood(obj::CivecmI2Givens, pars)
	obj.pars = pars
	ll = loglikelihood(obj)
	#println(σ_α_β(obj)[1,1], "\t", ll)
	return ll
end