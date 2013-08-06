using Distributions
# Min Civecm kode
module Civecm

import Base: convert, copy, eigvals, show
import Distributions: loglikelihood
# import Profile.@iprofile
export civecmI1, civecmI2, civecmI2alt, loglikelihood, lrtest, ranktest, setrank, show, simulate, VAR

# Auxilliary functions

function bar(A::Matrix)
	qrA = qr(A)
	return full(qrA[:Q]) / qrA[:R]'
end

function gls(Y::Matrix, X::Matrix, H::Matrix, K::SparseMatrixCSC, k::Vector, Omega::Matrix)
	iT, px = size(X)
	r = size(H, 2)
	Sxx = X'X / iT
	Sxy = X'Y / iT
	if length(k) == 0
	    k = spzeros(px * r, 1)
	    G = spzeros(px, r)
	else
		G = reshape(k, px, r)
	end
	lhs = K'kron(H' / Omega * H, Sxx) * K
	rhs = K'reshape(Sxy / Omega * H - Sxx * G * H' / Omega * H, px * r)
	return reshape(K * (qrpfact!(lhs) \ rhs) + k, px, r)
end

function lagmatrix(A::Matrix, lags::AbstractArray{Int64, 1})
	(iT, ip) = size(A)
	if isempty(lags) return Array(Float64, iT, 0) end
	ans = Array(Float64, iT - max(lags), ip * size(lags, 1))
	for i in 1:ip
		for j in 1:size(lags, 1)
			for k in 1:size(ans, 1)
				ans[k, i + (j - 1)*ip] = A[k + max(lags) - lags[j], i]
			end
		end
	end
	return ans
end

function mreg(Y::VecOrMat, X::Matrix)
	coef = qrpfact!(X'X)\(X'Y)
	residuals = Y - X*coef
	(coef, residuals)
end

# Note. This is different from the difinition used in the literature following Johansen (and T.W. Anderson). Here I define the reduced rank regression decomposition of PI such that PI=α*diag(σ)*β'
function rrr!(Y::Matrix, X::Matrix)
	iT, iX = size(X)
	iY = size(Y, 2)
	svdX = svdfact!(X, true)
	svdY = svdfact!(Y, true)
	svdZ = svdfact!(svdX[:U]'svdY[:U], true)
	Sm1 = zeros(iX)
	index = svdX[:S] .> 10e-9 * max(max(X))
	Sm1[index] = 1 ./ svdX[:S][index]
	α = svdY[:V]*Diagonal(svdY[:S])*svdZ[:V]/sqrt(iT)
	β = sqrt(iT)*svdX[:V]*Diagonal(Sm1)*svdZ[:U]
	return α, svdZ[:S], β
end
function rrr!(Y::Matrix, X::Matrix, rank::Int64)
	α, values, β = rrr(Y, X)
	return α[:,1:rank], values[1:rank], β[:,1:rank]
end
rrr(Y::Matrix, X::Matrix, args...) = rrr!(copy(Y), copy(X), args...)

# AbstractCivecm
abstract AbstractCivecm

eigvals(obj::AbstractCivecm) = eigvals(convert(VAR, obj))

loglikelihood(obj::AbstractCivecm) = -0.5 * (size(obj.endogenous, 1) - obj.lags) * logdet(residualvariance(obj))

function residualvariance(obj::AbstractCivecm)
    mresiduals = residuals(obj)
    mOmega = mresiduals'mresiduals / size(mresiduals, 1)
    return mOmega
end
	
# show(io, obj::AbstractCivecm) = println("α: ", α(obj), "\nloglikelihood: ", loglikelihood(obj))

# I1
type CivecmI1 <: AbstractCivecm
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	α::Matrix{Float64}
	β::Matrix{Float64}
	Γ::Matrix{Float64}
	llConvCrit::Float64
	maxiter::Int64
	Z0::Matrix{Float64}
	Z1::Matrix{Float64}
	Z2::Matrix{Float64}
	R0::Matrix{Float64}
	R1::Matrix{Float64}
end

function civecmI1(endogenous::Matrix{Float64}, exogenous::Matrix{Float64}, lags::Int64, rank::Int64)
	ss, p = size(endogenous)
	pexo = size(exogenous, 2)
	p1 = p + pexo
	iT = ss - lags
	obj = CivecmI1(endogenous, 
				   exogenous, 
				   lags, 
				   Array(Float64, p, rank), 
				   Array(Float64, p1, rank),
				   Array(Float64, p, (lags - 1)*p + lags*pexo), 
				   1.0e-8, 
				   5000, 
				   Array(Float64, iT, p),
				   Array(Float64, iT, p1),
				   Array(Float64, iT, (lags-1)*p + lags*pexo),
				   Array(Float64, iT, p), 
				   Array(Float64, iT, p1))
	auxilliaryMatrices(obj)
	estimateEigen(obj)
	return obj
end
civecmI1(endogenous::Matrix{Float64}, exogenous::Matrix{Float64}, lags::Int64) = civecmI1(endogenous, exogenous, lags, size(endogenous, 2))
civecmI1(endogenous::Matrix{Float64}, lags::Int64) = civecmI1(endogenous, zeros(size(endogenous, 1), 0), lags)
civecmI1(endogenous::Matrix{Float64}, exogenous::Range1, lags::Int64) = civecmI1(endogenous, float64(reshape(exogenous, length(exogenous), 1)), lags)

function auxilliaryMatrices(obj::CivecmI1)
	iT, p = size(obj.Z0)
	pexo = size(obj.exogenous, 2)
	for j = 1:p
		for i = 1:iT
			obj.Z0[i,j] = obj.endogenous[i+obj.lags,j] - obj.endogenous[i+obj.lags-1,j]
			obj.Z1[i,j] = obj.endogenous[i+obj.lags-1,j]
		end
	end
	for k = 1:obj.lags - 1
		for j = 1:p
			for i = 1:iT
				obj.Z2[i,p*(k-1)+j] = obj.endogenous[i+obj.lags-k,j] - obj.endogenous[i+obj.lags-k-1,j]
			end
		end
	end
	for j = 1:pexo
		for i = 1:iT
			obj.Z1[i,p+j] = obj.exogenous[i+obj.lags-1,j]
			obj.Z2[i,p*(obj.lags-1)+j] = obj.exogenous[i+obj.lags,j] - obj.exogenous[i+obj.lags-1,j]
		end
	end
	for k = 1:obj.lags - 1
		for j = 1:pexo
			for i = 1:iT
				obj.Z2[i,p*(obj.lags-1)+pexo*k+j] = obj.exogenous[i+obj.lags-k,j] - obj.exogenous[i+obj.lags-k-1,j]
			end
		end
	end
	if size(obj.Z2, 2) > 0
		obj.R0[:] = mreg(obj.Z0, obj.Z2)[2]
		obj.R1[:] = mreg(obj.Z1, obj.Z2)[2]
	else
		obj.R0[:] = obj.Z0
		obj.R1[:] = obj.Z1
	end
	return obj
end

copy(obj::CivecmI1) = CivecmI1(copy(obj.endogenous), 
							   copy(obj.exogenous), 
							   obj.lags, 
							   copy(obj.α), 
							   copy(obj.β), 
							   copy(obj.Γ), 
							   obj.llConvCrit, 
							   obj.maxiter, 
							   copy(obj.Z0), 
							   copy(obj.Z1), 
							   copy(obj.Z2), 
							   copy(obj.R0), 
							   copy(obj.R1))

function show(io::IO, obj::CivecmI1)
	@printf("\n   ")
	for i = 1:size(obj.α, 2)
		@printf(" α(%d)", i)
	end
	println()	
	for i = 1:size(obj.R0, 2)
		@printf("V%d:", i)
		for j = 1:size(obj.α, 2)
			@printf("%9.3f", obj.α[i,j])
		end
		println()
	end
	@printf("\n         ")
	for i = 1:size(obj.R1, 2)
		@printf("%9s", string("V", i))
	end
	println()
	for i = 1:size(obj.β, 2)
		@printf("β(%d): ", i)
		for j = 1: 1:size(obj.R1, 2)
			@printf("%9.3f", obj.β[j,i])
		end
		println()
	end
	println("\nPi:")
	print_matrix(io, obj.α*obj.β')
end

function setrank(obj::CivecmI1, rank::Int64)
	obj.α = Array(Float64, size(obj.R0, 2), rank)
	obj.β = Array(Float64, size(obj.R1, 2), rank)
	return estimateEigen(obj)
end

function estimateEigen(obj::CivecmI1)
	obj.α[:], svdvals, obj.β[:] = rrr(obj.R0, obj.R1, size(obj.α, 2))
	obj.α[:] = obj.α*Diagonal(svdvals)
	obj.Γ[:] = mreg(obj.Z0 - obj.Z1*obj.β*obj.α', obj.Z2)[1]'
	return obj
end

function estimateSwitch(obj::CivecmI1, βMatrix::Matrix, βVector::Vector, αMatrix::Matrix)
	iT = size(obj.Z0, 1)
	S11 = scale!(obj.R1'obj.R1, 1/iT)
	S10 = scale!(obj.R1'obj.R0, 1/iT)
	ll0 = -realmax()
	ll1 = ll0
	for i = 1:obj.maxiter
		OmegaInv = inv(cholfact!(residualvariance(obj)))
		aoas11 = kron(obj.α'OmegaInv*obj.α, S11)
		phi = qrpfact!(βMatrix'*aoas11*βMatrix)\(βMatrix'*(vec(S10*OmegaInv*obj.α) - aoas11*βVector))
		obj.β = reshape(βMatrix*phi + βVector, size(obj.β)...)
		γ = qrpfact!(αMatrix'kron(OmegaInv, obj.β'S11*obj.β)*αMatrix)\(αMatrix'vec(obj.β'S10*OmegaInv))
		obj.α = reshape(αMatrix*γ, size(obj.α, 2), size(obj.α, 1))'
		ll1 = loglikelihood(obj)
		if abs(ll1 - ll0) < obj.llConvCrit break end
		ll0 = ll1
	end
	return obj
end

function ranktest(obj::CivecmI1, reps::Int64)
	_, svdvals, _ = rrr(obj.R0, obj.R1)
	tmpTrace = -size(obj.Z0, 1) * reverse(cumsum(reverse(log(1.0 - svdvals.^2))))
	tmpPVals = zeros(size(tmpTrace))
	rankdist = zeros(reps)
	iT, ip = size(obj.endogenous)
	for i = 1:size(tmpTrace, 1)
		print("Simulation of model H(", i, ")\r")
		for k = 1:reps
			rankdist[k] = I2TraceSimulate(randn(iT, ip - i + 1), ip - i + 1, obj.exogenous)
		end
		tmpPVals[i] = mean(rankdist .> tmpTrace[i])
	end
	print("                                                    \r")
	return TraceTest(tmpTrace, tmpPVals)
end
ranktest(obj::CivecmI1) = ranktest(obj, 10000)

residuals(obj::CivecmI1) = obj.R0 - obj.R1*obj.β*obj.α'

# function simulate(obj::CivecmI1, innovations::Matrix{Float64})
# 	X = similar(innovations)
	
# 	for i = 2:size(X, 1)
# 		X[i,:] = X[i-1,:]*obj.β*obj.α' + X[i-1,:]
# 		for j = 1:obj.lags - 1
# 			X[i,:] += (X[i-j,:] - X[i-j-1,:])*obj.Gammat[1:size(obj.R0,2),:]
# 		end

## I1 ranktest

type TraceTest
	values::Vector{Float64}
	pvalues::Vector{Float64}
end

function show(io::IO, obj::TraceTest)
	@printf("\n Rank    Value  p-value\n")
	for i = 1:length(obj.values)
		@printf("%5d%9.3f%9.3f\n", i-1, obj.values[i], obj.pvalues[i])
	end
end

## LR test
type LRTest{T<:CivecmI1}
	H0::T
	HA::T
	value::Float64
	df::Int64
end

lrtest(obj0::CivecmI1, objA::CivecmI1, df::Integer) = LRTest(obj0, objA, 2*(loglikelihood(objA) - loglikelihood(obj0)), df)

function bootstrap(obj::LRTest, reps::Integer)
	bootH0 = copy(obj.H0)
	bootHA = copy(obj.HA)
	bootVAR = convert(VAR, bootH0)
	iT, p = size(obj.H0.Z0)
	H0Residuals = residuals(obj.H0)
	mbr = mean(H0Residuals, 1)
	bi = Array(Float64, iT)
	bootResiduals = similar(H0Residuals)
	for i = 1:reps
		bi[:] = randn(iT)
		bootResiduals[:] = copy(H0Residuals)
		for k = 1:p
			for j = 1:iT
			 	bootResiduals[j,k] -= mbr[k]
			 	bootResiduals[j,k] *= bi[j]
			 	bootResiduals[j,k] += mbr[k]
			end
		end
		bootH0.endogenous[:] = simulate(bootVAR, iT)
	end
end
# VAR
type VAR
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	endocoefs::Array{Float64, 3}
	exocoefs::Matrix{Float64}
end

function convert(::Type{VAR}, obj::CivecmI1)
	p = size(obj.endogenous, 2)
	endocoefs = Array(Float64, p, p, obj.lags)
	endocoefs[:,:,1] = obj.α*obj.β' + eye(p)
	if obj.lags > 1
		endocoefs[:,:,1] += obj.Γ[:,1:p]
		endocoefs[:,:,obj.lags] = -obj.Γ[:,(obj.lags-2)*p+1:(obj.lags-1)*p]
		for i = 1:obj.lags - 2
			endocoefs[:,:,i+1] = obj.Γ[:,p*(obj.lags-i-1)+1:p*(obj.lags-i)] - obj.Γ[:,p*(obj.lags-i-2)+1:p*(obj.lags-i-1)]
		end
	end
	return VAR(obj.endogenous, obj.exogenous, endocoefs, zeros(0,0))
end

function eigvals(obj::VAR)
	p = size(obj.endogenous, 2)
	k = size(obj.endocoefs, 3)
	return eigvals([reshape(obj.endocoefs, p, p*k); eye((k-1)*p, k*p)])
end

function simulate(obj::VAR, innovations::Matrix{Float64}, init::Matrix{Float64})
	iT, p = size(innovations)
	k = size(obj.endocoefs, 3)
	if size(obj.endocoefs, 1) != p error("Wrong dimensions") end
	if size(init, 2) != p error("Wrong dimensions") end
	if size(init, 1) != k error("Wrong dimensions") end
	Y = zeros(iT + k, p)
	Y[1:k,:] = init
	for t = 1:iT
		for i = 1:k
			Y[t+k,:] += Y[t+k-i,:]*obj.endocoefs[:,:,i]' + innovations[t,:]
		end
	end
	return Y
end
simulate(obj::VAR, innovations::Matrix{Float64}) = simulate(obj, innovations, obj.endogenous[1:size(obj.endocoefs, 3),:])
simulate(obj::VAR, iT::Integer) = simulate(obj, randn(iT, size(obj.endogenous, 2)))

# I2

type CivecmI2 <: AbstractCivecm
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	rankI1::Int64
	rankI2::Int64
	α::Matrix{Float64}
	β::Matrix{Float64}
	ν::Matrix{Float64}
	ξ::Matrix{Float64}
	γ::Matrix{Float64}
	σ::Matrix{Float64}
	llConvCrit::Float64
	maxiter::Int64
	method::ASCIIString
	Z0::Matrix{Float64}
	Z1::Matrix{Float64}
	Z2::Matrix{Float64}
	Z3::Matrix{Float64}
	R0::Matrix{Float64}
	R1::Matrix{Float64}
	R2::Matrix{Float64}
end

function civecmI2(endogenous::Matrix{Float64}, exogenous::Matrix{Float64}, lags::Int64, rankI1::Int64, rankI2::Int64)
	ss, p = size(endogenous)
	iT = ss - lags
	pexo = size(exogenous, 2)
	p1 = p + pexo
	obj = CivecmI2(endogenous, 
					exogenous,
					lags,
					rankI1,
					rankI2,
					Array(Float64, p, rankI1), 
					Array(Float64, p1, rankI1),
					Array(Float64, p1, rankI1),
					Array(Float64, p, rankI2),
					Array(Float64, p1, rankI2),
					Array(Float64, p, rankI1),
					1.0e-8, 
					50000, 
					"Johansen",
					Array(Float64, iT, p), 
					Array(Float64, iT, p1),
					Array(Float64, iT, p1),
					Array(Float64, iT, p*(lags - 2) + pexo*(lags - 1)),
					Array(Float64, iT, p),
					Array(Float64, iT, p1), 
					Array(Float64, iT, p1))
	auxilliaryMatrices(obj)
	estimate(obj)
	return obj
end
civecmI2(endogenous::Matrix{Float64}, exogenous::Matrix{Float64}, lags::Int64) = civecmI2(endogenous, exogenous, lags, size(endogenous, 2), 0)
civecmI2(endogenous::Matrix{Float64}, lags::Int64) = civecmI2(endogenous, zeros(size(endogenous, 1), 0), lags)
civecmI2(endogenous::Matrix{Float64}, exogenous::Range1, lags::Int64) = civecmI2(endogenous, float64(reshape(exogenous, length(exogenous), 1)), lags)

function auxilliaryMatrices(obj::CivecmI2)
	iT, p = size(obj.Z0)
	pexo = size(obj.exogenous, 2)
	for j = 1:p
		for i = 1:iT
			obj.Z0[i,j] = obj.endogenous[i+obj.lags,j] - 2obj.endogenous[i+obj.lags-1,j] + obj.endogenous[i+obj.lags-2,j]
			obj.Z1[i,j] = obj.endogenous[i+obj.lags-1,j] - obj.endogenous[i+obj.lags-2,j]
			obj.Z2[i,j] = obj.endogenous[i+obj.lags-1,j]
		end
	end
	for k = 1:obj.lags - 2
		for j = 1:p
			for i = 1:iT
				obj.Z3[i,p*(k-1)+j] = obj.endogenous[i+obj.lags-k,j] - 2obj.endogenous[i+obj.lags-k-1,j] + obj.endogenous[i+obj.lags-k-2,j]
			end
		end
	end
	for j = 1:pexo
		for i = 1:iT
			obj.Z1[i,p+j] = obj.exogenous[i+obj.lags-1,j] - obj.exogenous[i+obj.lags-2,j]
			obj.Z2[i,p+j] = obj.exogenous[i+obj.lags-1,j]
			obj.Z3[i,p*(obj.lags-2)+j] = obj.exogenous[i+obj.lags,j] - 2obj.exogenous[i+obj.lags-1,j] + obj.exogenous[i+obj.lags-2,j]
		end
	end
	for k = 1:obj.lags - 2
		for j = 1:pexo
			for i = 1:iT
				obj.Z3[i,p*(obj.lags-2)+pexo*k+j] = obj.exogenous[i+obj.lags-k,j] - 2obj.exogenous[i+obj.lags-k-1,j] + obj.exogenous[i+obj.lags-k-2,j]
			end
		end
	end
	if size(obj.Z3, 2) > 0
		obj.R0[:] = mreg(obj.Z0, obj.Z3)[2]
		obj.R1[:] = mreg(obj.Z1, obj.Z3)[2]
		obj.R2[:] = mreg(obj.Z2, obj.Z3)[2]
	else
		obj.R0[:] = obj.Z0
		obj.R1[:] = obj.Z1
		obj.R2[:] = obj.Z2
	end
	return obj
end

copy(obj::CivecmI2) = CivecmI2(copy(obj.endogenous),
							   copy(obj.exogenous),
							   obj.lags,
							   obj.rankI1,
							   obj.rankI2,
							   copy(obj.α),
							   copy(obj.β),
							   copy(obj.ν),
							   copy(obj.ξ),
							   copy(obj.γ),
							   copy(obj.σ),
							   obj.llConvCrit,
							   obj.maxiter,
							   obj.method,
							   copy(obj.Z0),
							   copy(obj.Z1),
							   copy(obj.Z2),
							   copy(obj.Z3),
							   copy(obj.R0),
							   copy(obj.R1),
							   copy(obj.R2))

function setrank(obj::CivecmI2, rankI1::Int64, rankI2::Int64)
	if rankI1 + rankI2 > size(obj.endogenous, 2)
		error("Illegal choice of rank")
	else
		p = size(obj.endogenous, 2)
		p1 = p + size(obj.exogenous, 2)
		obj.rankI1 = rankI1
		obj.rankI2 = rankI2
		obj.α = Array(Float64, p, rankI1)
		obj.β = Array(Float64, p1, rankI1)
		obj.ν = Array(Float64, p1, rankI1)
		obj.ξ = Array(Float64, p, rankI2)
		obj.γ = Array(Float64, p1, rankI2)
		obj.σ = Array(Float64, p, rankI1)
	end
	return estimate(obj)
end

function estimate(obj::CivecmI2)
	if obj.method == "MP" return estimateSwitch(obj) end
	if obj.method == "Johansen" return estimateτSwitch(obj) end
	error("No method named %obj.method")
end

function estimateSwitch(obj::CivecmI2)
	iT, ip = size(obj.R0)
	ip1 = size(obj.R1, 2)

	if max(obj.rankI1, obj.rankI2) == 0
		obj.α[:] 	= zeros(ip, obj.rankI1)
		obj.β[:] 	= zeros(ip1, obj.rankI1)
		obj.ν[:] 		= zeros(ip1, obj.rankI1)            
		obj.ξ[:] 		= zeros(ip, obj.rankI2)
		obj.γ[:] 	= zeros(ip1, obj.rankI2)
		obj.σ[:] 	= zeros(ip, obj.rankI1)
	elseif obj.rankI1 == ip
		tmpFit 		= setrank(civecmI1(obj.endogenous, obj.exogenous, obj.lags), obj.rankI1)
		obj.α[:] 	= tmpFit.α
		obj.β[:] 	= tmpFit.β
		obj.σ[:] 	= eye(ip, obj.rankI1)
		tmpFit2 	= [obj.R1 obj.R2 * obj.β] \ obj.R0
		tmpGam 		= tmpFit2[1:ip1,:]'
		obj.ν[:] 	= (obj.α \ (tmpGam - obj.σ * obj.β'))'
		obj.ξ[:] 	= zeros(ip, 0)
		obj.γ[:] 	= zeros(ip1, 0)
	else
		tmp1 = speye(2 * obj.rankI1 + obj.rankI2 + 1, 2 * obj.rankI1 + obj.rankI2)
		tmp2 = copy(tmp1[[reshape(reshape(1:2 * obj.rankI1, obj.rankI1, 2)', 2 * obj.rankI1), reshape([(2 * obj.rankI1 + obj.rankI2 + 1) * ones(Int, obj.rankI2) (1:obj.rankI2) + 2 * obj.rankI1]', 2 * obj.rankI2), reshape([(2 * obj.rankI1 + obj.rankI2 + 1) * ones(Int, obj.rankI1) 1:obj.rankI1]', 2 * obj.rankI1)], :])
		K = kron(tmp2, speye(size(obj.R1, 2)))
	    
	    R2R1 = [obj.R2 obj.R1]
	   	tmpX = [obj.R2 * obj.β + obj.R1 * obj.ν obj.R1 * obj.γ obj.R1 * obj.β]
		tmpCoef = tmpX \ obj.R0
		mOmega = eye(ip)
		tmpGLS = gls(obj.R0, [obj.R2 obj.R1], [obj.α obj.ξ obj.σ], K, [], mOmega)
		for k = 1:20
			# Iterative procedure
			# Initialisation
			if obj.rankI1 == 0
				obj.α[:] 	= zeros(ip, 0)
				obj.β[:] 	= zeros(ip1, 0)
				obj.ν[:] 		= zeros(ip1, 0)
				obj.γ[:] 	= rrr(obj.R0, obj.R1, obj.rankI2)[3]
				obj.ξ[:] 		= ((obj.R1 * obj.γ) \ obj.R0)' #'
				obj.σ[:] 	= zeros(ip, 0)
				mOmega = residualvariance(obj)
			else
			#if ip == sum(obj.rank)
				if k == 1
					tmpFit = setrank(civecmI1(obj.endogenous, obj.exogenous, obj.lags), obj.rankI1+ obj.rankI2)
					obj.α[:] 	= tmpFit.α[:,1:obj.rankI1]
					obj.β[:] 	= tmpFit.β[:,1:obj.rankI1]
					# tmpFit2 	= mreg(obj.R0, [obj.R1 obj.R2 * obj.β])[1]
					# tmpGam 		= copy(tmpFit2[1:ip1,:])'
					obj.γ[:] 	= tmpFit.β[:,obj.rankI1+1:obj.rankI1+obj.rankI2]
					# obj.σ 	= eye(ip, obj.rankI1)
					obj.σ[:]	= Array(Float64, ip, obj.rankI1)
					obj.ν[:]	= full(qrfact!([obj.β obj.γ])[:Q], false)[:,end-obj.rankI1+1:end]
					# obj.ν 		= (obj.α \ (tmpGam - obj.σ * obj.β'))'
					obj.ξ[:] 	= Array(Float64, ip, obj.rankI2)
					mOmega 		= residualvariance(obj)
				else
					obj.β[:] 	= randn(ip1, obj.rankI1)
					obj.ν[:] 	= randn(ip1, obj.rankI1)            
					obj.γ[:] 	= randn(ip1, obj.rankI2)
					mOmega 		= residualvariance(obj)
				end
			end
			# else
				# m = min(obj.rankI1, ip - obj.rankI1 - obj.rankI2);
			 	# tmpFit = CivecmI2alt(obj.endogenous, obj.exogenous, obj.lags).setrank([obj.rankI1 - m, obj.rankI2 + 2 * m]);
				# [obj.α, obj.β, obj.ν, obj.ξ, obj.γ, obj.σ] = CivecmI2alt.initialPars(tmpFit.α, tmpFit.β, tmpFit.ν, tmpFit.ξ, tmpFit.γ, tmpFit.σ, 1, m)
			

			ll = -1.0e9
			ll0 = ll

			for j = 0:obj.maxiter
				# The AC-step
				# println(obj.β)
				tmpX[:] 		= [obj.R2 * obj.β + obj.R1 * obj.ν obj.R1 * obj.γ obj.R1 * obj.β]
				tmpCoef[:] 		= qrpfact!(tmpX'tmpX) \ (tmpX'obj.R0)
				obj.α[:] 	= tmpCoef[1:obj.rankI1, :]'
				obj.ξ[:] 		= tmpCoef[obj.rankI1 + 1:(obj.rankI1 + obj.rankI2),:]'
				obj.σ[:] 	= tmpCoef[obj.rankI1 + obj.rankI2 + 1:end, :]'
				# The CC Step ala Rocco
				tmpGLS[:] 		= gls(obj.R0, R2R1, [obj.α obj.ξ obj.σ], K, [], mOmega)
				obj.ν[:] 		= tmpGLS[ip1 + 1:, 1:obj.rankI1]
				obj.γ[:] 	= tmpGLS[ip1 + 1:, obj.rankI1 + 1:obj.rankI1 + obj.rankI2]
				obj.β[:] 	= tmpGLS[ip1 + 1:, obj.rankI1 + obj.rankI2 + 1:]
                # Residual variable step
                ll0 		= ll
                mOmega[:]	= residualvariance(obj)
                ll 			= -0.5 * logdet(cholfact(mOmega))
                # @printf("Average log-likelihood value: %f\n", ll)
                if abs(ll - ll0) < obj.llConvCrit
                	@printf("Convergence in %d iterations.\n", j - 1)
                	break
                end
            end
            if norm(obj.ν) > 1.0e6; println(obj.α, obj.ν, obj.α*obj.ν'); end
            if
				abs(ll - ll0) < obj.llConvCrit
				break
			end             
			print("Om igen!")
		end
	end
	return obj
end

function estimateτSwitch(obj::CivecmI2)
	# Dimentions
	p = size(obj.R0, 2)
	iT, p1 = size(obj.R2)
	rs = obj.rankI1 + obj.rankI2

	# Moment matrices
	S10 = obj.R1'obj.R0/iT
	S11 = obj.R1'obj.R1/iT
	S20 = obj.R2'obj.R0/iT
	S21 = obj.R2'obj.R1/iT
	S22 = obj.R2'obj.R2/iT

	# Memory allocation
	Rτ = Array(Float64, iT, p1)
	R1τ = Array(Float64, iT, rs)
	workX = Array(Float64, rs, p1)
	mX = Array(Float64, iT, p1)
	workY = Array(Float64, rs, p)
	mY = Array(Float64, iT, p)
	αort = Array(Float64, p, p - obj.rankI1)
	workRRR = Array(Float64, obj.rankI1)
	ρδ = Array(Float64, p1, obj.rankI1)
	ρ = sub(ρδ, 1:rs, 1:obj.rankI1)
	ρort = Array(Float64, rs, rs - obj.rankI1)
	δ = sub(ρδ, rs+1:p1, 1:obj.rankI1)
	ζt = Array(Float64, rs, p)
	ζtαort = Array(Float64, rs, p - obj.rankI1)
	res = Array(Float64, iT, p)
	_, _, τ = rrr(obj.R0, obj.R1, rs)
	τort = Array(Float64, p1, p1 - rs)
	# Ω = Cholesky(eye(p), 'U')
	Ω = Array(Float64, p, p)
	A = Array(Float64, rs, rs)
	B = Array(Float64, p1, p1)
	C = Array(Float64, rs, rs)
	D = Array(Float64, p1, p1)
	E = Array(Float64, p1*rs)

	# Algorithm
	ll = -realmax()
	ll0 = ll
	for j = 1:obj.maxiter
		τort[:] = null(τ')
		Rτ[:,1:rs] = obj.R2*τ
		Rτ[:,rs+1:end] = obj.R1*τort
		R1τ[:] = obj.R1*τ
		workX[:], mX[:] = mreg(Rτ, R1τ)
		workY[:], mY[:] = mreg(obj.R0, R1τ)
		obj.α[:], workRRR[:], ρδ[:] = rrr(mY, mX, obj.rankI1)
		obj.α[:] = obj.α*Diagonal(workRRR)
		ζt[:], res[:] = mreg(obj.R0 - Rτ*ρδ*obj.α', R1τ)
		Ω[:] = res'res/iT
		ll = -0.5logdet(cholfact(Ω))
		if abs(ll - ll0) < obj.llConvCrit 
			@printf("Convergence in %d iterations.\n", j - 1)
			break 
		end
		ll0 = ll
		# LinAlg.LAPACK.potrf!('U', Ω.UL)
		αort[:] = null(obj.α')
		A[:] = ρ*obj.α'*(Ω\obj.α)*ρ'
		B[:] = S22
		ζtαort[:] = ζt*αort
		C[:] = ζtαort*(cholfact!(αort'Ω*αort)\(ζtαort'))
		D[:] = S11
		E[:] = S20*(Ω\obj.α)*ρ' - S21*(τort*δ*obj.α' + τ*ζt)*(Ω\obj.α)*ρ' + S10*αort*(cholfact(	αort'Ω*αort)\(ζtαort'))
		τ[:] = reshape(qrpfact!(kron(A,B) + kron(C,D))\E, rs, p1)
		tmp = sqrtm(Hermitian(τ'S22*τ))
		if !isreal(tmp) 
			tmp1 = sqrtm(Hermitian(LinAlg.syrk_wrapper('T', cholfact(S22, :U)[:U]*τ)))
			println(tmp) 
			println(tmp1)
		end
		τ[:] = τ/sqrtm(Hermitian(τ'S22*τ))
	end
	# Back to Mosconi/Paruolo pars
	obj.β[:] = τ*ρ
	obj.ν[:] = τort*δ
	ρort[:] = null(ρ')
	obj.ξ[:] = ζt'ρort
	obj.γ[:] = τ*ρort
	obj.σ[:] = ζt'pinv(ρ')
	return obj
end

function ranktest(obj::CivecmI2)
	ip 			= size(obj.endogenous, 2)
	ll0 		= loglikelihood(setrank(obj, ip, 0))
	tmpTrace 	= zeros(ip, ip + 1)
	for i = 0:ip - 1
		for j = 0:ip - i
			tmpTrace[i + 1, i + j + 1] = 2 * (ll0 - loglikelihood(setrank(obj, i, j)))
		end
	end
	tmpTrace
end

function ranktest(obj::CivecmI2, reps::Int64)
	vals 	= ranktest(obj)
	pvals 	= ranktestpvalues(obj, vals, reps)
	return (vals, pvals)
end

function ranktestpvalues(obj::CivecmI2, testvalues::Matrix, reps::Int64)
	(iT, ip) = size(obj.endogenous)
	pvals = zeros(ip, ip + 1)
	rankdist = zeros(reps)
	for i = 0:ip - 1
	    for j = 0:ip - i
			for k = 1:reps
				rankdist[k] = I2TraceSimulate(randn(iT, ip - i), j, obj.exogenous)
			end
			pvals[i + 1, i + j + 1] = mean(rankdist .> testvalues[i + 1, i + j + 1])
			@printf("Simulation of model H(%d,%d). %3.2f percent completed.\n", i, j, 100 * (0.5 * i * (i + 1) + i * (ip - i + 1) + j + 1) / (0.5 * ip^2 + 1.5 * ip))
		end
	end
	return pvals
end

function residuals(obj::CivecmI2)
    res = obj.R0 - obj.R2 * obj.β * obj.α' - obj.R1*obj.ν*obj.α' - obj.R1*obj.γ*obj.ξ' - obj.R1*obj.β*obj.σ'
    return res
end

# Simulation of rank test
function fS(dX::Matrix{Float64}, Y::Matrix{Float64}, dZ::Matrix{Float64})
	A = dX[2:,:]::Matrix{Float64}
	B = Y[1:size(Y, 1) - 1,:]::Matrix{Float64}
	C = dZ[2:,:]::Matrix{Float64}
	return (A'B)*(cholfact!(B'B)\(B'C))
end

function I2TraceSimulate(eps::Matrix{Float64}, s::Int64, exo::Matrix{Float64})	
	iT 	= size(eps, 1)
	w 	= cumsum(eps) / sqrt(iT)
	w2i = cumsum(w[:,s + 1:]) / iT

	m1 	= [w[:,1:s] w2i exo]
	m2 	= [w[:,s + 1:] diff([zeros(1,size(exo, 2)); exo])]

	if size(m2, 2) > 0
		tmpCoef = (m2'm2) \ (m2'm1)
		g = m1 - m2 * tmpCoef
	else
		g = m1
	end
	epsOrth 	= eps / chol(eps'eps / iT)
	tmp1 		= eigvals(fS(epsOrth, g, epsOrth) / iT)
	if size(eps, 2) > s
		tmp2 	= eigvals(fS(epsOrth[:,s + 1:], m2, epsOrth[:,s + 1:]) / iT)
	else
		tmp2 	= [0.0]
	end
	return (-iT * (sum(log(1.0 - tmp1)) + sum(log(1.0 - tmp2))))
end

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
end