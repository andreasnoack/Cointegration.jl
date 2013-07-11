# Min Civecm kode
module Civecm

import Base.show

export civecmI1, civecmI2, civecmI2alt, show, ranktest, setrank

abstract AbstractCivecm

logLik(obj::AbstractCivecm) = -0.5 * (size(obj.endogenous, 1) - obj.lags) * logdet(residualvariance(obj))

function rrr(Y::Matrix, X::Matrix)
	iT, iX = size(X)
	iY = size(Y, 2)
	svdX = svdfact(X, true)
	svdY = svdfact(Y, true)
	svdZ = svdfact(svdX[:U]'svdY[:U], true)
	values = svdZ[:S].^2
	Sm1 = zeros(iX)
	index = svdX[:S] .> 10e-9 * max(max(X))
	Sm1[index] = 1 ./ svdX[:S][index]
	vectors = sqrt(iT) * svdX[:V] * scale(Sm1, svdZ[:U])
	return (values, vectors)
end

function rrr(Y::Matrix, X::Matrix, rank::Int64)
	(tmp1, tmp2) = rrr(Y, X)
	return (tmp1[1:rank], tmp2[:,1:rank])
end

function bar(A::Matrix)
	qrA = qr(A)
	return full(qrA[:Q]) / qrA[:R]'
end

function mreg(Y::VecOrMat, X::Matrix)
	# try
	# 	coef = X \ Y
	# 	residuals = Y - X*coef
	# 	(coef, residuals)
	# catch
	# 	(mU, vS, mV) = svd(X, true)
	# 	vSinv = zeros(size(vS))
	# 	vSinv[vS .> 0] = 1 ./ vS[vS .> 0]
	# 	coef = mV * diagm(vSinv) * mU'Y
	# 	residuals = Y - X*coef
	# 	(coef, residuals)		
	# end
	coef = LinAlg.LAPACK.gelsd!(X'X, X'Y)[1]
	residuals = Y - X*coef
	(coef, residuals)
end

logdet(matrix::Matrix) = 2 * sum(log(diag(chol(matrix))))

function gls(Y::Matrix, X::Matrix, H::Matrix, K::SparseMatrixCSC, k::Vector, Omega::Matrix)
	(iT, px) = size(X)
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
	return reshape(K * (lhs \ rhs) + k, px, r)
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

function residualvariance(obj::AbstractCivecm)
    mresiduals = residuals(obj)
    mOmega = mresiduals'mresiduals / size(mresiduals, 1)
    return mOmega
end

function lagmatrixOld(A::Matrix, lags::AbstractArray{Int64, 1})
	(iT, ip) = size(A)
	ans = Array(Float64, iT - max(lags), ip * size(lags, 1))
	for i in 1:size(lags, 1)
		for j in 1:size(ans, 1)
			ans[j, ((i - 1)*ip + 1):(i*ip)] = A[j + max(lags) - lags[i], :]
		end
	end
	return ans
end
	
# show(io, obj::AbstractCivecm) = println("alpha: ", alpha(obj), "\nlogLik: ", logLik(obj))

# I1
type CivecmI1 <: AbstractCivecm
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	alpha::Matrix{Float64}
	beta::Matrix{Float64}
	llConvCrit::Float64
	maxIter::Int64
	Z0::Matrix{Float64}
	Z1::Matrix{Float64}
	Z2::Matrix{Float64}
	R0::Matrix{Float64}
	R1::Matrix{Float64}
	eigvals::Vector{Float64}
	eigvecs::Matrix{Float64}
end

function civecmI1(endogenous::Matrix{Float64}, exogenous::Matrix{Float64}, lags::Int64)
	mDX = diff(endogenous)
	mLDX = lagmatrix(mDX, 1:lags - 1)
	mDU = diff(exogenous)
	mLDU = lagmatrix(mDU, 0:lags - 1)
	Z0 = mDX[lags:, :]::Matrix{Float64}
	Z1 = [endogenous[lags:(end - 1), :] exogenous[lags:(end - 1), :]]::Matrix{Float64}
	Z2 = copy(mLDX)
	if size(mLDU, 1) > 0
		Z2 = [Z2 mLDU]
	end
	if size(Z2, 2) > 0
		R0 = mreg(Z0, Z2)[2]
		R1 = mreg(Z1, Z2)[2]
	else
		R0 = Z0
		R1 = Z1
	end
	(eigvals, eigvecs) = rrr(R0, R1)

	retobj = CivecmI1(endogenous, exogenous, 2, Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), 1.0e-8, 5000, Z0, Z1, Z2, R0, R1, eigvals, eigvecs)
	setrank(retobj, size(Z0, 2))
	return retobj
end
civecmI1(endogenous::Matrix{Float64}, lags::Int64) = civecmI1(endogenous, zeros(size(endogenous, 1), 0), lags)
civecmI1(endogenous::Matrix{Float64}, exogenous::Range1, lags::Int64) = civecmI1(endogenous, float64(reshape(exogenous, length(exogenous), 1)), lags)

function show(io::IO, obj::CivecmI1)
	@printf("\n   ")
	for i = 1:size(obj.alpha, 2)
		@printf(" alpha(%d)", i)
	end
	println()	
	for i = 1:size(obj.R0, 2)
		@printf("V%d:", i)
		for j = 1:size(obj.alpha, 2)
			@printf("%9.3f", obj.alpha[i,j])
		end
		println()
	end
	@printf("\n         ")
	for i = 1:size(obj.R1, 2)
		@printf("%9s", string("V", i))
	end
	println()
	for i = 1:size(obj.beta, 2)
		@printf("beta(%d): ", i)
		for j = 1: 1:size(obj.R1, 2)
			@printf("%9.3f", obj.beta[j,i])
		end
		println()
	end
	println("\nPi:")
	print_matrix(OUTPUT_STREAM, obj.alpha*obj.beta')
end

function setrank(obj::CivecmI1, rank::Int64)
	return estimateEigen(obj, rank)
end

function estimateEigen(obj::CivecmI1, rank::Int64)
	(obj.eigvals, obj.eigvecs) = rrr(obj.R0, obj.R1, size(obj.R0, 2))
	obj.beta = copy(obj.eigvecs[:, 1:rank])
	obj.alpha = mreg(obj.R0, obj.R1 * obj.beta)[1]' #'
	return obj
end

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

function ranktest(obj::CivecmI1, reps::Int64)
	tmpTrace = -size(obj.Z0, 1) * reverse(cumsum(reverse(log(1.0 - obj.eigvals))))
	tmpPVals = zeros(size(tmpTrace))
	rankdist = zeros(reps)
	(iT, ip) = size(obj.endogenous)
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

residuals(obj::CivecmI1) = obj.R0 - obj.R1*obj.beta*obj.alpha'

# I2

type CivecmI2 <: AbstractCivecm
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	rank::(Int64,Int64)
	alpha::Matrix{Float64}
	beta::Matrix{Float64}
	nu::Matrix{Float64}
	xi::Matrix{Float64}
	gamma::Matrix{Float64}
	sigma::Matrix{Float64}
	llConvCrit::Float64
	maxIter::Int64
	Z0::Matrix{Float64}
	Z1::Matrix{Float64}
	Z2::Matrix{Float64}
	Z3::Matrix{Float64}
	R0::Matrix{Float64}
	R1::Matrix{Float64}
	R2::Matrix{Float64}
end

function civecmI2(endogenous::Matrix{Float64}, exogenous::Matrix{Float64}, lags::Int64)
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

	return CivecmI2(endogenous, float64(exogenous), lags, (size(endogenous, 2), 0), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), 1.0e-8, 50000, Z0, Z1, Z2, Z3, R0, R1, R2)
end

civecmI2(endogenous::Matrix{Float64}, lags::Int64) = civecmI2(endogenous, zeros(size(endogenous, 1), 0), lags)
civecmI2(endogenous::Matrix{Float64}, exogenous::Range1, lags::Int64) = civecmI2(endogenous, float64(reshape(exogenous, length(exogenous), 1)), lags)

function setrank(obj::CivecmI2, rank::(Int64, Int64))
	if sum(rank) > size(obj.endogenous, 2)
		error("Illegal choice of rank")
	else
		obj.rank = rank
	end
	return estimate(obj)
end

function estimate(obj::CivecmI2)
	(iT, ip) = size(obj.R0)
	ip1 = size(obj.R1, 2)

	if max(obj.rank) == 0
		obj.alpha 	= zeros(ip, obj.rank[1])
		obj.beta 	= zeros(ip1, obj.rank[1])
		obj.nu 		= zeros(ip1, obj.rank[1])            
		obj.xi 		= zeros(ip, obj.rank[2])
		obj.gamma 	= zeros(ip1, obj.rank[2])
		obj.sigma 	= zeros(ip, obj.rank[1])
	elseif obj.rank[1] == ip
		tmpFit 		= setrank(CivecmI1(obj.endogenous, obj.exogenous, obj.lags), obj.rank[1])
		obj.alpha 	= copy(tmpFit.alpha)
		obj.beta 	= copy(tmpFit.beta)
		obj.sigma 	= eye(ip, obj.rank[1])
		tmpFit2 	= [obj.R1 obj.R2 * obj.beta] \ obj.R0
		tmpGam 		= copy(tmpFit2[1:ip1,:])'
		obj.nu 		= (obj.alpha \ (tmpGam - obj.sigma * obj.beta'))'
		obj.xi 		= zeros(ip, 0)
		obj.gamma 	= zeros(ip1, 0)
	else
		tmp1 = speye(2 * obj.rank[1] + obj.rank[2] + 1, 2 * obj.rank[1] + obj.rank[2])
		tmp2 = copy(tmp1[[reshape(reshape(1:2 * obj.rank[1], obj.rank[1], 2)', 2 * obj.rank[1]), reshape([(2 * obj.rank[1] + obj.rank[2] + 1) * ones(Int, obj.rank[2]) (1:obj.rank[2]) + 2 * obj.rank[1]]', 2 * obj.rank[2]), reshape([(2 * obj.rank[1] + obj.rank[2] + 1) * ones(Int, obj.rank[1]) 1:obj.rank[1]]', 2 * obj.rank[1])], :])
		K = kron(tmp2, speye(size(obj.R1, 2)))
	    
		for k = 1:20
			# Iterative procedure
			# Initialisation
			if obj.rank[1] == 0
				obj.alpha 	= zeros(ip, 0)
				obj.beta 	= zeros(ip1, 0)
				obj.nu 		= zeros(ip1, 0)
				obj.gamma 	= rrr(obj.R0, obj.R1, obj.rank[2])[2]
				obj.xi 		= ((obj.R1 * obj.gamma) \ obj.R0)' #'
				obj.sigma 	= zeros(ip, 0)
				mOmega = residualvariance(obj)
			else
			#if ip == sum(obj.rank)
				if k == 1
					tmpFit = setrank(CivecmI1(obj.endogenous, obj.exogenous, obj.lags), obj.rank[1])
					obj.alpha 	= copy(tmpFit.alpha)
					obj.beta 	= copy(tmpFit.beta)
					# tmpFit2 	= mreg(obj.R0, [obj.R1 obj.R2 * obj.beta])[1]
					# tmpGam 		= copy(tmpFit2[1:ip1,:])'
					obj.gamma 	= randn(ip1, obj.rank[2])
					# obj.sigma 	= eye(ip, obj.rank[1])
					obj.sigma	= randn(ip, obj.rank[1])
					obj.nu		= randn(ip1, obj.rank[1])
					# obj.nu 		= (obj.alpha \ (tmpGam - obj.sigma * obj.beta'))'
					obj.xi 		= randn(ip, obj.rank[2])
					mOmega 		= residualvariance(obj)
				else
					obj.beta 	= randn(ip1, obj.rank[1])
					obj.nu 		= randn(ip1, obj.rank[1])            
					obj.gamma 	= randn(ip1, obj.rank[2])
					mOmega 		= residualvariance(obj)
				end
			end
			# else
				# m = min(obj.rank[1], ip - sum(obj.rank));
			 	# tmpFit = CivecmI2alt(obj.endogenous, obj.exogenous, obj.lags).setrank([obj.rank[1] - m, obj.rank[2] + 2 * m]);
				# [obj.alpha, obj.beta, obj.nu, obj.xi, obj.gamma, obj.sigma] = CivecmI2alt.initialPars(tmpFit.alpha, tmpFit.beta, tmpFit.nu, tmpFit.xi, tmpFit.gamma, tmpFit.sigma, 1, m)
			

			ll = -1.0e9
			ll0 = ll
			for j = 0:obj.maxIter
				# The AC-step
				# println(obj.beta)
				tmpX 		= [obj.R2 * obj.beta + obj.R1 * obj.nu obj.R1 * obj.gamma obj.R1 * obj.beta]
				tmpCoef 	= (tmpX'tmpX) \ (tmpX'obj.R0)
				obj.alpha 	= copy(tmpCoef[1:obj.rank[1], :]') 					#'
				obj.xi 		= copy(tmpCoef[obj.rank[1] + 1:sum(obj.rank),:]') 	#'
				obj.sigma 	= copy(tmpCoef[sum(obj.rank) + 1:end, :]') 			#'
				# The CC Step ala Rocco
				tmpCoef 	= gls(obj.R0, [obj.R2 obj.R1], [obj.alpha obj.xi obj.sigma], K, [], mOmega)
				obj.nu 		= copy(tmpCoef[ip1 + 1:, 1:obj.rank[1]])
				obj.gamma 	= copy(tmpCoef[ip1 + 1:, obj.rank[1] + 1:sum(obj.rank)])
				obj.beta 	= copy(tmpCoef[ip1 + 1:, sum(obj.rank) + 1:])
                # Residual variable step
                ll0 		= ll
                mOmega 		= residualvariance(obj)
                ll 			= -0.5 * iT * logdet(mOmega) / iT
                # @printf("Average log-likelihood value: %f\n", ll)
                if abs(ll - ll0) < obj.llConvCrit
                	@printf("Convergence in %d iterations.\n", j)
                	break
                end
            end
            if norm(obj.nu) > 1.0e6; println(obj.alpha, obj.nu, obj.alpha*obj.nu'); end
            if
				abs(ll - ll0) < obj.llConvCrit
				break
			end             
			print("Om igen!")
		end
	end
	return obj
end

function ranktest(obj::CivecmI2)
	ip 			= size(obj.endogenous, 2)
	ll0 		= logLik(setrank(obj, (ip, 0)))
	tmpTrace 	= zeros(ip, ip + 1)
	for i = 0:ip - 1
		for j = 0:ip - i
			tmpTrace[i + 1, i + j + 1] = 2 * (ll0 - logLik(setrank(obj, (i, j))))
		end
	end
	tmpTrace
end

function ranktest(obj::CivecmI2, reps::Int64)
	vals 	= rank(obj)
	pvals 	= rankpvalues(obj, vals, reps)
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
    res = obj.R0 - obj.R2 * obj.beta * obj.alpha' - obj.R1 * [obj.nu obj.gamma obj.beta] * [obj.alpha obj.xi obj.sigma]'	#'
    return res
end

# Simulation of rank test
function fS(dX::Matrix{Float64}, Y::Matrix{Float64}, dZ::Matrix{Float64})
	A = dX[2:,:]::Matrix{Float64}
	B = Y[1:size(Y, 1) - 1,:]::Matrix{Float64}
	C = dZ[2:,:]::Matrix{Float64}
	return (A'B)*((B'B)\(B'C))
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
	epsOrth 	= eps / chol(eps'eps / iT)	#'
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
	maxIter::Int64
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

residuals(obj::CivecmI2Givens) = obj.R0 - obj.R2*beta(obj)*Sigma_alpha_beta(obj)*alpha(obj)' - obj.R1*Gamma(obj)'

function alpha(obj::CivecmI2Givens, full::Bool)
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
alpha(obj::CivecmI2Givens) = alpha(obj, false)

function beta(obj::CivecmI2Givens, full::Bool)
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
beta(obj::CivecmI2Givens) = beta(obj, false)

function Sigma_alpha_beta(obj::CivecmI2Givens)
	p = size(obj.Z0, 2)
	p1 = size(obj.Z1, 2)
	count = div(p*(p-1),2)-div((p-obj.rank[1])*(p-obj.rank[1]-1),2) + 
		div(p1*(p1-1),2)-div((p1-obj.rank[1])*(p1-obj.rank[1]-1),2) + 1
	return diagm(obj.pars[count:count + obj.rank[1] - 1])
end

function Gamma(obj::CivecmI2Givens)
	p = size(obj.Z0, 2)
	p1 = size(obj.Z1, 2)
	count = div(p*(p-1),2)-div((p-obj.rank[1])*(p-obj.rank[1]-1),2) + 
		div(p1*(p1-1),2)-div((p1-obj.rank[1])*(p1-obj.rank[1]-1),2) + 
		obj.rank[1] + 1
	ans = Array(Float64, p, p1)
	ans[:,1:obj.rank[1]] = obj.pars[count:count + p*obj.rank[1] - 1]
	ans[1:obj.rank[1],obj.rank[1] + 1:end] = obj.pars[count + p*obj.rank[1]:count + p*obj.rank[1] + (p - obj.rank[1])*obj.rank[1] - 1]
	ans[obj.rank[1] + 1:,obj.rank[1] + 1:] = xi(obj)*Sigma_xi_eta(obj)*etapar(obj)'
	return alpha(obj, true)*ans*beta(obj, true)'
end

function xi(obj::CivecmI2Givens)
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

function etapar(obj::CivecmI2Givens)
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

function Sigma_xi_eta(obj::CivecmI2Givens)
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
		
function logLik(obj::CivecmI2Givens, pars)
	obj.pars = pars
	ll = logLik(obj)
	#println(Sigma_alpha_beta(obj)[1,1], "\t", ll)
	return ll
end
end