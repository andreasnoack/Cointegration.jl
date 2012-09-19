load("sparse.jl")

# My 'fixes' to Julia code
function svd{T<:Union(Float64,Float32,Complex128,Complex64)}(A::StridedMatrix{T}, vecs::Int64)
	if vecs != 0
		error("Not supported")
	else
    	Base._jl_lapack_gesvd('S', 'A', copy(A))
    end
end

function eig(A::Matrix, vecs::Bool)
	if vecs
		return eig(A)
	else
		if ishermitian(A) return Base._jl_lapack_syev('N','U',copy(A)) end
                                        # Only compute right eigenvectors
		if iscomplex(A) return Base._jl_lapack_geev('N','N',copy(A))[2:3] end
		VL, WR, WI, VR = Base._jl_lapack_geev('N','N',copy(A))
		if all(WI .== 0.) return WR end
		return complex(WR, WI)
	end
end

# Min Civecm kode

abstract Civecm

logLik(obj::Civecm) = -0.5 * (size(obj.endogenous, 1) - obj.lags) * logdet(residualvariance(obj))

function rrr(Y::Matrix{Float64}, X::Matrix{Float64})
	(iT, iX) = size(X)
	iY = size(Y, 2)
	(Ux, Sx, Vx) = svd(X, 0)
	Uy = svd(Y, 0)[1]
	Uz, Sz = svd(Ux' * Uy, 0)[1:2]
	values = Sz.^2
	Sm1 = zeros(size(values))
	index = Sx .> 10e-9 * max(max(X))
	Sm1[index] = 1 ./ Sx[index]
	vectors = sqrt(iT) * Vx * diagm(Sm1) * Uz
	return (values, vectors)
end

function rrr(Y::Matrix, X::Matrix, rank::Int64)
	(tmp1, tmp2) = rrr(Y, X)
	return (tmp1[1:rank], tmp2[:,1:rank])
end

function bar(matrix::Matrix)
	(Q, R) = qr(matrix)
	return Q / R'
end

function mreg(Y::VecOrMat, X::Matrix)
	try
		coef = X \ Y
		residuals = Y - X*coef
		(coef, residuals)
	catch
		(mU, vS, mV) = svd(X, 0)
		vSinv = zeros(size(vS))
		vSinv[vS .> 0] = 1 ./ vS[vS .> 0]
		coef = mV * diagm(vSinv) * mU'Y
		residuals = Y - X*coef
		(coef, residuals)		
	end
end

logdet(matrix::Matrix) = 2 * sum(log(diag(factors(chol(matrix)))))

function gls(Y::Matrix, X::Matrix, H::Matrix, K::Matrix, k::Vector, Omega::Matrix)
	(iT, px) = size(X)
	r = size(H, 2)
	Sxx = X'X / iT
	Sxy = X'Y / iT
	if sum(size(k)) == 0
	    k = zeros(px * r)
	end
	G = reshape(k, px, r)
	lhs = K'kron(H' / Omega * H, Sxx) * K
	rhs = K'reshape(Sxy / Omega * H - Sxx * G * H' / Omega * H, px * r)
	return reshape(K * (lhs \ rhs) + k, px, r)
end

function lagmatrix(A::Matrix, lags::AbstractArray{Int64, 1})
	(iT, ip) = size(A)
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

function residualvariance(obj::Civecm)
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
	

# I1
type CivecmI1 <: Civecm
	endogenous::Matrix
	exogenous::Matrix
	lags::Int64
	alpha::Matrix
	beta::Matrix
	llConvCrit::Float64
	maxIter::Int64
	Z0::Matrix
	Z1::Matrix
	Z2::Matrix
	R0::Matrix
	R1::Matrix
	eigvals::Vector
	eigvecs::Matrix
end

function CivecmI1(endogenous::Matrix, exogenous::VecOrMat, lags::Int64)
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

	return CivecmI1(endogenous, exogenous, 2, Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), 1.0e-8, 5000, Z0, Z1, Z2, R0, R1, eigvals, eigvecs)
end

function setrank(obj::CivecmI1, rank::Int64)
	return estimateEigen(obj, rank)
end

function estimateEigen(obj::CivecmI1, rank::Int64)
	(obj.eigvals, obj.eigvecs) = rrr(obj.R0, obj.R1, size(obj.R0, 2))
	obj.beta = obj.eigvecs[:, 1:rank]
	obj.alpha = mreg(obj.R0, obj.R1 * obj.beta)[1]
	obj.alpha = obj.alpha'
	return obj
end

function ranktest(obj::CivecmI1)
	return -size(obj.Z0, 1) * reverse(cumsum(reverse(log(1.0 - obj.eigvals))))
end

function ranktest(obj::CivecmI1, reps::Int64)
	tmpTrace = ranktest(obj)
	tmpPVals = zeros(size(tmpTrace))
	rankdist = zeros(reps)
	(iT, ip) = size(obj.endogenous)
	for i = 1:size(tmpTrace, 1)
		for k = 1:reps
			rankdist[k] = I2TraceSimulate(randn(iT, ip - i + 1), ip - i + 1, obj.exogenous)
		end
		tmpPVals[i] = mean(rankdist .> tmpTrace[i])
		println("Simulation of model H(", i, "). ", 100 * i / size(tmpTrace, 1), " percent completed")
	end
	return (tmpTrace, tmpPVals)
end

# I2

type CivecmI2 <: Civecm
	endogenous::Matrix
	exogenous::Matrix
	lags::Int64
	rank::(Int64,Int64)
	alpha::Matrix
	beta::Matrix
	nu::Matrix
	xi::Matrix
	gamma::Matrix
	sigma::Matrix
	llConvCrit::Float64
	maxIter::Int64
	Z0::Matrix
	Z1::Matrix
	Z2::Matrix
	Z3::Matrix
	R0::Matrix
	R1::Matrix
	R2::Matrix
end

function CivecmI2(endogenous::Matrix, exogenous::Matrix, lags::Int64)
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

	return CivecmI2(endogenous, exogenous, lags, (size(endogenous, 2), 0), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), Array(Float64, size(endogenous, 2), size(endogenous, 2)), 1.0e-8, 5000, Z0, Z1, Z2, Z3, R0, R1, R2)
end

CivecmI2(endogenous::Matrix, lags::Int64) = CivecmI2(endogenous, zeros(size(endogenous, 1), 0), lags)

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
		obj.alpha = zeros(ip, obj.rank[1])
		obj.beta = zeros(ip1, obj.rank[1])
		obj.nu = zeros(ip1, obj.rank[1])            
		obj.xi = zeros(ip, obj.rank[2])
		obj.gamma = zeros(ip1, obj.rank[2])
		obj.sigma = zeros(ip, obj.rank[1])
	else
	
		# FixMe! Should be speye when sparse matrices are better supported
		tmp1 = eye(2 * obj.rank[1] + obj.rank[2] + 1, 2 * obj.rank[1] + obj.rank[2])
		tmp2 = tmp1[[reshape(reshape(1:2 * obj.rank[1], obj.rank[1], 2)', 2 * obj.rank[1]), reshape([(2 * obj.rank[1] + obj.rank[2] + 1) * ones(Int, obj.rank[2]) (1:obj.rank[2]) + 2 * obj.rank[1]]', 2 * obj.rank[2]), reshape([(2 * obj.rank[1] + obj.rank[2] + 1) * ones(Int, obj.rank[1]) 1:obj.rank[1]]', 2 * obj.rank[1])], :]
		K = kron(tmp2, eye(size(obj.R1, 2)))
	    
		for k = 1:20
			obj.alpha = randn(ip, obj.rank[1])
			obj.beta = randn(ip1, obj.rank[1])
			obj.nu = randn(ip1, obj.rank[1])            
			obj.xi = randn(ip, obj.rank[2])
			obj.gamma = randn(ip1, obj.rank[2])
			obj.sigma = randn(ip, obj.rank[1])
			mOmega = residualvariance(obj)
	
			# Iterative procedure
			# Initialisation
		
			if obj.rank[1] == 0
				obj.gamma = rrr(obj.R0, obj.R1, obj.rank[2])[2]
				tmpxi = mreg(obj.R0, obj.R1 * obj.gamma)[1]
				obj.xi = tmpxi'
			elseif ip == sum(obj.rank)
				tmpFit = setrank(CivecmI1(obj.endogenous, obj.exogenous, obj.lags), obj.rank[1])
				obj.alpha = tmpFit.alpha
				obj.beta = tmpFit.beta
				tmpFit2 = mreg(obj.R0, [obj.R1 obj.R2 * obj.beta])[1]
				tmpGam = tmpFit2[1:ip1,:]'
				obj.sigma = eye(ip, obj.rank[1])
				obj.nu = (obj.alpha \ (tmpGam - obj.sigma * obj.beta'))'
			else
				# m = min(obj.rank[1], ip - sum(obj.rank));
			 	# tmpFit = CivecmI2alt(obj.endogenous, obj.exogenous, obj.lags).setrank([obj.rank[1] - m, obj.rank[2] + 2 * m]);
				# [obj.alpha, obj.beta, obj.nu, obj.xi, obj.gamma, obj.sigma] = CivecmI2alt.initialPars(tmpFit.alpha, tmpFit.beta, tmpFit.nu, tmpFit.xi, tmpFit.gamma, tmpFit.sigma, 1, m)
			end
			ll = -1.0e9
			ll0 = ll
			for j = 0:obj.maxIter
				# The AC-step
				# println(obj.beta)
				tmpCoef = mreg(obj.R0, [obj.R2 * obj.beta + obj.R1 * obj.nu obj.R1 * obj.gamma obj.R1 * obj.beta])[1]
				obj.alpha = tmpCoef[1:obj.rank[1], :]'
				obj.xi = tmpCoef[obj.rank[1] + 1:sum(obj.rank),:]'
				obj.sigma = tmpCoef[sum(obj.rank) + 1:end, :]'
				# The CC Step ala Rocco
				tmpCoef = gls(obj.R0, [obj.R2 obj.R1], [obj.alpha obj.xi obj.sigma], K, zeros(size(K, 1)), mOmega)
				obj.nu = tmpCoef[ip1 + 1:, 1:obj.rank[1]]
				obj.gamma = tmpCoef[ip1 + 1:, obj.rank[1] + 1:sum(obj.rank)]
				obj.beta = tmpCoef[ip1 + 1:, sum(obj.rank) + 1:]
	
				# Residual variable step
				ll0 = ll
				mOmega = residualvariance(obj)
				ll = -0.5 * iT * logdet(mOmega)
				if abs(ll - ll0) < obj.llConvCrit
				    print("Convergence in ", j, " iterations.\n")
				    break
				end
			end
			if abs(ll - ll0) < obj.llConvCrit
				break
			end
			print("Om igen!")
		end
	end
	return obj
end

function ranktest(obj::CivecmI2)
	ip = size(obj.endogenous, 2)
	ll0 = logLik(setrank(obj, (ip, 0)))
	tmpTrace = zeros(ip, ip + 1)
	for i = 0:ip - 1
		for j = 0:ip - i
			tmpTrace[i + 1, i + j + 1] = 2 * (ll0 - logLik(setrank(obj, (i, j))))
		end
	end
	tmpTrace
end

function ranktest(obj::CivecmI2, reps::Int64)
	vals = ranktest(obj)
	pvals = ranktestpvalues(obj, vals, reps)
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
			print("Simulation of model H(", i, ",", j, "). ", 100 * (0.5 * i * (i + 1) + i * (ip - i + 1) + j + 1) / (0.5 * ip^2 + 1.5 * ip), " percent completed\r")
		end
	end
	return pvals
end

function residuals(obj::CivecmI2)
    res = obj.R0 - obj.R2 * obj.beta * obj.alpha' - obj.R1 * [obj.nu obj.gamma obj.beta] * [obj.alpha obj.xi obj.sigma]'
    return res
end

function residualvariance(obj::CivecmI1)
	return "Hej"
end

# Simulation of rank test

function fS(dX::Matrix{Float64}, Y::Matrix{Float64}, dZ::Matrix{Float64})
	A = dX[2:,:]::Matrix{Float64}
	B = Y[1:size(Y, 1) - 1,:]::Matrix{Float64}
	C = dZ[2:,:]::Matrix{Float64}
	return (A'B)*((B'B)\(B'C))
end

function I2TraceSimulate(eps::Matrix{Float64}, s::Int64, exo::Matrix{Float64})	
    iT = size(eps, 1)
    w = cumsum(eps) / sqrt(iT)
    w2i = cumsum(w[:,s + 1:]) / iT

    m1 = [w[:,1:s] w2i exo]
    m2 = [w[:,s + 1:] diff([zeros(1,size(exo, 2));exo])]

    if size(m2, 2) > 0
    	tmpCoef = m2 \ m1
    	g = m1 - m2 * tmpCoef
    else
    	g = m1
    end
    epsOrth = eps / factors(chol(eps'eps / iT))
    tmp1 = eig(fS(epsOrth, g, epsOrth) / iT, false)
    if size(eps, 2) > s
    	tmp2 = eig(fS(epsOrth[:,s + 1:], m2, epsOrth[:,s + 1:]) / iT, false)
    else
    	tmp2 = 0.0
    end
    return (-iT * (sum(log(1.0 - tmp1)) + sum(log(1.0 - tmp2))))::Float64
end
