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
