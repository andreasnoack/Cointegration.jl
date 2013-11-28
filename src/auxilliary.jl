function bar!(A::Matrix)
	qrA = qrfact!(A)
	return full(qrA[:Q]) / qrA[:R]'
end
bar(A::Matrix) = bar!(copy(A))

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
	ans = Array(Float64, iT - maximum(lags), ip * size(lags, 1))
	for i in 1:ip
		for j in 1:size(lags, 1)
			for k in 1:size(ans, 1)
				ans[k, i + (j - 1)*ip] = A[k + maximum(lags) - lags[j], i]
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

function normalitytest(res::StridedVecOrMat)
	n = size(res, 1)
	y = res
	y = (y .- mean(y))
	y = isa(y, Vector) ? y/sqrt(dot(y,y)/n) : y/sqrtm(y'y/n)
	rtb1 = mean(y.^3, 1)
	b2 = mean(y.^4, 1)
	z1 = Float64[normalitytestz1(n, t) for t in rtb1]
	z2 = Float64[normalitytestz2(n, rtb1[t]*rtb1[t], b2[t]) for t in 1:length(b2)]
	return NormalityTest(z1.^2 + z2.^2, dot(z1,z1) + dot(z2,z2))
end

function normalitytestz1(n::Integer, rtb1::Real)
	# Skewness
	β = (3*(n*(n + 27) - 70)*(n + 1)*(n + 3))/((n - 2)*(n + 5)*(n + 7)*(n + 9))
	ω2 = -1 + sqrt(2*(β - 1))
	δ = 1/sqrt(0.5log(ω2))
	y = rtb1*sqrt(((ω2 - 1)/2)*((n + 1)*(n + 3))/(6*(n - 2)))
	return δ*log(y + sqrt(y*y + 1))
end

function normalitytestz2(n::Integer, b1::Real, b2::Real)
	# Kurtosis
	δ = (n - 3)*(n + 1)*(n*(n + 15) - 4)
	a = ((n - 2)*(n + 5)*(n + 7)*(n*(n + 27) - 70))/(6δ)
	c = ((n - 7)*(n + 5)*(n + 7)*(n*(n + 2) - 5))/(6δ)
	k = ((n + 5)*(n + 7)*(n*(n*(n + 37) + 11) - 313))/(12δ)
	α = a + b1*c
	χ = (b2 - 1 - b1)*2*k
	return (cbrt(χ/(2α)) - 1 + 1/(9α))*sqrt(9α)
end

# Note. This is different from the difinition used in the literature following Johansen (and T.W. Anderson). Here I define the reduced rank regression decomposition of PI such that PI=α*diag(σ)*β'
function rrr!(Y::Matrix, X::Matrix)
	iT, iX = size(X)
	iY = size(Y, 2)
	if iX == 0 return zeros(iY,0), zeros(0), zeros(iX, 0) end
	svdX = svdfact!(X, true)
	svdY = svdfact!(Y, true)
	svdZ = svdfact!(svdX[:U]'svdY[:U], true)
	Sm1 = zeros(iX)
	index = svdX[:S] .> 10e-9*maximum(X)
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

function student!(X::Matrix)
	m, n = size(X)
	mX = mean(X,1)
	for j = 1:n
		for i = 1:m
			X[i,j] -= mX[j]
		end
	end
	SX = X'X
	for i = 1:m
		X[i,:] = cholfact!((SX - X[i,:]'*X[i,:])/m,:L)[:L]\vec(X[i,:])
	end
	return X
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
