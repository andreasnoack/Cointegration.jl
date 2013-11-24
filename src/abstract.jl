abstract AbstractCivecm

eigvals(obj::AbstractCivecm) = eigvals(convert(VAR, obj))

loglikelihood(obj::AbstractCivecm) = -0.5 * (size(obj.endogenous, 1) - obj.lags) * logdet(residualvariance(obj))

# aic(obj::Civecm) = 2*(npars(obj) - loglikelihood(obj))

function residualvariance(obj::AbstractCivecm)
    mresiduals = residuals(obj)
    mOmega = mresiduals'mresiduals / size(mresiduals, 1)
    return mOmega
end

type NormalityTest
	univariate::Vector{Float64}
	multivariate::Float64
end

function normalitytest(obj::AbstractCivecm)
	n = size(obj.R0, 1)
	y = residuals(obj)
	y = (y .- mean(y))
	y = y/sqrtm(y'y/n)
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

function show(io::IO, obj::NormalityTest)
	println("Univarite tests:")
	println("Test values   df   p-values")
	for t in obj.univariate
		@printf("%11.2f%5d%11.2f\n", t, 2, ccdf(Chisq(2), t))
	end
	println("\nMultivariate test:")
	println("Test values   df   p-values")
	@printf("%11.2f%5d%11.2f\n", obj.multivariate, 2*length(obj.univariate), ccdf(Chisq(2*length(obj.univariate)), obj.multivariate))
end
