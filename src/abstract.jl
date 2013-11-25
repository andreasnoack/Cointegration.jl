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

normalitytest(obj::AbstractCivecm) = normalitytest(residuals(obj))

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
