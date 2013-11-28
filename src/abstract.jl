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

## LR test
type LRTest{T<:AbstractCivecm}
	H0::T
	HA::T
	value::Float64
	df::Int64
end

lrtest(obj0::AbstractCivecm, objA::AbstractCivecm, df::Integer) = LRTest(obj0, objA, 2*(loglikelihood(objA) - loglikelihood(obj0)), df)

function bootstrap(obj::LRTest, reps::Integer)
	lrvals = Array(Float64, reps)
	bootH0 = copy(obj.H0)
	bootHA = copy(obj.HA)
	bootHA.endogenous = bootH0.endogenous
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
		bootH0.endogenous[:] = simulate(bootVAR, bootResiduals)
		auxilliaryMatrices(bootH0)
		auxilliaryMatrices(bootHA)
		estimate(bootH0)
		estimate(bootHA)
		lrvals[i] = -2*(loglikelihood(bootH0) - loglikelihood(bootHA))
	end
	return lrvals
end