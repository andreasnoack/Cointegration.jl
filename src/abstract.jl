abstract AbstractCivecm

eigvals(obj::AbstractCivecm) = eigvals(convert(VAR, obj))

loglikelihood(obj::AbstractCivecm) = (nobs = size(endogenous(obj), 1) - obj.lags; -0.5*nobs*(logdet(cholfact!(residualvariance(obj))) - size(endogenous(obj), 2)*(log(nobs) - 1 - log(2π))))
# loglikelihood(A::StridedMatrix) = -0.5*size(A, 1)*(2sum(log(abs(diag(qrfact(A).factors)/sqrt(size(A,1))))) - size(A, 2)*(log(size(A, 1)) - 1 - log(2π)))
loglikelihood(A::StridedMatrix) = -0.5*size(A, 1)*(logdet(cov(A, corrected = false, mean = 0)) - size(A, 2)*(log(size(A, 1)) - 1 - log(2π)))

# aic(obj::Civecm) = 2*(npars(obj) - loglikelihood(obj))

function residualvariance(obj::AbstractCivecm)
    mresiduals = residuals(obj)
    mOmega = mresiduals'mresiduals / size(mresiduals, 1)
    return mOmega
end

normalitytest(obj::AbstractCivecm) = normalitytest(residuals(obj))

## LR test
type LRTest{T<:AbstractCivecm}
	H0::T
	HA::T
	value::Float64
	df::Int64
end

lrtest(obj0::AbstractCivecm, objA::AbstractCivecm, df::Integer) = LRTest(obj0, objA, 2*(loglikelihood(objA) - loglikelihood(obj0)), df)

function bootstrap(obj::LRTest, reps::Integer, simH0 = true)
	lrvals = Array(Float64, reps)
	bootH0 = copy(obj.H0)
	bootHA = copy(obj.HA)
	bootHA.endogenous = bootH0.endogenous
	bootVAR = simH0 ? convert(VAR, bootH0) : convert(VAR, bootHA)
	iT, p = size(obj.H0.Z0)
	# simResiduals = residuals(obj.H0)
	simResiduals = residuals(obj.HA)
	mbr = mean(simResiduals, 1)
	bi = Array(Float64, iT)
	bootResiduals = similar(simResiduals)
	for i = 1:reps
		bi[:] = randn(iT)
		bootResiduals[:] = copy(simResiduals)
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
