type CivecmI1 <: AbstractCivecm
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	α::Matrix{Float64}
	β::Matrix{Float64}
	Γ::Matrix{Float64}
	Hα::Matrix
	Hβ::Matrix
	hβ::Vector
	llConvCrit::Float64
	maxiter::Int64
	verbose::Bool
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
				   eye(p*rank),
				   eye(p1*rank),
				   zeros(p1*rank), 
				   1.0e-8, 
				   5000, 
				   false,
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

β(obj::CivecmI1) = obj.β

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
							   copy(obj.Hα),
							   copy(obj.Hβ),
							   copy(obj.hβ),
							   obj.llConvCrit,
							   obj.maxiter,
							   obj.verbose,
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
	Base.print_matrix(io, obj.α*obj.β')
end

function setrank(obj::CivecmI1, rank::Int64)
	obj.α = Array(Float64, size(obj.R0, 2), rank)
	obj.β = Array(Float64, size(obj.R1, 2), rank)
	obj.Hα = eye(size(obj.R0, 2)*rank)
	obj.Hβ = eye(size(obj.R1, 2)*rank)
	obj.hβ = zeros(size(obj.R1, 2)*rank)
	return estimateEigen(obj)
end

function estimate(obj::CivecmI1; method = :switch)
	if method == :switch || method == :Boswijk
		return estimateSwitch(obj)
	elseif method == :eigen
		return estimateEigen(obj)
	else
		error("no such method")
	end
end

function estimateEigen(obj::CivecmI1)
	obj.α[:], svdvals, obj.β[:] = rrr(obj.R0, obj.R1, size(obj.α, 2))
	obj.α[:] = obj.α*Diagonal(svdvals)
	obj.Γ[:] = mreg(obj.Z0 - obj.Z1*obj.β*obj.α', obj.Z2)[1]'
	return obj
end

function estimateSwitch(obj::CivecmI1)
	iT = size(obj.Z0, 1)
	S11 = scale!(obj.R1'obj.R1, 1/iT)
	S10 = scale!(obj.R1'obj.R0, 1/iT)
	ll0 = -realmax()
	ll1 = ll0
	for i = 1:obj.maxiter
		OmegaInv = inv(cholfact!(residualvariance(obj)))
		aoas11 = kron(obj.α'OmegaInv*obj.α, S11)
		phi = qrfact!(obj.Hβ'*aoas11*obj.Hβ, pivot=true)\(obj.Hβ'*(vec(S10*OmegaInv*obj.α) - aoas11*obj.hβ))
		obj.β = reshape(obj.Hβ*phi + obj.hβ, size(obj.β)...)
		γ = qrfact!(obj.Hα'kron(OmegaInv, obj.β'S11*obj.β)*obj.Hα, pivot=true)\(obj.Hα'vec(obj.β'S10*OmegaInv))
		obj.α = reshape(obj.Hα*γ, size(obj.α, 2), size(obj.α, 1))'
		ll1 = loglikelihood(obj)
		if obj.verbose @printf("log-likelihood: %f\n", ll1) end
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

