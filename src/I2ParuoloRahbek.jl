type CivecmI2 <: AbstractCivecm
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	lags::Int64
	rankI1::Int64
	rankI2::Int64
	α::Matrix{Float64}
	ρδ::Matrix{Float64}
	Hρδ::Matrix{Float64}
	τ::Matrix{Float64}
	Hτ::Matrix{Float64}
	hτ::Vector{Float64}
	τ⊥::Matrix{Float64}
	ζt::Matrix{Float64}
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
					eye(p1),
					Array(Float64, p1, rankI1 + rankI2),
					eye(p1*(rankI1 + rankI2)),
					zeros(p1*(rankI1 + rankI2)),
					Array(Float64, p1, p1 - rankI1 - rankI2),
					Array(Float64, rankI1 + rankI2, p),
					1.0e-8, 
					5000, 
					"ParuoloRahbek",
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
							   copy(obj.ρδ),
							   copy(obj.Hρδ),
							   copy(obj.τ),
							   copy(obj.Hτ), 
							   copy(obj.hτ),
							   copy(obj.τ⊥),							   
							   copy(obj.ζt),
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
		obj.α 	= Array(Float64, p, rankI1)
		obj.ρδ 	= Array(Float64, p1, rankI1)
		obj.Hρδ = eye(p1)
		obj.τ 	= Array(Float64, p1, rankI1 + rankI2)
		obj.Hτ 	= eye(p1*(rankI1 + rankI2))
		obj.hτ 	= zeros(p1*(rankI1 + rankI2))
		obj.τ⊥ 	= Array(Float64, p1, p1 - rankI1 - rankI2)		
		obj.ζt 	= Array(Float64, rankI1 + rankI2, p)
	end
	return estimate(obj)
end

function estimate(obj::CivecmI2)
	if obj.method == "ParuoloRahbek" 
		# if obj.rankI1 == 0 || obj.rankI2 == size(obj.R0,2) - obj.rankI1 return estimate2step(obj) end
		return estimateτSwitch(obj)
	end
	error("No method named %obj.method")
end

# function estimateSwitch(obj::CivecmI2)
# 	iT, ip = size(obj.R0)
# 	ip1 = size(obj.R1, 2)

# 	if max(obj.rankI1, obj.rankI2) == 0
# 		obj.α[:] 	= zeros(ip, obj.rankI1)
# 		obj.β[:] 	= zeros(ip1, obj.rankI1)
# 		obj.ν[:] 	= zeros(ip1, obj.rankI1)            
# 		obj.ξ[:] 	= zeros(ip, obj.rankI2)
# 		obj.γ[:] 	= zeros(ip1, obj.rankI2)
# 		obj.σ[:] 	= zeros(ip, obj.rankI1)
# 	elseif obj.rankI1 == ip
# 		tmpFit 		= setrank(civecmI1(obj.endogenous, obj.exogenous, obj.lags), obj.rankI1)
# 		obj.α[:] 	= tmpFit.α
# 		obj.β[:] 	= tmpFit.β
# 		obj.σ[:] 	= eye(ip, obj.rankI1)
# 		tmpFit2 	= [obj.R1 obj.R2 * obj.β] \ obj.R0
# 		tmpGam 		= tmpFit2[1:ip1,:]'
# 		obj.ν[:] 	= (obj.α \ (tmpGam - obj.σ * obj.β'))'
# 		obj.ξ[:] 	= zeros(ip, 0)
# 		obj.γ[:] 	= zeros(ip1, 0)
# 	else
# 		tmp1 = speye(2 * obj.rankI1 + obj.rankI2 + 1, 2 * obj.rankI1 + obj.rankI2)
# 		tmp2 = copy(tmp1[[reshape(reshape(1:2 * obj.rankI1, obj.rankI1, 2)', 2 * obj.rankI1), reshape([(2 * obj.rankI1 + obj.rankI2 + 1) * ones(Int, obj.rankI2) (1:obj.rankI2) + 2 * obj.rankI1]', 2 * obj.rankI2), reshape([(2 * obj.rankI1 + obj.rankI2 + 1) * ones(Int, obj.rankI1) 1:obj.rankI1]', 2 * obj.rankI1)], :])
# 		K = kron(tmp2, speye(size(obj.R1, 2)))
	    
# 	    R2R1 = [obj.R2 obj.R1]
# 	   	tmpX = [obj.R2 * obj.β + obj.R1 * obj.ν obj.R1 * obj.γ obj.R1 * obj.β]
# 		tmpCoef = tmpX \ obj.R0
# 		mOmega = eye(ip)
# 		tmpGLS = gls(obj.R0, [obj.R2 obj.R1], [obj.α obj.ξ obj.σ], K, [], mOmega)
# 		for k = 1:20
# 			# Iterative procedure
# 			# Initialisation
# 			if obj.rankI1 == 0
# 				obj.α[:] 	= zeros(ip, 0)
# 				obj.β[:] 	= zeros(ip1, 0)
# 				obj.ν[:] 		= zeros(ip1, 0)
# 				obj.γ[:] 	= rrr(obj.R0, obj.R1, obj.rankI2)[3]
# 				obj.ξ[:] 		= ((obj.R1 * obj.γ) \ obj.R0)' #'
# 				obj.σ[:] 	= zeros(ip, 0)
# 				mOmega = residualvariance(obj)
# 			else
# 			#if ip == sum(obj.rank)
# 				if k == 1
# 					tmpFit = setrank(civecmI1(obj.endogenous, obj.exogenous, obj.lags), obj.rankI1+ obj.rankI2)
# 					obj.α[:] 	= tmpFit.α[:,1:obj.rankI1]
# 					obj.β[:] 	= tmpFit.β[:,1:obj.rankI1]
# 					# tmpFit2 	= mreg(obj.R0, [obj.R1 obj.R2 * obj.β])[1]
# 					# tmpGam 		= copy(tmpFit2[1:ip1,:])'
# 					obj.γ[:] 	= tmpFit.β[:,obj.rankI1+1:obj.rankI1+obj.rankI2]
# 					# obj.σ 	= eye(ip, obj.rankI1)
# 					obj.σ[:]	= Array(Float64, ip, obj.rankI1)
# 					obj.ν[:]	= full(qrfact!([obj.β obj.γ])[:Q], false)[:,end-obj.rankI1+1:end]
# 					# obj.ν 		= (obj.α \ (tmpGam - obj.σ * obj.β'))'
# 					obj.ξ[:] 	= Array(Float64, ip, obj.rankI2)
# 					mOmega 		= residualvariance(obj)
# 				else
# 					obj.β[:] 	= randn(ip1, obj.rankI1)
# 					obj.ν[:] 	= randn(ip1, obj.rankI1)            
# 					obj.γ[:] 	= randn(ip1, obj.rankI2)
# 					mOmega 		= residualvariance(obj)
# 				end
# 			end
# 			# else
# 				# m = min(obj.rankI1, ip - obj.rankI1 - obj.rankI2);
# 			 	# tmpFit = CivecmI2alt(obj.endogenous, obj.exogenous, obj.lags).setrank([obj.rankI1 - m, obj.rankI2 + 2 * m]);
# 				# [obj.α, obj.β, obj.ν, obj.ξ, obj.γ, obj.σ] = CivecmI2alt.initialPars(tmpFit.α, tmpFit.β, tmpFit.ν, tmpFit.ξ, tmpFit.γ, tmpFit.σ, 1, m)
			

# 			ll = -1.0e9
# 			ll0 = ll

# 			for j = 0:obj.maxiter
# 				# The AC-step
# 				# println(obj.β)
# 				tmpX[:] 		= [obj.R2 * obj.β + obj.R1 * obj.ν obj.R1 * obj.γ obj.R1 * obj.β]
# 				tmpCoef[:] 		= qrpfact!(tmpX'tmpX) \ (tmpX'obj.R0)
# 				obj.α[:] 	= tmpCoef[1:obj.rankI1, :]'
# 				obj.ξ[:] 		= tmpCoef[obj.rankI1 + 1:(obj.rankI1 + obj.rankI2),:]'
# 				obj.σ[:] 	= tmpCoef[obj.rankI1 + obj.rankI2 + 1:end, :]'
# 				# The CC Step ala Rocco
# 				tmpGLS[:] 		= gls(obj.R0, R2R1, [obj.α obj.ξ obj.σ], K, [], mOmega)
# 				obj.ν[:] 		= tmpGLS[ip1 + 1:, 1:obj.rankI1]
# 				obj.γ[:] 	= tmpGLS[ip1 + 1:, obj.rankI1 + 1:obj.rankI1 + obj.rankI2]
# 				obj.β[:] 	= tmpGLS[ip1 + 1:, obj.rankI1 + obj.rankI2 + 1:]
#                 # Residual variable step
#                 ll0 		= ll
#                 mOmega[:]	= residualvariance(obj)
#                 ll 			= -0.5 * logdet(cholfact(mOmega))
#                 # @printf("Average log-likelihood value: %f\n", ll)
#                 if abs(ll - ll0) < obj.llConvCrit
#                 	@printf("Convergence in %d iterations.\n", j - 1)
#                 	break
#                 end
#             end
#             if norm(obj.ν) > 1.0e6; println(obj.α, obj.ν, obj.α*obj.ν'); end
#             if
# 				abs(ll - ll0) < obj.llConvCrit
# 				break
# 			end             
# 			print("Om igen!")
# 		end
# 	end
# 	return obj
# end

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
	Rτ 		= Array(Float64, iT, p1)
	R1τ 	= Array(Float64, iT, rs)
	workX 	= Array(Float64, rs, p1)
	mX 		= Array(Float64, iT, p1)
	workY 	= Array(Float64, rs, p)
	mY 		= Array(Float64, iT, p)
	αort 	= Array(Float64, p, p - obj.rankI1)
	workRRR = Array(Float64, obj.rankI1)
	ρ 		= sub(obj.ρδ, 1:rs, 1:obj.rankI1)
	ρort 	= Array(Float64, rs, rs - obj.rankI1)
	δ 		= sub(obj.ρδ, rs+1:p1, 1:obj.rankI1)
	φ_ρδ 	= Array(Float64, size(obj.Hρδ, 2), obj.rankI1)
	φ_τ 	= Array(Float64, size(obj.Hτ, 2))
	ζtαort = Array(Float64, rs, p - obj.rankI1)
	res = Array(Float64, iT, p)
	estimate2step(obj)
	# Ω = Cholesky(eye(p), 'U')
	Ω = Array(Float64, p, p)
	A = Array(Float64, rs, rs)
	B = Array(Float64, p1, p1)
	C = Array(Float64, rs, rs)
	D = Array(Float64, p1, p1)
	E = Array(Float64, p1*rs)
	ABCD = Array(Float64, p1*rs, p1*rs)

	# Algorithm
	ll = -realmax()
	ll0 = ll
	for j = 1:obj.maxiter
		obj.τ⊥[:] = null(obj.τ')[:,1:p1 - obj.rankI1 - obj.rankI2]
		Rτ[:,1:rs] = obj.R2*obj.τ
		Rτ[:,rs+1:end] = obj.R1*obj.τ⊥
		R1τ[:] = obj.R1*obj.τ
		workX[:], mX[:] = mreg(Rτ, R1τ)
		workY[:], mY[:] = mreg(obj.R0, R1τ)
		obj.α[:], workRRR[:], φ_ρδ[:] = rrr(mY, mX*obj.Hρδ, obj.rankI1)
		obj.ρδ[:] = obj.Hρδ*φ_ρδ
		obj.α *= Diagonal(workRRR)
		obj.ζt[:], res[:] = mreg(obj.R0 - Rτ*obj.ρδ*obj.α', R1τ)
		Ω[:] = res'res/iT
		ll = loglikelihood(obj)
		if abs(ll - ll0) < obj.llConvCrit 
			@printf("Convergence in %d iterations.\n", j - 1)
			break 
		end
		ll0 = ll
		# LinAlg.LAPACK.potrf!('U', Ω.UL)
		αort[:] = null(obj.α')
		A[:] = ρ*obj.α'*(Ω\obj.α)*ρ'
		B[:] = S22
		ζtαort[:] = obj.ζt*αort
		C[:] = ζtαort*(cholfact!(αort'Ω*αort)\(ζtαort'))
		D[:] = S11
		E[:] = S20*(Ω\obj.α)*ρ' - S21*(obj.τ⊥*δ*obj.α' + obj.τ*obj.ζt)*(Ω\obj.α)*ρ' + S10*αort*(cholfact!(αort'Ω*αort)\(ζtαort'))
		ABCD[:] = kron(A,B) + kron(C,D)
		φ_τ[:] 	= (ABCD*obj.Hτ)\(E - ABCD*obj.hτ)
		obj.τ[:] = obj.Hτ*φ_τ + obj.hτ
		# obj.τ[:] = obj.τ/real(sqrtm(obj.τ'S22*obj.τ))
		# obj.τ[:] = full(qrfact!(obj.τ)[:Q])
		# obj.τ[:] = obj.τ/sqrtm(Hermitian(obj.τ'S11*obj.τ))
	end
	return obj
end

function estimate2step(obj)
	_, Res0 = mreg(obj.R0, obj.R1)
	_, Res1 = mreg(obj.R2, obj.R1)
	obj.α[:], vals, β = rrr(Res0, Res1, obj.rankI1)
	obj.α *= Diagonal(vals)
	Γt = obj.R1\(obj.R0 - obj.R2*β*obj.α')
	β⊥ = null(β')
	ξ, vals, η = rrr((obj.R0 - obj.R1*β*bar(β)'Γt)*null(obj.α'), obj.R1*β⊥, obj.rankI2)
	ξ *= Diagonal(vals)
	obj.τ[:,1:obj.rankI1] = β
	obj.τ[:,obj.rankI1 + 1:end] = β⊥*η
	obj.τ⊥[:] = β⊥*null(η')
	obj.ρδ[1:obj.rankI1 + obj.rankI2,:] = eye(obj.rankI1 + obj.rankI2, obj.rankI1)
	obj.ρδ[obj.rankI1 + obj.rankI2 + 1:end,:] = bar(β⊥*null(η'))'Γt*bar(obj.α)
	obj.ζt[1:obj.rankI1,:] = bar(β)'Γt
	obj.ζt[obj.rankI1 + 1:end,:] = bar(β⊥*η)'Γt
	return obj
end

function ranktest(obj::CivecmI2)
	ip 			= size(obj.endogenous, 2)
	r, s 		= obj.rankI1, obj.rankI2
	ll0 		= loglikelihood(setrank(obj, ip, 0))
	tmpTrace 	= zeros(ip, ip + 1)
	for i = 0:ip - 1
		for j = 0:ip - i
			tmpTrace[i + 1, i + j + 1] = 2 * (ll0 - loglikelihood(setrank(obj, i, j)))
		end
	end
	setrank(obj, r, s)
	tmpTrace
end

function ranktest(obj::CivecmI2, reps::Int64)
	vals 	= ranktest(obj)
	pvals 	= ranktestPvaluesSimluateAsymp(obj, vals, reps)
	return (vals, pvals)
end

function ranktestPvaluesSimluateAsymp(obj::CivecmI2, testvalues::Matrix, reps::Int64)
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
function ranktestPvaluesBootstrap(obj::CivecmI2, testvalues::Matrix, reps::Int64)
	iT, p = size(obj.R0)
	r, s = obj.rankI1, obj.rankI2
	bootobj = copy(obj)
	objres = Array(Float64, iT, p)
	workres = Array(Float64, iT, p)
	mm = Array(Float64, p)
	bootbool = BitArray(reps)
	pvals = zeros(p,p+1)
	for i = 0:p-1
		for j = 0:p-i
			objvar = convert(VAR, setrank(obj, i, j))
			objres[:] = residuals(obj)
			mm[:] = mean(objres, 1)
			tmpval = testvalues[i+1,i+j+1]
			for k = 1:reps
				bn = randn(iT)
				for l = 1:p
					for m = 1:iT
						workres[m,l] = (objres[m,l] - mm[l])*bn[m] + mm[l]
					end
				end
				bootobj.endogenous[:] = simulate(objvar, workres)
				auxilliaryMatrices(bootobj)
				bootbool[k] = 2*(loglikelihood(setrank(bootobj, p, 0)) - loglikelihood(setrank(bootobj, i, j))) > tmpval
			end
			pvals[i+1,j+i+1] = mean(bootbool)
		end
	end
	setrank(obj, r, s)
	return pvals
end

function residuals(obj::CivecmI2)
    res = obj.R0 - obj.R2*obj.τ*obj.ρδ[1:obj.rankI1 + obj.rankI2,:]*obj.α' - obj.R1*null(obj.τ')*obj.ρδ[obj.rankI1 + obj.rankI2 + 1:end,:]*obj.α' - obj.R1*obj.τ*obj.ζt
    return res
end

function show(io::IO, obj::CivecmI2)
	println("β':")
	println(β(obj)')
	println("τ⊥δ:")
	println(null(τ(obj)')*δ(obj))
	println("τ':")
	println(τ(obj)')	
end

# Coefficients
α(obj::CivecmI2) = fit.α
β(obj::CivecmI2) = τ(obj)*ρ(obj)
τ(obj::CivecmI2) = obj.τ
# τ(obj::CivecmI2) = copy(obj.τ)
ρ(obj::CivecmI2) = obj.ρδ[1:obj.rankI1+obj.rankI2,:]
δ(obj::CivecmI2) = obj.ρδ[obj.rankI1+obj.rankI2+1:end,:]