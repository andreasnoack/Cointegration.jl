type VAR
	endogenous::Matrix{Float64}
	exogenous::Matrix{Float64}
	endocoefs::Array{Float64, 3}
	exocoefs::Matrix{Float64}
end

function eigvals(obj::VAR)
	p = size(obj.endogenous, 2)
	k = size(obj.endocoefs, 3)
	return eigvals([reshape(obj.endocoefs, p, p*k); eye((k-1)*p, k*p)])
end

function simulate(obj::VAR, innovations::Matrix{Float64}, init::Matrix{Float64})
	iT, p = size(innovations)
	k = size(obj.endocoefs, 3)
	if size(obj.endocoefs, 1) != p error("Wrong dimensions") end
	if size(init, 2) != p error("Wrong dimensions") end
	if size(init, 1) != k error("Wrong dimensions") end
	Y = zeros(iT + k, p)
	Y[1:k,:] = init
	for t = 1:iT
		for i = 1:k
			Y[t+k,:] += Y[t+k-i,:]*obj.endocoefs[:,:,i]' + innovations[t,:]
		end
	end
	return Y
end
simulate(obj::VAR, innovations::Matrix{Float64}) = simulate(obj, innovations, obj.endogenous[1:size(obj.endocoefs, 3),:])
simulate(obj::VAR, iT::Integer) = simulate(obj, randn(iT, size(obj.endogenous, 2)))

function convert(::Type{VAR}, obj::CivecmI1)
	p = size(obj.endogenous, 2)
	endocoefs = Array(Float64, p, p, obj.lags)
	endocoefs[:,:,1] = obj.α*obj.β' + eye(p)
	if obj.lags > 1
		endocoefs[:,:,1] += obj.Γ[:,1:p]
		endocoefs[:,:,obj.lags] = -obj.Γ[:,(obj.lags-2)*p+1:(obj.lags-1)*p]
		for i = 1:obj.lags - 2
			endocoefs[:,:,i+1] = obj.Γ[:,p*(obj.lags-i-1)+1:p*(obj.lags-i)] - obj.Γ[:,p*(obj.lags-i-2)+1:p*(obj.lags-i-1)]
		end
	end
	return VAR(obj.endogenous, obj.exogenous, endocoefs, zeros(0,0))
end