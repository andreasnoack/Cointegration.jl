mutable struct VAR
    endogenous::Matrix{Float64}
    exogenous::Matrix{Float64}
    endocoefs::Array{Float64,3}
    exocoefs::Matrix{Float64}
end

eigen(obj::VAR) = eigen(companion(obj))
eigvals(obj::VAR) = eigvals(companion(obj))

function companion(obj::VAR)
    p = size(obj.endogenous, 2)
    k = size(obj.endocoefs, 3)
    return [reshape(obj.endocoefs, p, p * k); Matrix{Float64}(I, (k - 1) * p, k * p)]
end

function simulate(obj::VAR, innovations::Matrix{Float64}, init::Matrix{Float64})
    iT, p = size(innovations)
    k = size(obj.endocoefs, 3)
    if size(obj.endocoefs, 1) != p
        throw(ArgumentError("Wrong dimensions"))
    end
    if size(init, 2) != p
        throw(ArgumentError("Wrong dimensions"))
    end
    if size(init, 1) != k
        throw(ArgumentError("Wrong dimensions"))
    end
    Y = zeros(iT + k, p)
    Y[1:k, :] = init
    for t = 1:iT
        for i = 1:k
            Y[t+k, :] += (obj.endocoefs[:, :, i] * Y[t+k-i, :] + innovations[t, :])
        end
    end
    return Y
end
simulate(obj::VAR, innovations::Matrix{Float64}) =
    simulate(obj, innovations, obj.endogenous[1:size(obj.endocoefs, 3), :])
simulate(obj::VAR, iT::Integer) = simulate(obj, randn(iT, size(obj.endogenous, 2)))

function convert(::Type{VAR}, obj::CivecmI1)
    p = size(obj.data.endogenous, 2)
    endocoefs = Array{Float64}(undef, p, p, obj.data.lags)
    endocoefs[:, :, 1] = obj.α * obj.β[1:p, :]' + I
    if obj.data.lags > 1
        endocoefs[:, :, 1] += obj.Γ[:, 1:p]
        endocoefs[:, :, obj.data.lags] = -obj.Γ[:, (obj.data.lags-2)*p+1:(obj.data.lags-1)*p]
        for i = 1:obj.data.lags-2
            endocoefs[:, :, i+1] =
                obj.Γ[:, p*(obj.data.lags-i-1)+1:p*(obj.data.lags-i)] -
                obj.Γ[:, p*(obj.data.lags-i-2)+1:p*(obj.data.lags-i-1)]
        end
    end
    return VAR(obj.data.endogenous, obj.data.exogenous, endocoefs, zeros(0, 0))
end

function convert(::Type{VAR}, obj::CivecmI2)
    p = size(obj.data.endogenous, 2)
    endocoefs = Array{Float64}(undef, p, p, obj.data.lags)
    endocoefs[:, :, 1] =
        2I + (obj.α*ρ(obj)'*τ(obj)'+obj.α*δ(obj)'*obj.τ⊥'+obj.ζt'*τ(obj)')[1:p, 1:p]
    endocoefs[:, :, 2] = -I - (obj.α*δ(obj)'*obj.τ⊥'+obj.ζt'*τ(obj)')[1:p, 1:p]
    if obj.data.lags > 2
        Ψ =
            (
                obj.data.Z3 \ (
                    obj.data.Z0 - obj.data.Z2 * obj.τ * ρ(obj) * obj.α' -
                    obj.data.Z1 * (obj.τ⊥ * δ(obj) * obj.α' + obj.τ * obj.ζt)
                )
            )'
        endocoefs[:, :, 1] += Ψ[:, 1:p]
        endocoefs[:, :, obj.data.lags] = -Ψ[:, (obj.data.lags-3)*p+1:(obj.data.lags-2)*p]
        if obj.data.lags > 3
            endocoefs[:, :, 2] += Ψ[:, p+1:2p] - 2 * Ψ[:, 1:p]
            endocoefs[:, :, obj.data.lags-1] =
                Ψ[:, (obj.data.lags-4)*p+1:(obj.data.lags-3)*p] -
                2 * Ψ[:, (obj.data.lags-3)*p+1:(obj.data.lags-2)*p]
        end
        for i = 1:obj.data.lags-4
            endocoefs[:, :, i+2] =
                Ψ[:, p*(i-1)+1:p*i] - 2 * Ψ[:, p*i+1:p*(i+1)] + Ψ[:, p*(i+1)+1:p*(i+2)]
        end
    end
    return VAR(obj.data.endogenous, obj.data.exogenous, endocoefs, zeros(0, 0))
end
