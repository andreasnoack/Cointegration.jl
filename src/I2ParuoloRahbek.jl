struct I2Data
    endogenous::Matrix{Float64}
    exogenous::Matrix{Float64}
    unrestricted::Matrix{Float64}
    lags::Int64
    Z0::Matrix{Float64}
    Z1::Matrix{Float64}
    Z2::Matrix{Float64}
    Z3::Matrix{Float64}
    R0::Matrix{Float64}
    R1::Matrix{Float64}
    R2::Matrix{Float64}
end

function I2Data(
    endogenous::Matrix{Float64},
    exogenous::Matrix{Float64},
    unrestricted::Matrix{Float64},
    lags::Int64
)
    ss, p = size(endogenous)
    iT = ss - lags
    pexo = size(exogenous, 2)
    punres = size(unrestricted, 2)
    p1 = p + pexo

    Z0 = Matrix{Float64}(undef, iT, p)
    Z1 = Matrix{Float64}(undef, iT, p1)
    Z2 = Matrix{Float64}(undef, iT, p1)
    Z3 = Matrix{Float64}(undef, iT, p * (lags - 2) + pexo * (lags - 1) + punres)
    R0 = Matrix{Float64}(undef, iT, p)
    R1 = Matrix{Float64}(undef, iT, p1)
    R2 = Matrix{Float64}(undef, iT, p1)

    # Endogenous variables
    for j = 1:p
        for i = 1:iT
            Z0[i, j] =
                endogenous[i+lags, j] - 2endogenous[i+lags-1, j] +
                endogenous[i+lags-2, j]
            Z1[i, j] = endogenous[i+lags-1, j] - endogenous[i+lags-2, j]
            Z2[i, j] = endogenous[i+lags-1, j]
        end
    end
    for k = 1:lags-2
        for j = 1:p
            for i = 1:iT
                Z3[i, p*(k-1)+j] =
                    endogenous[i+lags-k, j] - 2endogenous[i+lags-k-1, j] +
                    endogenous[i+lags-k-2, j]
            end
        end
    end

    # Exogenous variables
    for j = 1:pexo
        for i = 1:iT
            Z1[i, p+j] = exogenous[i+lags-1, j] - exogenous[i+lags-2, j]
            Z2[i, p+j] = exogenous[i+lags-1, j]
            Z3[i, p*(lags-2)+j] =
                exogenous[i+lags, j] - 2exogenous[i+lags-1, j] +
                exogenous[i+lags-2, j]
        end
    end
    for k = 1:lags-2
        for j = 1:pexo
            for i = 1:iT
                Z3[i, p*(lags-2)+pexo*k+j] =
                    exogenous[i+lags-k, j] - 2exogenous[i+lags-k-1, j] +
                    exogenous[i+lags-k-2, j]
            end
        end
    end

    # Unrestriced variables
    Z3[:, (end-punres+1):end] = unrestricted[(lags+1):end, :]

    if size(Z3, 2) > 0 && norm(Z3) > size(Z3, 1) * eps()
        R0[:] = mreg(Z0, Z3)[2]
        R1[:] = mreg(Z1, Z3)[2]
        R2[:] = mreg(Z2, Z3)[2]
    else
        R0[:] = Z0
        R1[:] = Z1
        R2[:] = Z2
    end
    return I2Data(
        endogenous,
        exogenous,
        unrestricted,
        lags,
        Z0,
        Z1,
        Z2,
        Z3,
        R0,
        R1,
        R2
    )
end

struct CivecmI2 <: AbstractCivecm
    data::I2Data
    rankI1::Int64
    rankI2::Int64
    α::Matrix{Float64}
    ρδ::Matrix{Float64}
    Hρδ::Matrix{Float64}
    hρδ::Vector{Float64}
    τ::Matrix{Float64}
    Hτ::Matrix{Float64}
    hτ::Vector{Float64}
    τ⊥::Matrix{Float64}
    ζt::Matrix{Float64}
    llConvCrit::Float64
    maxiter::Int64
    convCount::Ref{Int64}
    method::String
    verbose::Bool
end

function civecmI2(
    endogenous::Matrix{Float64};
    exogenous::Matrix{Float64} = Matrix{Float64}(undef, size(endogenous, 1), 0),
    unrestricted::Matrix{Float64} = zeros(size(endogenous, 1), 0),
    lags::Int64 = 2,
    rankI1::Int64 = size(endogenous, 2),
    rankI2::Int64 = 0,
)
    data = I2Data(
        endogenous,
        exogenous,
        unrestricted,
        lags,
    )

    return setrank(data, size(endogenous, 2), 0)
end

copy(obj::CivecmI2) = CivecmI2(
    copy(obj.data),
    obj.rankI1,
    obj.rankI2,
    copy(obj.α),
    copy(obj.ρδ),
    copy(obj.Hρδ),
    copy(obj.hρδ),
    copy(obj.τ),
    copy(obj.Hτ),
    copy(obj.hτ),
    copy(obj.τ⊥),
    copy(obj.ζt),
    obj.llConvCrit,
    obj.maxiter,
    obj.convCount,
    obj.method,
    obj.verbose,
)

function setrank(data::I2Data, rankI1::Int64, rankI2::Int64)
    if rankI1 + rankI2 > size(data.endogenous, 2)
        error("Illegal choice of rank")
    end
    return estimateτSwitch(data, rankI1, rankI2, 5000, 1e-8, false)
end

setrank(obj::CivecmI2, rankI1::Int64, rankI2::Int64) =
    setrank(obj.data, rankI1, rankI2)

function estimateτSwitch(
    data::I2Data,
    rankI1::Int,
    rankI2::Int,
    maxiter::Int,
    llConvCrit::Float64,
    verbose::Bool,
)
    # Timer
    tt = time()

    # Dimentions
    p = size(data.R0, 2)
    iT, p1 = size(data.R2)
    rs = rankI1 + rankI2

    # Result matrices
    α = Matrix{Float64}(undef, p, rankI1)
    ρδ = Matrix{Float64}(undef, p1, rankI1)
    Hρδ = Matrix{Float64}(I, p1 * rankI1, p1 * rankI1)
    hρδ = zeros(p1 * rankI1)
    τ = Matrix{Float64}(undef, p1, rankI1 + rankI2)
    Hτ = Matrix{Float64}(I, p1 * (rankI1 + rankI2), p1 * (rankI1 + rankI2))
    hτ = zeros(p1 * (rankI1 + rankI2))
    τ⊥ = Matrix{Float64}(undef, p1, p1 - rankI1 - rankI2)
    ζt = Matrix{Float64}(undef, rankI1 + rankI2, p)

    # Temporary matrices
    Rτ = Matrix{Float64}(undef, iT, p1)
    R1τ = Matrix{Float64}(undef, iT, rs)
    workX = Matrix{Float64}(undef, rs, p1)
    mX = Matrix{Float64}(undef, iT, p1)
    workY = Matrix{Float64}(undef, rs, p)
    mY = Matrix{Float64}(undef, iT, p)
    α⊥ = Matrix{Float64}(undef, p, p - rankI1)
    workRRR = Vector{Float64}(undef, rankI1)
    ρ = view(ρδ, 1:rs, 1:rankI1)
    δ = view(ρδ, rs+1:p1, 1:rankI1)
    φ_τ = Vector{Float64}(undef, size(Hτ, 2))
    res = Matrix{Float64}(undef, iT, p)
    Ω = Matrix{Float64}(I, p, p)

    # Choose initial values from two step estimation procedure
    estimate2step!(
        data,
        rankI1,
        rankI2,
        α,
        ρδ,
        τ,
        τ⊥,
        ζt
    )

    # Algorithm
    ll = -floatmax()
    ll0 = -floatmax()
    j = 1
    local convCount
    for j = 1:maxiter
        if verbose
            time() - tt > 1 && println("\nIteration:", j)
        end
        τ⊥[:] = nullspace(τ')[:, 1:p1-rankI1-rankI2]
        Rτ[:, 1:rs] = data.R2 * τ
        Rτ[:, rs+1:end] = data.R1 * τ⊥
        R1τ[:] = data.R1 * τ
        workX[:], mX[:] = mreg(Rτ, R1τ)
        workY[:], mY[:] = mreg(data.R0, R1τ)
        if j == 1
            # Initiate parameters
            α[:], workRRR[:], ρδ[:] = rrr(mY, mX, rankI1)
            rmul!(α, Diagonal(workRRR))
            ζt[:], res[:] = mreg(data.R0 - Rτ * ρδ * α', R1τ)
            Ω = res'res / iT
            if verbose
                # if time() - tt > 1
                # println("\nτ:\n", τ)
                println("ll:", loglikelihood(obj))
                # end
            end
        else
            switch!(
                mY,
                mX,
                ρδ,
                α,
                Ω,
                Hρδ,
                hρδ,
                maxiter = maxiter,
                xtol = llConvCrit,
            )
            ζt .= R1τ \ (data.R0 - Rτ * ρδ * α')
        end
        # ll = loglikelihood(obj)
        if verbose
            if time() - tt > 1
                # println("\nτ:\n", τ)
                println("Right after switching given τ\nll:", ll)
            end
        end
        if ll - ll0 < -llConvCrit
            println("Old likelihood: $(ll0)\nNew likelihood: $(ll)\nIteration: $(j)")
            error("Likelihood cannot decrease")
        elseif abs(ll - ll0) < llConvCrit && j > 1 # Use abs to avoid spurious stops due to noise
            verbose && @printf("Convergence in %d iterations.\n", j - 1)
            convCount = j
            break
        end
        if isnan(ll)
            @warn "nans in loglikehood. Aborting!"
            convCount = maxiter
            break
        end
        ll0 = ll
        α⊥ = nullspace(α')[:, 1:p-rankI1]
        κ = ζt * α⊥
        Ωα = Ω \ α
        ψ = τ⊥ * δ + τ * ζt * (Ωα / (α' * Ωα))
        sqrtΩ = sqrt(Ω)
        tmpX = kron(sqrtΩ \ α * ρ', data.R2) + kron(pinv(sqrtΩ * α⊥)'κ', data.R1)
        φ_τ[:] =
            (tmpX * Hτ) \ (vec((data.R0 - data.R1 * ψ * α') / sqrtΩ) - tmpX * hτ)
        τ[:] = Hτ * φ_τ + hτ

        myres =
            data.R0 - data.R2 * τ * ρ * α' -
            data.R1 * (ψ * α' + τ * κ * ((α⊥'Ω * α⊥) \ α⊥'Ω))
        ll = loglikelihood(myres)
        if verbose
            if time() - tt > 1
                # println("\nτ:\n", τ)
                println("Rigth after estimation of τ\nll:", ll)
                tt = time()
            end
        end
        if ll - ll0 < -llConvCrit
            println("Old likelihood: $(ll0)\nNew likelihood: $(ll)\nIteration: $(j)")
            error("Likelihood cannot decrease")
        elseif abs(ll - ll0) < llConvCrit # Use abs to avoid spurious stops due to noise
            verbose && @printf("Convergence in %d iterations.\n", j - 1)
            convCount = j
            break
        end
        if isnan(ll)
            @warn "nans in loglikehood. Aborting!"
            convCount = maxiter
            break
        end
        ll0 = ll
    end
    return CivecmI2(
        data,
        rankI1,
        rankI2,
        α,
        ρδ,
        Hρδ,
        hρδ,
        τ,
        Hτ,
        hτ,
        τ⊥,
        ζt,
        llConvCrit,
        maxiter,
        Ref(convCount),
        "ParuoloRahbek",
        verbose
    )
end

function estimate2step!(
    data::I2Data,
    rankI1::Int,
    rankI2::Int,
    α::Matrix{Float64},
    ρδ::Matrix{Float64},
    τ::Matrix{Float64},
    τ⊥::Matrix{Float64},
    ζt::Matrix{Float64}
)
    _, Res0 = mreg(data.R0, data.R1)
    _, Res1 = mreg(data.R2, data.R1)
    α, vals, β = rrr(Res0, Res1, rankI1)
    rmul!(α, Diagonal(vals))
    Γt = data.R1 \ (data.R0 - data.R2 * β * α')
    β⊥ = nullspace(β')
    ξ, vals, η =
        rrr((data.R0 - data.R1 * β * bar(β)'Γt) * nullspace(α'), data.R1 * β⊥, rankI2)
    ξ *= Diagonal(vals)
    τ[:, 1:rankI1] = β
    τ[:, rankI1+1:end] = β⊥ * η
    τ⊥[:] = β⊥ * nullspace(η')
    ρδ[1:rankI1+rankI2, :] =
        Matrix{Float64}(I, rankI1 + rankI2, rankI1)
    ρδ[rankI1+rankI2+1:end, :] = bar(β⊥ * nullspace(η'))'Γt * bar(α)
    ζt[1:rankI1, :] = bar(β)'Γt
    ζt[rankI1+1:end, :] = bar(β⊥ * η)'Γt
    return nothing
end

function ranktest(obj::CivecmI2)
    ip = size(obj.data.endogenous, 2)
    r, s = obj.rankI1, obj.rankI2
    ll0 = loglikelihood(setrank(obj, ip, 0))
    tmpTrace = zeros(ip, ip + 1)
    for i = 0:ip-1
        for j = 0:ip-i
            obj.verbose && println("r=$(i), s=$(j)")
            tmpTrace[i+1, i+j+1] = 2 * (ll0 - loglikelihood(setrank(obj, i, j)))
        end
    end
    setrank(obj, r, s)
    tmpTrace
end

function ranktest(rng::AbstractRNG, obj::CivecmI2, reps::Int64)
    vals = ranktest(obj)
    pvals = ranktestPvaluesSimluateAsymp(rng, obj, vals, reps)
    return (vals, pvals)
end
ranktest(obj::CivecmI2, reps::Int64) = ranktest(Random.default_rng(), obj, reps)

function ranktestPvaluesSimluateAsymp(
    rng::AbstractRNG,
    obj::CivecmI2,
    testvalues::Matrix,
    reps::Int64,
)
    (iT, ip) = size(obj.data.endogenous)
    pvals = zeros(ip, ip + 1)
    rankdist = zeros(reps)

    # Handling the progress bar
    prgr = Progress(ip * (ip + 1) >> 1; dt = 0.5, desc = "Simulating I(2) rank test...")
    for i = 0:ip-1
        for j = 0:ip-i
            next!(prgr)
            for k = 1:reps
                rankdist[k] = I2TraceSimulate(randn(rng, iT, ip - i), j, obj.data.exogenous)
            end
            pvals[i+1, i+j+1] = mean(rankdist .> testvalues[i+1, i+j+1])
        end
    end
    return pvals
end
function ranktestPvaluesBootstrap(obj::CivecmI2, testvalues::Matrix, reps::Int64)
    iT, p = size(obj.data.R0)
    r, s = obj.rankI1, obj.rankI2
    bootobj = copy(obj)
    objres = Matrix{Float64}(undef, iT, p)
    workres = Matrix{Float64}(undef, iT, p)
    mm = Vector{Float64}(undef, p)
    bootbool = BitArray(reps)
    pvals = zeros(p, p + 1)
    for i = 0:p-1
        for j = 0:p-i
            objvar = convert(VAR, setrank(obj, i, j))
            objres[:] = residuals(obj)
            mm[:] = mean(objres, 1)
            tmpval = testvalues[i+1, i+j+1]
            for k = 1:reps
                bn = randn(iT)
                for l = 1:p
                    for m = 1:iT
                        workres[m, l] = (objres[m, l] - mm[l]) * bn[m] + mm[l]
                    end
                end
                bootobj.endogenous[:] = simulate(objvar, workres)
                auxilliaryMatrices(bootobj)
                bootbool[k] =
                    2 * (
                        loglikelihood(setrank(bootobj, p, 0)) -
                        loglikelihood(setrank(bootobj, i, j))
                    ) > tmpval
            end
            pvals[i+1, j+i+1] = mean(bootbool)
        end
    end
    setrank(obj, r, s)
    return pvals
end

function residuals(obj::CivecmI2)
    res =
        obj.data.R0 - obj.data.R2 * obj.τ * ρ(obj) * obj.α' - obj.data.R1 * obj.τ⊥ * δ(obj) * obj.α' -
        obj.data.R1 * obj.τ * obj.ζt
    return res
end

function show(io::IO, ::MIME"text/plain", obj::CivecmI2)
    println(io, "β':")
    show(io, MIME"text/plain"(), copy(β(obj)'))

    println(io, "\n\nτ⊥δ:")
    show(IOContext(io, :compact => true), MIME"text/plain"(), obj.τ⊥ * δ(obj)) #'

    println(io, "\n\nτ':")
    show(io, MIME"text/plain"(), copy(τ(obj)'))
end

# Coefficients
α(obj::CivecmI2) = fit.α
β(obj::CivecmI2) = τ(obj) * ρ(obj)
τ(obj::CivecmI2) = obj.τ
ρ(obj::CivecmI2) = obj.ρδ[1:obj.rankI1+obj.rankI2, :]
δ(obj::CivecmI2) = obj.ρδ[obj.rankI1+obj.rankI2+1:end, :]
