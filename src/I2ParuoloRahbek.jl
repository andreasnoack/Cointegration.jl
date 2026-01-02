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

function convert(::Type{I2Data}, data::I1Data)
    # FIXME! Maybe this can be done more efficiently
    return I2Data(
        data.endogenous,
        data.exogenous,
        data.unrestricted,
        data.lags,
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

function convert(::Type{CivecmI2}, obj::CivecmI1)
    p, rankI1 = size(obj.α)
    p1 = size(obj.β, 1)
    # For I(1) models we can compute
    rankI2 = p - rankI1

    # We have that
    #
    # I(1): ΔXₜ = αβ'Xₜ₋₁ + ΓΔXₜ₋₁
    # I(2): Δ²Xₜ = α*(ρ'τ'Xₜ₋₁ + δ'τ⊥'ΔXₜ₋₁) + ζτ'ΔXₜ₋₁
    #
    # so we can set
    #
    # α = α
    # β = τ*ρ
    # Γ - I = αδ'τ⊥' + ζτ'

    # We can choose τ and ρ freely as long as β = τ*ρ so it is convenient to
    # choose τ to be orthonormal since that allows for easy access to the bar
    # and the bot versions of τ. We can get an orthonormal τ via the QR
    # factorization. This makes ρ triangular instead of the identity which is
    # the choice of Johansen (1997)
    Q, R = qr(obj.β)
    τ = Q[:, 1:(rankI1 + rankI2)] # check if this falls back to scalar indexing
    τ⊥ = Q[:, (rankI1 + rankI2 + 1):end]
    ρ = [R; zeros(rankI2, rankI1)]

    # ᾱ'*(Γ - I)*τ̄⊥ = δ'
    δ = ((obj.Γ[:, 1:p1] - Matrix(I, p, p1)) * τ⊥)' * bar(obj.α)

    # (Γ - I)τ̄ = ζ
    ζ = (obj.Γ[:, 1:p1] - Matrix(I, p, p1))*τ

    # FIXME! The restriction handling needs to be moved to a different place
    Hρδ = Matrix{Float64}(I, p1 * rankI1, p1 * rankI1)
    hρδ = zeros(p1 * rankI1)
    Hτ = Matrix{Float64}(I, p1 * (rankI1 + rankI2), p1 * (rankI1 + rankI2))
    hτ = zeros(p1 * (rankI1 + rankI2))

    return CivecmI2(
        # FIXME! Should this data conversion be avoided?
        convert(I2Data, obj.data),
        rankI1,
        rankI2,
        obj.α,
        [ρ; δ],
        Hρδ,
        hρδ,
        τ,
        Hτ,
        hτ,
        τ⊥,
        ζ',
        # FIXME! This convergence nonsense shouldn't be here
        1e-8,
        5000,
        Ref(0),
        "ReducedRankRegression",
        false
    )
end

function setrank(data::I2Data, rankI1::Int64, rankI2::Int64)

    p = size(data.R0, 2)
    p1 = size(data.R1, 2)

    if rankI1 < 0
        throw(ArgumentError("rankI1 must be positive"))
    end
    if rankI2 < 0
        throw(ArgumentError("rankI2 must be positive"))
    end
    if rankI1 + rankI2 > p
        throw(ArgumentError("rankI1 + rankI2 must be less than p but rankI1=$rankI1, rankI2=$rankI2, and p=$p"))
    end

    if p == rankI1 + rankI2 # the I1 case
        # When p == rankI1 + rankI2 then it is an I(1) model so we estimate with civemcI1 and convert to CivecmI2

        return convert(
            CivecmI2,
            # Use the CivecmI1 constructor when implemented similarly to the CivecmI2 constructor
            civecmI1(
                data.endogenous,
                exogenous = data.exogenous,
                unrestricted = data.unrestricted,
                lags = data.lags,
                rank = rankI1
            )
        )
    elseif rankI1 == 0
        # When rankI1 == 0 then the I(2) model reduces to
        #
        # Δ²Xₜ = ζτ'ΔXₜ₋₁
        #
        # i.e. a reduces rank regression of Δ²Xₜ against ΔXₜ₋₁

        _rrr = rrr(data.R0, data.R1)
        α = Matrix{Float64}(undef, p, 0)
        ρδ = Matrix{Float64}(undef, p1, 0)
        Hρδ = Matrix{Float64}(I, p1 * rankI1, p1 * rankI1)
        hρδ = zeros(p1 * rankI1)
        τ = _rrr.β[:, 1:rankI2]
        Hτ = Matrix{Float64}(I, p1 * (rankI1 + rankI2), p1 * (rankI1 + rankI2))
        hτ = zeros(p1 * (rankI1 + rankI2))
        τ⊥ = nullspace(τ')
        # Our reduces rank regression is an SVD so the singular values should
        # be applies to one of the vector matrices to match the usual statistical
        # regressentation
        ζt = Diagonal(_rrr.s[1:rankI2]) * _rrr.α[:, 1:rankI2]'

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
            NaN,
            0,
            Ref(0),
            "ReducedRankRegression",
            false
        )
    else
        # Originally, this model was estimated based on a two-step procedure
        # see Johansen (ET, 1995). The estimates from this procedure can then
        # be used as starting values for the τ-switching algorithm of Johansen
        # (SJS, 1995). However, an alterntive and often better initialization
        # is based on the submodel with (rankI1 - 1, rankI2 + 2), see my dissertation.

        # It seems like these two difference initialization can results in two
        # local extrema of the log-likelhood for the model (rankI1, rankI2).
        # Hence we estimate based on both set of starting values and pick the 
        # one with the better fit

        # Johansen's initialization
        ft = estimate2step(data, rankI1, rankI2)
        τ₀ = ft.τ
        @debug "Initial value based on two step procedure" τ₀
        fJohansen = estimateτSwitch(data, rankI1, rankI2, τ₀, 5000, 1e-8, true)
        @debug "loglikelihood" loglikelihood(fJohansen)

        # Noack's initialization

        ft = setrank(data, rankI1 - 1, rankI2 + 2)
        τ₀ = [β(ft) ft.τ*(nullspace(ρ(ft)'))[:, 1:(rankI2 + 1)]]
        # τ₀ = [β(ft) ft.τ*(nullspace(ρ(ft)'))[:, 2:end]]
        # τ₀ = [β(ft) ft.τ*(nullspace(ρ(ft)'))*randn(rankI2 + 2, rankI2 + 1)]
        @debug "Initial value based on the model (r - 1, s + 2)" τ₀
        fNoack = estimateτSwitch(data, rankI1, rankI2, τ₀, 5000, 1e-8, true)
        @debug "loglikelihood" loglikelihood(fNoack)

        return loglikelihood(fJohansen) >= loglikelihood(fNoack) ? fJohansen : fNoack
    end
end

setrank(obj::CivecmI2, rankI1::Int64, rankI2::Int64) =
    setrank(obj.data, rankI1, rankI2)

function estimateτSwitch(
    data::I2Data,
    rankI1::Int,
    rankI2::Int,
    τ₀::Matrix{Float64},
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

    # Check sizes
    if size(τ₀) != (p1, rankI1 + rankI2)
        throw(DimensionMismatch("τ₀ should have size $((p1, rankI1 + rankI2)) but had size $(size(τ₀))"))
    end

    # Result matrices
    α = Matrix{Float64}(undef, p, rankI1)
    ρδ = Matrix{Float64}(undef, p1, rankI1)
    Hρδ = Matrix{Float64}(I, p1 * rankI1, p1 * rankI1)
    hρδ = zeros(p1 * rankI1)
    τ = copy(τ₀)
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

    @debug "initial τ" τ

    # Algorithm
    ll = -floatmax()
    ll0 = -floatmax()
    j = 1
    iteration_s = " Iteration: "
    print(iteration_s)
    for j = 1:maxiter
        print(lpad(j, ndigits(maxiter)))
        @debug "Iteration" j
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

        @debug "Rigth after estimation of τ" ll τ

        print(repeat('\b', ndigits(maxiter)))

        if ll - ll0 < -llConvCrit
            @info "Old likelihood: $(ll0)\nNew likelihood: $(ll)\nIteration: $(j)"
            error("Likelihood cannot decrease")
        elseif abs(ll - ll0) < llConvCrit # Use abs to avoid spurious stops due to noise
            @debug "Convergence in iterations:" j - 1 ll ll0 llConvCrit
            break
        end
        if isnan(ll)
            @warn "nans in loglikehood. Aborting!"
            j = maxiter
            break
        end
        ll0 = ll
    end

    print(repeat('\b', length(iteration_s)))

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
        Ref(j),
        "ParuoloRahbek",
        verbose
    )
end

function estimate2step(
    data::I2Data,
    rankI1::Int,
    rankI2::Int
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
    τ = [β β⊥ * η]
    τ⊥ = β⊥ * nullspace(η')
    ρδ = [
        Matrix{Float64}(I, rankI1 + rankI2, rankI1);
        bar(β⊥ * nullspace(η'))'Γt * bar(α)
    ]
    ζt = [
        bar(β)'Γt;
        bar(β⊥ * η)'Γt
    ]
    return CivecmI2(
        data,
        rankI1,
        rankI2,
        α,
        ρδ,
        zeros(0, 0),
        zeros(0),
        τ,
        zeros(0, 0),
        zeros(0),
        τ⊥,
        ζt,
        NaN,
        -1,
        Ref(-1),
        "2Step",
        false
    )
end

struct I2RankTest
    model_dict::Dict{Tuple{Int,Int},CivecmI2}
end

function show(io::IO, ::MIME"text/plain", ranktest::I2RankTest)
    print(io, string(summary(ranktest), "\n\n"))
    p = maximum(first.(keys(ranktest.model_dict)))
    pvals = zeros(p, p + 1)
    llA = loglikelihood(ranktest.model_dict[(p, 0)])
    for i in 0:(p - 1)
        for j in 0:(p - i)
            pvals[i + 1, i + j + 1] = 2*(llA - loglikelihood(ranktest.model_dict[(i, j)]))
        end
    end
    show(io, MIME"text/plain"(), pvals)
end

function ranktest(obj::CivecmI2)
    p = size(obj.data.endogenous, 2)
    trace_dict = Dict((p, 0) => setrank(obj, p, 0))
    for i = 0:p-1
        for j = 0:p-i
            obj.verbose && println("r=$(i), s=$(j)")
            trace_dict[(i, j)] = setrank(obj, i, j)
        end
    end
    return I2RankTest(trace_dict)
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
    testvalues::I2RankTest,
    reps::Int64,
)
    (iT, ip) = size(obj.data.endogenous)
    pvals = zeros(ip, ip + 1)
    rankdist = zeros(reps)
    llA = loglikelihood(testvalues.model_dict[(ip, 0)])

    # Handling the progress bar
    prgr = Progress(ip * (ip + 1) >> 1; dt = 0.5, desc = "Simulating I(2) rank test...")
    for i = 0:ip-1
        for j = 0:ip-i
            next!(prgr)
            for k = 1:reps
                rankdist[k] = I2TraceSimulate(randn(rng, iT, ip - i), j, obj.data.exogenous)
            end
            stat_ij = -2*(loglikelihood(testvalues.model_dict[(i, j)]) - llA)
            pvals[i + 1, i + j + 1] = mean(rankdist .> stat_ij)
            @debug "stat and p-value" rankI1=i rankI2=j median(rankdist) maximum(rankdist) stat_ij mean(rankdist .> stat_ij)
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
