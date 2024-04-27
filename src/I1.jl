struct CivecmI1 <: AbstractCivecm
    endogenous::Matrix{Float64}
    exogenous::Matrix{Float64}
    unrestricted::Matrix{Float64}
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

function civecmI1(
    endogenous::Matrix{Float64};
    exogenous::Matrix{Float64} = zeros(size(endogenous, 1), 0),
    unrestricted::Matrix{Float64} = zeros(size(endogenous, 1), 0),
    lags::Int64 = 2,
    rank::Int64 = size(endogenous, 2),
)

    ss, p = size(endogenous)
    pexo = size(exogenous, 2)
    punres = size(unrestricted, 2)
    p1 = p + pexo
    iT = ss - lags

    obj = CivecmI1(
        endogenous,
        exogenous,
        unrestricted,
        lags,
        Matrix{Float64}(undef, p, rank),
        Matrix{Float64}(undef, p1, rank),
        Matrix{Float64}(undef, p, (lags - 1) * p + lags * pexo + punres),
        Matrix{Float64}(I, p * rank, p * rank),
        Matrix{Float64}(I, p1 * rank, p1 * rank),
        zeros(p1 * rank),
        1.0e-8,
        5000,
        false,
        Matrix{Float64}(undef, iT, p),
        Matrix{Float64}(undef, iT, p1),
        Matrix{Float64}(undef, iT, (lags - 1) * p + lags * pexo + punres),
        Matrix{Float64}(undef, iT, p),
        Matrix{Float64}(undef, iT, p1),
    )
    auxilliaryMatrices!(obj)
    estimateEigen!(obj)
    return obj
end
# civecmI1(endogenous::Matrix{Float64}, exogenous::Matrix{Float64}, lags::Int64) = civecmI1(endogenous, exogenous, lags, size(endogenous, 2))
# civecmI1(endogenous::Matrix{Float64}, lags::Int64) = civecmI1(endogenous, zeros(size(endogenous, 1), 0), lags)
# civecmI1(endogenous::Matrix{Float64}, exogenous::UnitRange, lags::Int64) = civecmI1(endogenous, float64(reshape(exogenous, length(exogenous), 1)), lags)

function Base.getproperty(obj::CivecmI1, s::Symbol)
    if s === :Π
        return obj.α * obj.β'
    else
        return getfield(obj, s)
    end
end

function auxilliaryMatrices!(obj::CivecmI1)
    iT, p = size(obj.Z0)
    pexo = size(obj.exogenous, 2)
    punres = size(obj.unrestricted, 2)

    # Endogenous variables
    ## k = 1
    for j = 1:p
        for i = 1:iT
            # First differences
            obj.Z0[i, j] = obj.endogenous[i+obj.lags, j] - obj.endogenous[i+obj.lags-1, j]

            # Lagged levels
            obj.Z1[i, j] = obj.endogenous[i+obj.lags-1, j]
        end
    end
    ## Lags (k > 1)
    for k = 1:obj.lags-1
        for j = 1:p
            for i = 1:iT
                # Lagged first differences
                obj.Z2[i, p*(k-1)+j] =
                    obj.endogenous[i+obj.lags-k, j] - obj.endogenous[i+obj.lags-k-1, j]
            end
        end
    end

    # Exogenous variables
    ## k = 1
    for j = 1:pexo
        for i = 1:iT
            # Lagged levels
            obj.Z1[i, p+j] = obj.exogenous[i+obj.lags-1, j]

            # Lagged first differences
            obj.Z2[i, p*(obj.lags-1)+j] =
                obj.exogenous[i+obj.lags, j] - obj.exogenous[i+obj.lags-1, j]
        end
    end
    ## Lags (k > 1)
    for k = 1:obj.lags-1
        for j = 1:pexo
            for i = 1:iT
                # Lagged first differences
                obj.Z2[i, p*(obj.lags-1)+pexo*k+j] =
                    obj.exogenous[i+obj.lags-k, j] - obj.exogenous[i+obj.lags-k-1, j]
            end
        end
    end

    # Unrestriced variables
    obj.Z2[:, (end-punres+1):end] = obj.unrestricted[(obj.lags+1):end, :]

    # Compute concentrated quantities
    if size(obj.Z2, 2) > 0
        obj.R0[:] = mreg(obj.Z0, obj.Z2)[2]
        obj.R1[:] = mreg(obj.Z1, obj.Z2)[2]
    else
        obj.R0[:] = obj.Z0
        obj.R1[:] = obj.Z1
    end

    return obj
end

copy(obj::CivecmI1) = CivecmI1(
    copy(obj.endogenous),
    copy(obj.exogenous),
    copy(obj.unrestricted),
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
    copy(obj.R1),
)

function show(io::IO, ::MIME"text/plain", obj::CivecmI1)
    print(io, summary(obj))
    print(io, "\n\nα:\n")
    show(io, MIME"text/plain"(), obj.α)
    print(io, "\n\nβᵀ:\n")
    show(io, MIME"text/plain"(), copy(obj.β'))
    print(io, "\n\nΠ:\n")
    show(io, MIME"text/plain"(), obj.α * obj.β')
end

function setrank(obj::CivecmI1, rank::Int64)
    α = Matrix{Float64}(undef, size(obj.R0, 2), rank)
    β = Matrix{Float64}(undef, size(obj.R1, 2), rank)
    Hα = Matrix{Float64}(I, size(obj.R0, 2) * rank, size(obj.R0, 2) * rank)
    Hβ = Matrix{Float64}(I, size(obj.R1, 2) * rank, size(obj.R1, 2) * rank)
    hβ = zeros(size(obj.R1, 2) * rank)

    newobj = CivecmI1(
        obj.endogenous,
        obj.exogenous,
        obj.unrestricted,
        obj.lags,
        α,
        β,
        obj.Γ,
        Hα,
        Hβ,
        hβ,
        obj.llConvCrit,
        obj.maxiter,
        obj.verbose,
        obj.Z0,
        obj.Z1,
        obj.Z2,
        obj.R0,
        obj.R1,
    )

    return estimateEigen!(newobj)
end

function restrict(obj::CivecmI1; Hβ = nothing, hβ = nothing, Hα = nothing, verbose = false)

    _Hβ = Hβ === nothing ? obj.Hβ : Hβ
    _hβ = hβ === nothing ? obj.hβ : hβ
    _Hα = Hα === nothing ? obj.Hα : Hα

    newobj = CivecmI1(
        obj.endogenous,
        obj.exogenous,
        obj.unrestricted,
        obj.lags,
        copy(obj.α),
        copy(obj.β),
        copy(obj.Γ),
        _Hα,
        _Hβ,
        _hβ,
        obj.llConvCrit,
        obj.maxiter,
        obj.verbose,
        obj.Z0,
        obj.Z1,
        obj.Z2,
        obj.R0,
        obj.R1,
    )

    estimateSwitch!(newobj, verbose = verbose)

    return newobj
end

function estimate!(obj::CivecmI1; method = :switch)
    if method == :switch || method == :Boswijk
        return estimateSwitch!(obj)
    elseif method == :eigen
        return estimateEigen!(obj)
    else
        throw(ArgumentError("method mist be :switch or :Boswijk but was :$method"))
    end
end

function estimateEigen!(obj::CivecmI1)
    obj.α[:], svdvals, obj.β[:] = rrr(obj.R0, obj.R1, size(obj.α, 2))
    obj.α[:] = obj.α * Diagonal(svdvals)
    obj.Γ[:] = mreg(obj.Z0 - obj.Z1 * obj.β * obj.α', obj.Z2)[1]'
    return obj
end

function estimateSwitch!(obj::CivecmI1; verbose = false)
    iT = size(obj.Z0, 1)
    S11 = rmul!(obj.R1'obj.R1, 1 / iT)
    S10 = rmul!(obj.R1'obj.R0, 1 / iT)
    ll0 = -floatmax()
    ll1 = ll0
    for _ = 1:obj.maxiter
        OmegaInv = inv(cholesky!(residualvariance(obj)))
        aoas11 = kron(obj.α' * OmegaInv * obj.α, S11)
        φ =
            qr!(obj.Hβ' * aoas11 * obj.Hβ, ColumnNorm()) \
            (obj.Hβ' * (vec(S10 * OmegaInv * obj.α) - aoas11 * obj.hβ))
        obj.β .= reshape(obj.Hβ * φ + obj.hβ, size(obj.β)...)
        γ =
            qr!(obj.Hα' * kron(OmegaInv, obj.β' * S11 * obj.β) * obj.Hα, ColumnNorm()) \
            (obj.Hα' * vec(obj.β' * S10 * OmegaInv))
        obj.α .= reshape(obj.Hα * γ, size(obj.α, 2), size(obj.α, 1))'
        ll1 = loglikelihood(obj)
        if verbose
            @printf("log-likelihood: %f\n", ll1)
        end
        if abs(ll1 - ll0) < obj.llConvCrit
            break
        end
        ll0 = ll1
    end
    return obj
end

function ranktest(rng::AbstractRNG, obj::CivecmI1, reps::Int)
    _, svdvals, _ = rrr(obj.R0, obj.R1)
    tmpTrace = -size(obj.Z0, 1) * reverse(cumsum(reverse(log.(1 .- svdvals .^ 2))))
    tmpPVals = zeros(size(tmpTrace))
    rankdist = zeros(reps)
    iT, ip = size(obj.endogenous)
    for i = axes(tmpTrace, 1)
        print("Simulation of model H(", i, ")\r")
        for k = 1:reps
            rankdist[k] =
                I2TraceSimulate(randn(rng, iT, ip - i + 1), ip - i + 1, obj.exogenous)
        end
        tmpPVals[i] = mean(rankdist .> tmpTrace[i])
    end
    print("                                                    \r")
    return TraceTest(tmpTrace, tmpPVals)
end
ranktest(obj::CivecmI1, reps::Int) = ranktest(Random.default_rng(), obj, reps)
ranktest(obj::CivecmI1) = ranktest(obj, 10000)

residuals(obj::CivecmI1) = obj.R0 - obj.R1 * obj.β * obj.α'

## I1 ranktest

struct TraceTest
    values::Vector{Float64}
    pvalues::Vector{Float64}
end

function show(io::IO, ::MIME"text/plain", obj::TraceTest)
    println(io, summary(obj))
    @printf(io, "\n Rank    Value  p-value")
    for i = 1:length(obj.values)
        @printf(io, "\n%5d%9.3f%9.3f", i - 1, obj.values[i], obj.pvalues[i])
    end
end
