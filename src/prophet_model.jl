@doc raw"""
Contructs a matrix with as many columns as changepoints and rows as observations `t`. Checkpoints
are assumed to happen at moment `t_change`. The matrix A results in A[i,j] = 1 if at time i there
are l >= j changepoints activated. 

# Examples
```julia-repl
julia> get_changepoint_matrix(1:10,[2,4,8])
10×3 Matrix{Int8}:
0.0  0.0  0.0
1.0  0.0  0.0
1.0  0.0  0.0
1.0  1.0  0.0
⋮         
1.0  1.0  0.0
1.0  1.0  1.0
1.0  1.0  1.0
1.0  1.0  1.0
```
"""
#FIXME: Change the times to dates or whatever works
function get_changepoint_matrix(
    t::Union{Vector{<:Real},AbstractRange{<:Real}}, 
    t_change::Union{Vector{<:Real},AbstractRange{<:Real}}; 
    T::Integer=size(t,1), 
    S::Integer=size(t_change,1))

    #Create empty arrays
    A      = zeros(Int8, T, S);
    a_row  = zeros(Int8, 1, S);
    cp_idx = 1;

    #Fill in each row of A
    for i in 1:T 
        while (cp_idx <= S) && (t[i] >= t_change[cp_idx])
            a_row[:, cp_idx] .= 1;
            cp_idx = cp_idx + 1;
        end
        A[i,:] = a_row;
    end

    return A;
end
export get_changepoint_matrix

function logistic_gamma(
    k::Real, 
    m::Real, 
    delta::Union{Vector{<:Real},AbstractRange{<:Real}},
    t_change::Union{Vector{<:Real},AbstractRange{<:Real}};
    S::Integer=size(t_change,1)
    )

    gamma = Vector(undef, S);            #Adjusted offsets for piecewise continuity
    k_s   = vcat(k, k .+ cumsum(delta)); #Compute the rate in each segment
    m_pr  = m;                           #Offset in the previous segment
    for i in 1:S
        gamma[i] = (t_change[i] - m_pr)*(1 - k_s[i]/k_s[i + 1]);
        m_pr     = m_pr + gamma[i]; #Update for the next segment
    end
    return gamma;
end
export logistic_gamma

@doc raw"""
julia> logistic_trend(10, 2, 1:4, 1:100, 1:100, get_changepoint_matrix(1:100, [6,12,77,90]),[6,12,77,90])
"""
function logistic_trend(
    k::Real,
    m::Real,
    delta::Union{Vector{<:Real},AbstractRange{<:Real}},
    t::Union{Vector{<:Real},AbstractRange{<:Real}},
    cap::Union{Vector{<:Real},AbstractRange{<:Real}},
    A::Matrix{<:Real},
    t_change::Union{Vector{<:Real},AbstractRange{<:Real}};
    S::Integer=size(t_change,1)
    )

    gamma = logistic_gamma(k, m, delta, t_change; S = S);
    return cap .* logistic.((k .+ A * delta) .* (t - (m .+ A*gamma)));

end
export logistic_trend

function linear_trend(
    k::Real,
    m::Real,
    delta::Union{Vector{<:Real},AbstractRange{<:Real}},
    t::Union{Vector{<:Real},AbstractRange{<:Real}},
    A::Matrix{<:Real},
    t_change::Union{Vector{<:Real},AbstractRange{<:Real}}
    )
    return ((k .+ A*delta) .* t + (m .+ A * (-t_change .* delta)));

end
export linear_trend

function flat_trend(
    m::Real,
    T::Integer
    )
    return repeat([m], T);
end
export flat_trend

function get_trend(
    trend::String, 
    k::Real, 
    m::Real,
    delta::Union{Vector{<:Real},AbstractRange{<:Real}},
    t::Union{Vector{<:Real},AbstractRange{<:Real}},
    A::Matrix{<:Real},
    t_change::Union{Vector{<:Real},AbstractRange{<:Real}};
    cap::Union{Vector{<:Real},AbstractRange{<:Real}, Nothing} = nothing,
    T::Integer=size(t, 1),
    S::Integer=size(t_change, 1))

    if trend == "linear"
        tf = linear_trend(k, m, delta, t, A, t_change);
    elseif trend == "logistic"
        tf = logistic_trend(k, m, delta, t, cap, A, t_change; S = S)
    elseif trend == "flat"
        tf = flat_trend(m, T);
    end

    return tf
end
export get_trend

function validate_prophet_model_data(
    t::Union{Vector{<:Real},AbstractRange{<:Real}},         #Time
    y::Union{Vector{<:Real},AbstractRange{<:Real}},         #Time series indexed by time
    t_change::Union{Vector{<:Real},AbstractRange{<:Real}},  #Times when changepoints happen
    X::Union{VecOrMat{<:Real},AbstractRange{<:Real}},       #Regressors
    sigmas::Vector{<:Real},                                 #Scale on seasonality prior
    tau::Real,                                              #Scale on changepoints prior
    trend::String,                                          #'linear', 'logistic' or 'flat' 
    s_a::Vector{Int8},                                      #Indicator of additive features
    s_m::Vector{Int8};                                      #Indicator of multiplivative features
    cap::Union{Vector{<:Real},AbstractRange{<:Real},Nothing} = nothing, #Capacities for logistic trend
    T::Integer = size(t,1),
    S::Integer = size(t_change, 1)
    )

    #Validate dimensions for time
    if !isnothing(cap) && size(cap,1) != T
        error("`cap` and `t` must me of same size ($T != $(size(cap,1)))")
    end

    #Check that changepoints occurr in the time
    if min(t_change) <= min(t) || max(t_change) > max(t)
        error("Some changepoints in `t_change` are outside the range of `t`: [$(min(t_change)),$(max(t_change))]")
    end

    #Check T
    if T != size(t,1)
        error("T != size(t,1)")
    end

    #Check S
    if S != size(t_change,1)
        error("S != size(t_change,1)")
    end

    #Validate cap when logistic
    if isnothing(cap) && trend == "logistic"
        error("`cap` must be specified if using `trend == $trend`")
    end

    #Validate dimensions for time series
    if size(y,1) != T
        error("Time series `y` and time `t` must me of same size ($T != $(size(y,1)))")
    end

    #Validate dimensions for regressors
    if size(X,1) != T
        error("Regressors `X` and time `t` must me of same size ($T != $(size(X,1)))")
    end

    #Validate regressor sigmas and number of regressors
    if size(X,2) != size(sigmas,1)
        error("`sigmas` must have the same length as number of regressors `X` ($(size(X,2)) != $(size(sigmas,1)))")
    end

    #Validate regressor sigmas and number of regressors
    if tau < 0
        error("Scale on changepoints `tau` cannot be negative (tau = $tau)")
    end

    #Validate trend name
    if !(trend in ["linear", "logistic", "flat"])
        error("Invalid trend = $trend. Select 'linear', 'logistic' or 'flat'")
    end

    #Validate size of s_a
    if size(s_a, 1) != size(X,2)
        error("`s_a` indicators must have the same length as number of regressors `X` ($(size(X,2)) != $(size(s_a,1)))")
    end

    #Validate size of s_m
    if size(s_m, 1) != size(X,2)
        error("`s_m` indicators must have the same length as number of regressors `X` ($(size(X,2)) != $(size(s_m,1)))")
    end

end
export validate_prophet_model_data

#ProphetModel turing model
#https://github.com/facebook/prophet/blob/main/R/inst/stan/prophet.stan
@model function turing_prophet_sampling(
    t::Union{Vector{<:Real},AbstractRange{<:Real}},         #Time
    y::Union{Vector{<:Real},AbstractRange{<:Real}},         #Time series indexed by time
    t_change::Union{Vector{<:Real},AbstractRange{<:Real}},  #Times when changepoints happen
    sigmas::Union{Vector{<:Real},AbstractRange{<:Real}},
    trend::String, #'linear', 'logistic' or 'flat' 
    tau::Real,
    K::Integer,
    A::Matrix{Int8},
    X_sa::Matrix{Int8},
    X_sm::Matrix{Int8};
    T::Integer=size(t,1), 
    S::Integer=size(t_change,1),
    cap::Union{Vector{<:Real},AbstractRange{<:Real},Nothing} = nothing,
    )

    k          ~ Normal(0.0, 5.0);
    m          ~ Normal(0.0, 5.0);
    delta     .~ Laplace(0.0, tau);
    sigma_obs  ~ Normal(0.0, 0.5);
    beta       ~ MvNormal(zeros(sigmas, K), Diagonal(sigmas));

    alpha = get_trend(trend, k, m, delta, t, A, t_change; cap = cap, S = S, T = T);

    y ~ MvNormal(alpha.*(1 .+ X_sm*beta) + X_sa*beta, Diagonal(sigma_obs));

end
export turing_prophet_sampling

function prophet_model(
    t::Union{Vector{<:Real},AbstractRange{<:Real}},         #Time
    y::Union{Vector{<:Real},AbstractRange{<:Real}},         #Time series indexed by time
    t_change::Union{Vector{<:Real},AbstractRange{<:Real}},  #Times when changepoints happen
    X::Union{VecOrMat{<:Real},AbstractRange{<:Real}},       #Regressors
    sigmas::Vector{<:Real},                                 #Scale on seasonality prior
    tau::Real,                                              #Scale on changepoints prior
    trend::String,                                          #'linear', 'logistic' or 'flat' 
    s_a::Vector{Int8},                                      #Indicator of additive features
    s_m::Vector{Int8};                                      #Indicator of multiplivative features
    cap::Union{Vector{<:Real},AbstractRange{<:Real},Nothing} = nothing, #Capacities for logistic trend
    sampler::AbstractMCMC.AbstractSampler = NUTS(),
    discard_initial::Integer=0,
    thinning::Float64=1.0,
    ensemble::Union{<:AbstractMCMC.AbstractMCMCEnsemble,Nothing}=nothing,
    n_samples::Integer = 1_000, 
    nchains::Integer = 1,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    progress::Bool=Turing.PROGRESS[],
    kwargs...
)

    #Get dimension parameters
    T = size(t,1);
    S = size(t_change,1);
    K = size(X, 2);

    #Validate the inputs
    validate_prophet_model_data(t, y, t_change, X, sigmas, tau, trend, s_a, s_m; cap = cap, T = T, S = S);

    #Create the changepoint matrix
    A = get_changepoint_matrix(t, t_change; T = T, S = S);

    #Matrices to get additive terms
    X_sa = X .* reshape(repeat(s_a, T), T, K);
    X_sm = X .* reshape(repeat(s_m, T), T, K);

    model  = turing_prophet_sampling(t, y, t_change, sigmas, trend, tau, K, A, X_sa, X_sm; S = S, T = T, cap = cap);

    if isnothing(ensemble) || nchains == 1
        chains = sample(rng, model, sampler, n_samples; progress = progress, thinning = thinning, discard_initial = discard_initial, kwargs...);
    else 
        chains = sample(rng, model, ensemble, n_samples, nchains; progress = progress, thinning = thinning, discard_initial = discard_initial, kwargs...);
    end

    return chains;
end
export prophet_model