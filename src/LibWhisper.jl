module LibWhisper

using whisper_cpp_jll
export whisper_cpp_jll

using CEnum

mutable struct whisper_context end

const whisper_token = Cint

struct whisper_token_data
    id::whisper_token
    tid::whisper_token
    p::Cfloat
    pt::Cfloat
    ptsum::Cfloat
    t0::Int64
    t1::Int64
    vlen::Cfloat
end

function whisper_init(path_model)
    ccall((:whisper_init, libwhisper), Ptr{whisper_context}, (Ptr{Cchar},), path_model)
end

function whisper_free(ctx)
    ccall((:whisper_free, libwhisper), Cvoid, (Ptr{whisper_context},), ctx)
end

function whisper_pcm_to_mel(ctx, samples, n_samples, n_threads)
    ccall((:whisper_pcm_to_mel, libwhisper), Cint, (Ptr{whisper_context}, Ptr{Cfloat}, Cint, Cint), ctx, samples, n_samples, n_threads)
end

function whisper_set_mel(ctx, data, n_len, n_mel)
    ccall((:whisper_set_mel, libwhisper), Cint, (Ptr{whisper_context}, Ptr{Cfloat}, Cint, Cint), ctx, data, n_len, n_mel)
end

function whisper_encode(ctx, offset, n_threads)
    ccall((:whisper_encode, libwhisper), Cint, (Ptr{whisper_context}, Cint, Cint), ctx, offset, n_threads)
end

function whisper_decode(ctx, tokens, n_tokens, n_past, n_threads)
    ccall((:whisper_decode, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_token}, Cint, Cint, Cint), ctx, tokens, n_tokens, n_past, n_threads)
end

function whisper_sample_best(ctx)
    ccall((:whisper_sample_best, libwhisper), whisper_token_data, (Ptr{whisper_context},), ctx)
end

function whisper_sample_timestamp(ctx, is_initial)
    ccall((:whisper_sample_timestamp, libwhisper), whisper_token_data, (Ptr{whisper_context}, Bool), ctx, is_initial)
end

function whisper_tokenize(ctx, text, tokens, n_max_tokens)
    ccall((:whisper_tokenize, libwhisper), Cint, (Ptr{whisper_context}, Ptr{Cchar}, Ptr{whisper_token}, Cint), ctx, text, tokens, n_max_tokens)
end

# no prototype is found for this function at whisper.h:154:21, please use with caution
function whisper_lang_max_id()
    ccall((:whisper_lang_max_id, libwhisper), Cint, ())
end

function whisper_lang_id(lang)
    ccall((:whisper_lang_id, libwhisper), Cint, (Ptr{Cchar},), lang)
end

function whisper_lang_str(id)
    ccall((:whisper_lang_str, libwhisper), Ptr{Cchar}, (Cint,), id)
end

function whisper_lang_auto_detect(ctx, offset_ms, n_threads, lang_probs)
    ccall((:whisper_lang_auto_detect, libwhisper), Cint, (Ptr{whisper_context}, Cint, Cint, Ptr{Cfloat}), ctx, offset_ms, n_threads, lang_probs)
end

function whisper_n_len(ctx)
    ccall((:whisper_n_len, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_n_vocab(ctx)
    ccall((:whisper_n_vocab, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_n_text_ctx(ctx)
    ccall((:whisper_n_text_ctx, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_n_audio_ctx(ctx)
    ccall((:whisper_n_audio_ctx, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_is_multilingual(ctx)
    ccall((:whisper_is_multilingual, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_get_probs(ctx)
    ccall((:whisper_get_probs, libwhisper), Ptr{Cfloat}, (Ptr{whisper_context},), ctx)
end

function whisper_token_to_str(ctx, token)
    ccall((:whisper_token_to_str, libwhisper), Ptr{Cchar}, (Ptr{whisper_context}, whisper_token), ctx, token)
end

function whisper_token_eot(ctx)
    ccall((:whisper_token_eot, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_sot(ctx)
    ccall((:whisper_token_sot, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_prev(ctx)
    ccall((:whisper_token_prev, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_solm(ctx)
    ccall((:whisper_token_solm, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_not(ctx)
    ccall((:whisper_token_not, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_beg(ctx)
    ccall((:whisper_token_beg, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_lang(ctx, lang_id)
    ccall((:whisper_token_lang, libwhisper), whisper_token, (Ptr{whisper_context}, Cint), ctx, lang_id)
end

function whisper_token_translate()
    ccall((:whisper_token_translate, libwhisper), whisper_token, ())
end

function whisper_token_transcribe()
    ccall((:whisper_token_transcribe, libwhisper), whisper_token, ())
end

function whisper_print_timings(ctx)
    ccall((:whisper_print_timings, libwhisper), Cvoid, (Ptr{whisper_context},), ctx)
end

function whisper_reset_timings(ctx)
    ccall((:whisper_reset_timings, libwhisper), Cvoid, (Ptr{whisper_context},), ctx)
end

function whisper_print_system_info()
    ccall((:whisper_print_system_info, libwhisper), Ptr{Cchar}, ())
end

@cenum whisper_sampling_strategy::UInt32 begin
    WHISPER_SAMPLING_GREEDY = 0
    WHISPER_SAMPLING_BEAM_SEARCH = 1
end

# typedef void ( * whisper_new_segment_callback ) ( struct whisper_context * ctx , int n_new , void * user_data )
const whisper_new_segment_callback = Ptr{Cvoid}

# typedef bool ( * whisper_encoder_begin_callback ) ( struct whisper_context * ctx , void * user_data )
const whisper_encoder_begin_callback = Ptr{Cvoid}

struct __JL_Ctag_2
    n_past::Cint
end
function Base.getproperty(x::Ptr{__JL_Ctag_2}, f::Symbol)
    f === :n_past && return Ptr{Cint}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_2, f::Symbol)
    r = Ref{__JL_Ctag_2}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_2}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_2}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct __JL_Ctag_3
    n_past::Cint
    beam_width::Cint
    n_best::Cint
end
function Base.getproperty(x::Ptr{__JL_Ctag_3}, f::Symbol)
    f === :n_past && return Ptr{Cint}(x + 0)
    f === :beam_width && return Ptr{Cint}(x + 4)
    f === :n_best && return Ptr{Cint}(x + 8)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_3, f::Symbol)
    r = Ref{__JL_Ctag_3}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_3}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_3}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct whisper_full_params
    data::NTuple{128, UInt8}
end

function Base.getproperty(x::Ptr{whisper_full_params}, f::Symbol)
    f === :strategy && return Ptr{whisper_sampling_strategy}(x + 0)
    f === :n_threads && return Ptr{Cint}(x + 4)
    f === :n_max_text_ctx && return Ptr{Cint}(x + 8)
    f === :offset_ms && return Ptr{Cint}(x + 12)
    f === :duration_ms && return Ptr{Cint}(x + 16)
    f === :translate && return Ptr{Bool}(x + 20)
    f === :no_context && return Ptr{Bool}(x + 21)
    f === :single_segment && return Ptr{Bool}(x + 22)
    f === :print_special && return Ptr{Bool}(x + 23)
    f === :print_progress && return Ptr{Bool}(x + 24)
    f === :print_realtime && return Ptr{Bool}(x + 25)
    f === :print_timestamps && return Ptr{Bool}(x + 26)
    f === :token_timestamps && return Ptr{Bool}(x + 27)
    f === :thold_pt && return Ptr{Cfloat}(x + 28)
    f === :thold_ptsum && return Ptr{Cfloat}(x + 32)
    f === :max_len && return Ptr{Cint}(x + 36)
    f === :max_tokens && return Ptr{Cint}(x + 40)
    f === :speed_up && return Ptr{Bool}(x + 44)
    f === :audio_ctx && return Ptr{Cint}(x + 48)
    f === :prompt_tokens && return Ptr{Ptr{whisper_token}}(x + 56)
    f === :prompt_n_tokens && return Ptr{Cint}(x + 64)
    f === :language && return Ptr{Ptr{Cchar}}(x + 72)
    f === :greedy && return Ptr{__JL_Ctag_2}(x + 80)
    f === :beam_search && return Ptr{__JL_Ctag_3}(x + 84)
    f === :new_segment_callback && return Ptr{whisper_new_segment_callback}(x + 96)
    f === :new_segment_callback_user_data && return Ptr{Ptr{Cvoid}}(x + 104)
    f === :encoder_begin_callback && return Ptr{whisper_encoder_begin_callback}(x + 112)
    f === :encoder_begin_callback_user_data && return Ptr{Ptr{Cvoid}}(x + 120)
    return getfield(x, f)
end

function Base.getproperty(x::whisper_full_params, f::Symbol)
    r = Ref{whisper_full_params}(x)
    ptr = Base.unsafe_convert(Ptr{whisper_full_params}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{whisper_full_params}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function whisper_full_default_params(strategy)
    ccall((:whisper_full_default_params, libwhisper), whisper_full_params, (whisper_sampling_strategy,), strategy)
end

function whisper_full(ctx, params, samples, n_samples)
    ccall((:whisper_full, libwhisper), Cint, (Ptr{whisper_context}, whisper_full_params, Ptr{Cfloat}, Cint), ctx, params, samples, n_samples)
end

function whisper_full_parallel(ctx, params, samples, n_samples, n_processors)
    ccall((:whisper_full_parallel, libwhisper), Cint, (Ptr{whisper_context}, whisper_full_params, Ptr{Cfloat}, Cint, Cint), ctx, params, samples, n_samples, n_processors)
end

function whisper_full_n_segments(ctx)
    ccall((:whisper_full_n_segments, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_full_get_segment_t0(ctx, i_segment)
    ccall((:whisper_full_get_segment_t0, libwhisper), Int64, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_get_segment_t1(ctx, i_segment)
    ccall((:whisper_full_get_segment_t1, libwhisper), Int64, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_get_segment_text(ctx, i_segment)
    ccall((:whisper_full_get_segment_text, libwhisper), Ptr{Cchar}, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_n_tokens(ctx, i_segment)
    ccall((:whisper_full_n_tokens, libwhisper), Cint, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_get_token_text(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_text, libwhisper), Ptr{Cchar}, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_id(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_id, libwhisper), whisper_token, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_data(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_data, libwhisper), whisper_token_data, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_p(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_p, libwhisper), Cfloat, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

const WHISPER_SAMPLE_RATE = 16000

const WHISPER_N_FFT = 400

const WHISPER_N_MEL = 80

const WHISPER_HOP_LENGTH = 160

const WHISPER_CHUNK_SIZE = 30

# exports
const PREFIXES = ["whisper_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
