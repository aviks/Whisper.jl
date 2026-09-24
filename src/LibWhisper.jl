module LibWhisper

using whisper_cpp_jll
export whisper_cpp_jll

using CEnum: CEnum, @cenum

mutable struct whisper_context end

mutable struct whisper_state end

const whisper_pos = Int32

const whisper_token = Int32

const whisper_seq_id = Int32

@cenum whisper_alignment_heads_preset::UInt32 begin
    WHISPER_AHEADS_NONE = 0
    WHISPER_AHEADS_N_TOP_MOST = 1
    WHISPER_AHEADS_CUSTOM = 2
    WHISPER_AHEADS_TINY_EN = 3
    WHISPER_AHEADS_TINY = 4
    WHISPER_AHEADS_BASE_EN = 5
    WHISPER_AHEADS_BASE = 6
    WHISPER_AHEADS_SMALL_EN = 7
    WHISPER_AHEADS_SMALL = 8
    WHISPER_AHEADS_MEDIUM_EN = 9
    WHISPER_AHEADS_MEDIUM = 10
    WHISPER_AHEADS_LARGE_V1 = 11
    WHISPER_AHEADS_LARGE_V2 = 12
    WHISPER_AHEADS_LARGE_V3 = 13
    WHISPER_AHEADS_LARGE_V3_TURBO = 14
end

struct whisper_ahead
    n_text_layer::Cint
    n_head::Cint
end

struct whisper_aheads
    n_heads::Csize_t
    heads::Ptr{whisper_ahead}
end

struct whisper_context_params
    use_gpu::Bool
    flash_attn::Bool
    gpu_device::Cint
    dtw_token_timestamps::Bool
    dtw_aheads_preset::whisper_alignment_heads_preset
    dtw_n_top::Cint
    dtw_aheads::whisper_aheads
    dtw_mem_size::Csize_t
end

struct whisper_token_data
    id::whisper_token
    tid::whisper_token
    p::Cfloat
    plog::Cfloat
    pt::Cfloat
    ptsum::Cfloat
    t0::Int64
    t1::Int64
    t_dtw::Int64
    vlen::Cfloat
end

struct whisper_model_loader
    context::Ptr{Cvoid}
    read::Ptr{Cvoid}
    eof::Ptr{Cvoid}
    close::Ptr{Cvoid}
end

@cenum whisper_gretype::UInt32 begin
    WHISPER_GRETYPE_END = 0
    WHISPER_GRETYPE_ALT = 1
    WHISPER_GRETYPE_RULE_REF = 2
    WHISPER_GRETYPE_CHAR = 3
    WHISPER_GRETYPE_CHAR_NOT = 4
    WHISPER_GRETYPE_CHAR_RNG_UPPER = 5
    WHISPER_GRETYPE_CHAR_ALT = 6
end

struct whisper_grammar_element
    type::whisper_gretype
    value::UInt32
end

struct whisper_vad_params
    threshold::Cfloat
    min_speech_duration_ms::Cint
    min_silence_duration_ms::Cint
    max_speech_duration_s::Cfloat
    speech_pad_ms::Cint
    samples_overlap::Cfloat
end

function whisper_version()
    ccall((:whisper_version, libwhisper), Ptr{Cchar}, ())
end

function whisper_init_from_file_with_params(path_model, params)
    ccall((:whisper_init_from_file_with_params, libwhisper), Ptr{whisper_context}, (Ptr{Cchar}, whisper_context_params), path_model, params)
end

function whisper_init_from_buffer_with_params(buffer, buffer_size, params)
    ccall((:whisper_init_from_buffer_with_params, libwhisper), Ptr{whisper_context}, (Ptr{Cvoid}, Csize_t, whisper_context_params), buffer, buffer_size, params)
end

function whisper_init_with_params(loader, params)
    ccall((:whisper_init_with_params, libwhisper), Ptr{whisper_context}, (Ptr{whisper_model_loader}, whisper_context_params), loader, params)
end

function whisper_init_from_file_with_params_no_state(path_model, params)
    ccall((:whisper_init_from_file_with_params_no_state, libwhisper), Ptr{whisper_context}, (Ptr{Cchar}, whisper_context_params), path_model, params)
end

function whisper_init_from_buffer_with_params_no_state(buffer, buffer_size, params)
    ccall((:whisper_init_from_buffer_with_params_no_state, libwhisper), Ptr{whisper_context}, (Ptr{Cvoid}, Csize_t, whisper_context_params), buffer, buffer_size, params)
end

function whisper_init_with_params_no_state(loader, params)
    ccall((:whisper_init_with_params_no_state, libwhisper), Ptr{whisper_context}, (Ptr{whisper_model_loader}, whisper_context_params), loader, params)
end

function whisper_init_from_file(path_model)
    ccall((:whisper_init_from_file, libwhisper), Ptr{whisper_context}, (Ptr{Cchar},), path_model)
end

function whisper_init_from_buffer(buffer, buffer_size)
    ccall((:whisper_init_from_buffer, libwhisper), Ptr{whisper_context}, (Ptr{Cvoid}, Csize_t), buffer, buffer_size)
end

function whisper_init(loader)
    ccall((:whisper_init, libwhisper), Ptr{whisper_context}, (Ptr{whisper_model_loader},), loader)
end

function whisper_init_from_file_no_state(path_model)
    ccall((:whisper_init_from_file_no_state, libwhisper), Ptr{whisper_context}, (Ptr{Cchar},), path_model)
end

function whisper_init_from_buffer_no_state(buffer, buffer_size)
    ccall((:whisper_init_from_buffer_no_state, libwhisper), Ptr{whisper_context}, (Ptr{Cvoid}, Csize_t), buffer, buffer_size)
end

function whisper_init_no_state(loader)
    ccall((:whisper_init_no_state, libwhisper), Ptr{whisper_context}, (Ptr{whisper_model_loader},), loader)
end

function whisper_init_state(ctx)
    ccall((:whisper_init_state, libwhisper), Ptr{whisper_state}, (Ptr{whisper_context},), ctx)
end

function whisper_ctx_init_openvino_encoder_with_state(ctx, state, model_path, device, cache_dir)
    ccall((:whisper_ctx_init_openvino_encoder_with_state, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_state}, Ptr{Cchar}, Ptr{Cchar}, Ptr{Cchar}), ctx, state, model_path, device, cache_dir)
end

function whisper_ctx_init_openvino_encoder(ctx, model_path, device, cache_dir)
    ccall((:whisper_ctx_init_openvino_encoder, libwhisper), Cint, (Ptr{whisper_context}, Ptr{Cchar}, Ptr{Cchar}, Ptr{Cchar}), ctx, model_path, device, cache_dir)
end

function whisper_free(ctx)
    ccall((:whisper_free, libwhisper), Cvoid, (Ptr{whisper_context},), ctx)
end

function whisper_free_state(state)
    ccall((:whisper_free_state, libwhisper), Cvoid, (Ptr{whisper_state},), state)
end

@cenum whisper_sampling_strategy::UInt32 begin
    WHISPER_SAMPLING_GREEDY = 0
    WHISPER_SAMPLING_BEAM_SEARCH = 1
end

struct __JL_Ctag_1
    best_of::Cint
end
function Base.getproperty(x::Ptr{__JL_Ctag_1}, f::Symbol)
    f === :best_of && return Ptr{Cint}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_1, f::Symbol)
    r = Ref{__JL_Ctag_1}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_1}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_1}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct __JL_Ctag_2
    beam_size::Cint
    patience::Cfloat
end
function Base.getproperty(x::Ptr{__JL_Ctag_2}, f::Symbol)
    f === :beam_size && return Ptr{Cint}(x + 0)
    f === :patience && return Ptr{Cfloat}(x + 4)
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


# typedef void ( * whisper_new_segment_callback ) ( struct whisper_context * ctx , struct whisper_state * state , int n_new , void * user_data )
const whisper_new_segment_callback = Ptr{Cvoid}

# typedef void ( * whisper_progress_callback ) ( struct whisper_context * ctx , struct whisper_state * state , int progress , void * user_data )
const whisper_progress_callback = Ptr{Cvoid}

# typedef bool ( * whisper_encoder_begin_callback ) ( struct whisper_context * ctx , struct whisper_state * state , void * user_data )
const whisper_encoder_begin_callback = Ptr{Cvoid}

# typedef bool ( * ggml_abort_callback ) ( void * data )
const ggml_abort_callback = Ptr{Cvoid}

# typedef void ( * whisper_logits_filter_callback ) ( struct whisper_context * ctx , struct whisper_state * state , const whisper_token_data * tokens , int n_tokens , float * logits , void * user_data )
const whisper_logits_filter_callback = Ptr{Cvoid}

struct whisper_full_params
    data::NTuple{304, UInt8}
end

function Base.getproperty(x::Ptr{whisper_full_params}, f::Symbol)
    f === :strategy && return Ptr{whisper_sampling_strategy}(x + 0)
    f === :n_threads && return Ptr{Cint}(x + 4)
    f === :n_max_text_ctx && return Ptr{Cint}(x + 8)
    f === :offset_ms && return Ptr{Cint}(x + 12)
    f === :duration_ms && return Ptr{Cint}(x + 16)
    f === :translate && return Ptr{Bool}(x + 20)
    f === :no_context && return Ptr{Bool}(x + 21)
    f === :no_timestamps && return Ptr{Bool}(x + 22)
    f === :single_segment && return Ptr{Bool}(x + 23)
    f === :print_special && return Ptr{Bool}(x + 24)
    f === :print_progress && return Ptr{Bool}(x + 25)
    f === :print_realtime && return Ptr{Bool}(x + 26)
    f === :print_timestamps && return Ptr{Bool}(x + 27)
    f === :token_timestamps && return Ptr{Bool}(x + 28)
    f === :thold_pt && return Ptr{Cfloat}(x + 32)
    f === :thold_ptsum && return Ptr{Cfloat}(x + 36)
    f === :max_len && return Ptr{Cint}(x + 40)
    f === :split_on_word && return Ptr{Bool}(x + 44)
    f === :max_tokens && return Ptr{Cint}(x + 48)
    f === :debug_mode && return Ptr{Bool}(x + 52)
    f === :audio_ctx && return Ptr{Cint}(x + 56)
    f === :tdrz_enable && return Ptr{Bool}(x + 60)
    f === :suppress_regex && return Ptr{Ptr{Cchar}}(x + 64)
    f === :initial_prompt && return Ptr{Ptr{Cchar}}(x + 72)
    f === :carry_initial_prompt && return Ptr{Bool}(x + 80)
    f === :prompt_tokens && return Ptr{Ptr{whisper_token}}(x + 88)
    f === :prompt_n_tokens && return Ptr{Cint}(x + 96)
    f === :language && return Ptr{Ptr{Cchar}}(x + 104)
    f === :detect_language && return Ptr{Bool}(x + 112)
    f === :suppress_blank && return Ptr{Bool}(x + 113)
    f === :suppress_nst && return Ptr{Bool}(x + 114)
    f === :temperature && return Ptr{Cfloat}(x + 116)
    f === :max_initial_ts && return Ptr{Cfloat}(x + 120)
    f === :length_penalty && return Ptr{Cfloat}(x + 124)
    f === :temperature_inc && return Ptr{Cfloat}(x + 128)
    f === :entropy_thold && return Ptr{Cfloat}(x + 132)
    f === :logprob_thold && return Ptr{Cfloat}(x + 136)
    f === :no_speech_thold && return Ptr{Cfloat}(x + 140)
    f === :greedy && return Ptr{__JL_Ctag_1}(x + 144)
    f === :beam_search && return Ptr{__JL_Ctag_2}(x + 148)
    f === :new_segment_callback && return Ptr{whisper_new_segment_callback}(x + 160)
    f === :new_segment_callback_user_data && return Ptr{Ptr{Cvoid}}(x + 168)
    f === :progress_callback && return Ptr{whisper_progress_callback}(x + 176)
    f === :progress_callback_user_data && return Ptr{Ptr{Cvoid}}(x + 184)
    f === :encoder_begin_callback && return Ptr{whisper_encoder_begin_callback}(x + 192)
    f === :encoder_begin_callback_user_data && return Ptr{Ptr{Cvoid}}(x + 200)
    f === :abort_callback && return Ptr{ggml_abort_callback}(x + 208)
    f === :abort_callback_user_data && return Ptr{Ptr{Cvoid}}(x + 216)
    f === :logits_filter_callback && return Ptr{whisper_logits_filter_callback}(x + 224)
    f === :logits_filter_callback_user_data && return Ptr{Ptr{Cvoid}}(x + 232)
    f === :grammar_rules && return Ptr{Ptr{Ptr{whisper_grammar_element}}}(x + 240)
    f === :n_grammar_rules && return Ptr{Csize_t}(x + 248)
    f === :i_start_rule && return Ptr{Csize_t}(x + 256)
    f === :grammar_penalty && return Ptr{Cfloat}(x + 264)
    f === :vad && return Ptr{Bool}(x + 268)
    f === :vad_model_path && return Ptr{Ptr{Cchar}}(x + 272)
    f === :vad_params && return Ptr{whisper_vad_params}(x + 280)
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

function Base.propertynames(x::whisper_full_params, private::Bool = false)
    (:strategy, :n_threads, :n_max_text_ctx, :offset_ms, :duration_ms, :translate, :no_context, :no_timestamps, :single_segment, :print_special, :print_progress, :print_realtime, :print_timestamps, :token_timestamps, :thold_pt, :thold_ptsum, :max_len, :split_on_word, :max_tokens, :debug_mode, :audio_ctx, :tdrz_enable, :suppress_regex, :initial_prompt, :carry_initial_prompt, :prompt_tokens, :prompt_n_tokens, :language, :detect_language, :suppress_blank, :suppress_nst, :temperature, :max_initial_ts, :length_penalty, :temperature_inc, :entropy_thold, :logprob_thold, :no_speech_thold, :greedy, :beam_search, :new_segment_callback, :new_segment_callback_user_data, :progress_callback, :progress_callback_user_data, :encoder_begin_callback, :encoder_begin_callback_user_data, :abort_callback, :abort_callback_user_data, :logits_filter_callback, :logits_filter_callback_user_data, :grammar_rules, :n_grammar_rules, :i_start_rule, :grammar_penalty, :vad, :vad_model_path, :vad_params, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

function whisper_free_params(params)
    ccall((:whisper_free_params, libwhisper), Cvoid, (Ptr{whisper_full_params},), params)
end

function whisper_free_context_params(params)
    ccall((:whisper_free_context_params, libwhisper), Cvoid, (Ptr{whisper_context_params},), params)
end

function whisper_pcm_to_mel(ctx, samples, n_samples, n_threads)
    ccall((:whisper_pcm_to_mel, libwhisper), Cint, (Ptr{whisper_context}, Ptr{Cfloat}, Cint, Cint), ctx, samples, n_samples, n_threads)
end

function whisper_pcm_to_mel_with_state(ctx, state, samples, n_samples, n_threads)
    ccall((:whisper_pcm_to_mel_with_state, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_state}, Ptr{Cfloat}, Cint, Cint), ctx, state, samples, n_samples, n_threads)
end

function whisper_set_mel(ctx, data, n_len, n_mel)
    ccall((:whisper_set_mel, libwhisper), Cint, (Ptr{whisper_context}, Ptr{Cfloat}, Cint, Cint), ctx, data, n_len, n_mel)
end

function whisper_set_mel_with_state(ctx, state, data, n_len, n_mel)
    ccall((:whisper_set_mel_with_state, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_state}, Ptr{Cfloat}, Cint, Cint), ctx, state, data, n_len, n_mel)
end

function whisper_encode(ctx, offset, n_threads)
    ccall((:whisper_encode, libwhisper), Cint, (Ptr{whisper_context}, Cint, Cint), ctx, offset, n_threads)
end

function whisper_encode_with_state(ctx, state, offset, n_threads)
    ccall((:whisper_encode_with_state, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_state}, Cint, Cint), ctx, state, offset, n_threads)
end

function whisper_decode(ctx, tokens, n_tokens, n_past, n_threads)
    ccall((:whisper_decode, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_token}, Cint, Cint, Cint), ctx, tokens, n_tokens, n_past, n_threads)
end

function whisper_decode_with_state(ctx, state, tokens, n_tokens, n_past, n_threads)
    ccall((:whisper_decode_with_state, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_state}, Ptr{whisper_token}, Cint, Cint, Cint), ctx, state, tokens, n_tokens, n_past, n_threads)
end

function whisper_tokenize(ctx, text, tokens, n_max_tokens)
    ccall((:whisper_tokenize, libwhisper), Cint, (Ptr{whisper_context}, Ptr{Cchar}, Ptr{whisper_token}, Cint), ctx, text, tokens, n_max_tokens)
end

function whisper_token_count(ctx, text)
    ccall((:whisper_token_count, libwhisper), Cint, (Ptr{whisper_context}, Ptr{Cchar}), ctx, text)
end

function whisper_lang_max_id()
    ccall((:whisper_lang_max_id, libwhisper), Cint, ())
end

function whisper_lang_id(lang)
    ccall((:whisper_lang_id, libwhisper), Cint, (Ptr{Cchar},), lang)
end

function whisper_lang_str(id)
    ccall((:whisper_lang_str, libwhisper), Ptr{Cchar}, (Cint,), id)
end

function whisper_lang_str_full(id)
    ccall((:whisper_lang_str_full, libwhisper), Ptr{Cchar}, (Cint,), id)
end

function whisper_lang_auto_detect(ctx, offset_ms, n_threads, lang_probs)
    ccall((:whisper_lang_auto_detect, libwhisper), Cint, (Ptr{whisper_context}, Cint, Cint, Ptr{Cfloat}), ctx, offset_ms, n_threads, lang_probs)
end

function whisper_lang_auto_detect_with_state(ctx, state, offset_ms, n_threads, lang_probs)
    ccall((:whisper_lang_auto_detect_with_state, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_state}, Cint, Cint, Ptr{Cfloat}), ctx, state, offset_ms, n_threads, lang_probs)
end

function whisper_n_len(ctx)
    ccall((:whisper_n_len, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_n_len_from_state(state)
    ccall((:whisper_n_len_from_state, libwhisper), Cint, (Ptr{whisper_state},), state)
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

function whisper_model_n_vocab(ctx)
    ccall((:whisper_model_n_vocab, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_audio_ctx(ctx)
    ccall((:whisper_model_n_audio_ctx, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_audio_state(ctx)
    ccall((:whisper_model_n_audio_state, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_audio_head(ctx)
    ccall((:whisper_model_n_audio_head, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_audio_layer(ctx)
    ccall((:whisper_model_n_audio_layer, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_text_ctx(ctx)
    ccall((:whisper_model_n_text_ctx, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_text_state(ctx)
    ccall((:whisper_model_n_text_state, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_text_head(ctx)
    ccall((:whisper_model_n_text_head, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_text_layer(ctx)
    ccall((:whisper_model_n_text_layer, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_n_mels(ctx)
    ccall((:whisper_model_n_mels, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_ftype(ctx)
    ccall((:whisper_model_ftype, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_model_type(ctx)
    ccall((:whisper_model_type, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_get_logits(ctx)
    ccall((:whisper_get_logits, libwhisper), Ptr{Cfloat}, (Ptr{whisper_context},), ctx)
end

function whisper_get_logits_from_state(state)
    ccall((:whisper_get_logits_from_state, libwhisper), Ptr{Cfloat}, (Ptr{whisper_state},), state)
end

function whisper_token_to_str(ctx, token)
    ccall((:whisper_token_to_str, libwhisper), Ptr{Cchar}, (Ptr{whisper_context}, whisper_token), ctx, token)
end

function whisper_model_type_readable(ctx)
    ccall((:whisper_model_type_readable, libwhisper), Ptr{Cchar}, (Ptr{whisper_context},), ctx)
end

function whisper_token_eot(ctx)
    ccall((:whisper_token_eot, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_sot(ctx)
    ccall((:whisper_token_sot, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_solm(ctx)
    ccall((:whisper_token_solm, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_prev(ctx)
    ccall((:whisper_token_prev, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_nosp(ctx)
    ccall((:whisper_token_nosp, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
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

function whisper_token_translate(ctx)
    ccall((:whisper_token_translate, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

function whisper_token_transcribe(ctx)
    ccall((:whisper_token_transcribe, libwhisper), whisper_token, (Ptr{whisper_context},), ctx)
end

struct whisper_timings
    sample_ms::Cfloat
    encode_ms::Cfloat
    decode_ms::Cfloat
    batchd_ms::Cfloat
    prompt_ms::Cfloat
end

function whisper_get_timings(ctx)
    ccall((:whisper_get_timings, libwhisper), Ptr{whisper_timings}, (Ptr{whisper_context},), ctx)
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

function whisper_context_default_params_by_ref()
    ccall((:whisper_context_default_params_by_ref, libwhisper), Ptr{whisper_context_params}, ())
end

function whisper_context_default_params()
    ccall((:whisper_context_default_params, libwhisper), whisper_context_params, ())
end

function whisper_full_default_params_by_ref(strategy)
    ccall((:whisper_full_default_params_by_ref, libwhisper), Ptr{whisper_full_params}, (whisper_sampling_strategy,), strategy)
end

function whisper_full_default_params(strategy)
    ccall((:whisper_full_default_params, libwhisper), whisper_full_params, (whisper_sampling_strategy,), strategy)
end

function whisper_full(ctx, params, samples, n_samples)
    ccall((:whisper_full, libwhisper), Cint, (Ptr{whisper_context}, whisper_full_params, Ptr{Cfloat}, Cint), ctx, params, samples, n_samples)
end

function whisper_full_with_state(ctx, state, params, samples, n_samples)
    ccall((:whisper_full_with_state, libwhisper), Cint, (Ptr{whisper_context}, Ptr{whisper_state}, whisper_full_params, Ptr{Cfloat}, Cint), ctx, state, params, samples, n_samples)
end

function whisper_full_parallel(ctx, params, samples, n_samples, n_processors)
    ccall((:whisper_full_parallel, libwhisper), Cint, (Ptr{whisper_context}, whisper_full_params, Ptr{Cfloat}, Cint, Cint), ctx, params, samples, n_samples, n_processors)
end

function whisper_full_n_segments(ctx)
    ccall((:whisper_full_n_segments, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_full_n_segments_from_state(state)
    ccall((:whisper_full_n_segments_from_state, libwhisper), Cint, (Ptr{whisper_state},), state)
end

function whisper_full_lang_id(ctx)
    ccall((:whisper_full_lang_id, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_full_lang_id_from_state(state)
    ccall((:whisper_full_lang_id_from_state, libwhisper), Cint, (Ptr{whisper_state},), state)
end

function whisper_full_get_segment_t0(ctx, i_segment)
    ccall((:whisper_full_get_segment_t0, libwhisper), Int64, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_get_segment_t0_from_state(state, i_segment)
    ccall((:whisper_full_get_segment_t0_from_state, libwhisper), Int64, (Ptr{whisper_state}, Cint), state, i_segment)
end

function whisper_full_get_segment_t1(ctx, i_segment)
    ccall((:whisper_full_get_segment_t1, libwhisper), Int64, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_get_segment_t1_from_state(state, i_segment)
    ccall((:whisper_full_get_segment_t1_from_state, libwhisper), Int64, (Ptr{whisper_state}, Cint), state, i_segment)
end

function whisper_full_get_segment_speaker_turn_next(ctx, i_segment)
    ccall((:whisper_full_get_segment_speaker_turn_next, libwhisper), Bool, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_get_segment_speaker_turn_next_from_state(state, i_segment)
    ccall((:whisper_full_get_segment_speaker_turn_next_from_state, libwhisper), Bool, (Ptr{whisper_state}, Cint), state, i_segment)
end

function whisper_full_get_segment_text(ctx, i_segment)
    ccall((:whisper_full_get_segment_text, libwhisper), Ptr{Cchar}, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_get_segment_text_from_state(state, i_segment)
    ccall((:whisper_full_get_segment_text_from_state, libwhisper), Ptr{Cchar}, (Ptr{whisper_state}, Cint), state, i_segment)
end

function whisper_full_n_tokens(ctx, i_segment)
    ccall((:whisper_full_n_tokens, libwhisper), Cint, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_n_tokens_from_state(state, i_segment)
    ccall((:whisper_full_n_tokens_from_state, libwhisper), Cint, (Ptr{whisper_state}, Cint), state, i_segment)
end

function whisper_full_get_token_text(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_text, libwhisper), Ptr{Cchar}, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_text_from_state(ctx, state, i_segment, i_token)
    ccall((:whisper_full_get_token_text_from_state, libwhisper), Ptr{Cchar}, (Ptr{whisper_context}, Ptr{whisper_state}, Cint, Cint), ctx, state, i_segment, i_token)
end

function whisper_full_get_token_id(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_id, libwhisper), whisper_token, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_id_from_state(state, i_segment, i_token)
    ccall((:whisper_full_get_token_id_from_state, libwhisper), whisper_token, (Ptr{whisper_state}, Cint, Cint), state, i_segment, i_token)
end

function whisper_full_get_token_data(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_data, libwhisper), whisper_token_data, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_data_from_state(state, i_segment, i_token)
    ccall((:whisper_full_get_token_data_from_state, libwhisper), whisper_token_data, (Ptr{whisper_state}, Cint, Cint), state, i_segment, i_token)
end

function whisper_full_get_token_t0(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_t0, libwhisper), Int64, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_t0_from_state(state, i_segment, i_token)
    ccall((:whisper_full_get_token_t0_from_state, libwhisper), Int64, (Ptr{whisper_state}, Cint, Cint), state, i_segment, i_token)
end

function whisper_full_get_token_t1(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_t1, libwhisper), Int64, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_t1_from_state(state, i_segment, i_token)
    ccall((:whisper_full_get_token_t1_from_state, libwhisper), Int64, (Ptr{whisper_state}, Cint, Cint), state, i_segment, i_token)
end

function whisper_full_get_token_p(ctx, i_segment, i_token)
    ccall((:whisper_full_get_token_p, libwhisper), Cfloat, (Ptr{whisper_context}, Cint, Cint), ctx, i_segment, i_token)
end

function whisper_full_get_token_p_from_state(state, i_segment, i_token)
    ccall((:whisper_full_get_token_p_from_state, libwhisper), Cfloat, (Ptr{whisper_state}, Cint, Cint), state, i_segment, i_token)
end

function whisper_full_n_vad_segments(ctx)
    ccall((:whisper_full_n_vad_segments, libwhisper), Cint, (Ptr{whisper_context},), ctx)
end

function whisper_full_n_vad_segments_from_state(state)
    ccall((:whisper_full_n_vad_segments_from_state, libwhisper), Cint, (Ptr{whisper_state},), state)
end

function whisper_full_get_vad_segment_t0(ctx, i)
    ccall((:whisper_full_get_vad_segment_t0, libwhisper), Int64, (Ptr{whisper_context}, Cint), ctx, i)
end

function whisper_full_get_vad_segment_t0_from_state(state, i)
    ccall((:whisper_full_get_vad_segment_t0_from_state, libwhisper), Int64, (Ptr{whisper_state}, Cint), state, i)
end

function whisper_full_get_vad_segment_t1(ctx, i)
    ccall((:whisper_full_get_vad_segment_t1, libwhisper), Int64, (Ptr{whisper_context}, Cint), ctx, i)
end

function whisper_full_get_vad_segment_t1_from_state(state, i)
    ccall((:whisper_full_get_vad_segment_t1_from_state, libwhisper), Int64, (Ptr{whisper_state}, Cint), state, i)
end

mutable struct whisper_vad_context end

function whisper_vad_default_params()
    ccall((:whisper_vad_default_params, libwhisper), whisper_vad_params, ())
end

struct whisper_vad_context_params
    n_threads::Cint
    use_gpu::Bool
    gpu_device::Cint
end

function whisper_vad_default_context_params()
    ccall((:whisper_vad_default_context_params, libwhisper), whisper_vad_context_params, ())
end

function whisper_vad_init_from_file_with_params(path_model, params)
    ccall((:whisper_vad_init_from_file_with_params, libwhisper), Ptr{whisper_vad_context}, (Ptr{Cchar}, whisper_vad_context_params), path_model, params)
end

function whisper_vad_init_with_params(loader, params)
    ccall((:whisper_vad_init_with_params, libwhisper), Ptr{whisper_vad_context}, (Ptr{whisper_model_loader}, whisper_vad_context_params), loader, params)
end

function whisper_vad_detect_speech(vctx, samples, n_samples)
    ccall((:whisper_vad_detect_speech, libwhisper), Bool, (Ptr{whisper_vad_context}, Ptr{Cfloat}, Cint), vctx, samples, n_samples)
end

function whisper_vad_detect_speech_no_reset(vctx, samples, n_samples)
    ccall((:whisper_vad_detect_speech_no_reset, libwhisper), Bool, (Ptr{whisper_vad_context}, Ptr{Cfloat}, Cint), vctx, samples, n_samples)
end

function whisper_vad_reset_state(vctx)
    ccall((:whisper_vad_reset_state, libwhisper), Cvoid, (Ptr{whisper_vad_context},), vctx)
end

function whisper_vad_n_probs(vctx)
    ccall((:whisper_vad_n_probs, libwhisper), Cint, (Ptr{whisper_vad_context},), vctx)
end

function whisper_vad_probs(vctx)
    ccall((:whisper_vad_probs, libwhisper), Ptr{Cfloat}, (Ptr{whisper_vad_context},), vctx)
end

mutable struct whisper_vad_segments end

function whisper_vad_segments_from_probs(vctx, params)
    ccall((:whisper_vad_segments_from_probs, libwhisper), Ptr{whisper_vad_segments}, (Ptr{whisper_vad_context}, whisper_vad_params), vctx, params)
end

function whisper_vad_segments_from_samples(vctx, params, samples, n_samples)
    ccall((:whisper_vad_segments_from_samples, libwhisper), Ptr{whisper_vad_segments}, (Ptr{whisper_vad_context}, whisper_vad_params, Ptr{Cfloat}, Cint), vctx, params, samples, n_samples)
end

function whisper_vad_segments_n_segments(segments)
    ccall((:whisper_vad_segments_n_segments, libwhisper), Cint, (Ptr{whisper_vad_segments},), segments)
end

function whisper_vad_segments_get_segment_t0(segments, i_segment)
    ccall((:whisper_vad_segments_get_segment_t0, libwhisper), Cfloat, (Ptr{whisper_vad_segments}, Cint), segments, i_segment)
end

function whisper_vad_segments_get_segment_t1(segments, i_segment)
    ccall((:whisper_vad_segments_get_segment_t1, libwhisper), Cfloat, (Ptr{whisper_vad_segments}, Cint), segments, i_segment)
end

function whisper_vad_free_segments(segments)
    ccall((:whisper_vad_free_segments, libwhisper), Cvoid, (Ptr{whisper_vad_segments},), segments)
end

function whisper_vad_free(ctx)
    ccall((:whisper_vad_free, libwhisper), Cvoid, (Ptr{whisper_vad_context},), ctx)
end

function whisper_bench_memcpy(n_threads)
    ccall((:whisper_bench_memcpy, libwhisper), Cint, (Cint,), n_threads)
end

function whisper_bench_memcpy_str(n_threads)
    ccall((:whisper_bench_memcpy_str, libwhisper), Ptr{Cchar}, (Cint,), n_threads)
end

function whisper_bench_ggml_mul_mat(n_threads)
    ccall((:whisper_bench_ggml_mul_mat, libwhisper), Cint, (Cint,), n_threads)
end

function whisper_bench_ggml_mul_mat_str(n_threads)
    ccall((:whisper_bench_ggml_mul_mat_str, libwhisper), Ptr{Cchar}, (Cint,), n_threads)
end

# typedef void ( * ggml_log_callback ) ( enum ggml_log_level level , const char * text , void * user_data )
const ggml_log_callback = Ptr{Cvoid}

function whisper_log_set(log_callback, user_data)
    ccall((:whisper_log_set, libwhisper), Cvoid, (ggml_log_callback, Ptr{Cvoid}), log_callback, user_data)
end

function whisper_full_get_segment_no_speech_prob(ctx, i_segment)
    ccall((:whisper_full_get_segment_no_speech_prob, libwhisper), Cfloat, (Ptr{whisper_context}, Cint), ctx, i_segment)
end

function whisper_full_get_segment_no_speech_prob_from_state(state, i_segment)
    ccall((:whisper_full_get_segment_no_speech_prob_from_state, libwhisper), Cfloat, (Ptr{whisper_state}, Cint), state, i_segment)
end

# typedef void ( * ggml_abort_callback_t ) ( const char * error_message )
const ggml_abort_callback_t = Ptr{Cvoid}

function ggml_set_abort_callback(callback)
    ccall((:ggml_set_abort_callback, libwhisper), ggml_abort_callback_t, (ggml_abort_callback_t,), callback)
end

@cenum ggml_status::Int32 begin
    GGML_STATUS_ALLOC_FAILED = -2
    GGML_STATUS_FAILED = -1
    GGML_STATUS_SUCCESS = 0
    GGML_STATUS_ABORTED = 1
end

function ggml_status_to_string(status)
    ccall((:ggml_status_to_string, libwhisper), Ptr{Cchar}, (ggml_status,), status)
end

const ggml_fp16_t = UInt16

function ggml_fp16_to_fp32(arg1)
    ccall((:ggml_fp16_to_fp32, libwhisper), Cfloat, (ggml_fp16_t,), arg1)
end

function ggml_fp32_to_fp16(arg1)
    ccall((:ggml_fp32_to_fp16, libwhisper), ggml_fp16_t, (Cfloat,), arg1)
end

function ggml_fp16_to_fp32_row(arg1, arg2, arg3)
    ccall((:ggml_fp16_to_fp32_row, libwhisper), Cvoid, (Ptr{ggml_fp16_t}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

function ggml_fp32_to_fp16_row(arg1, arg2, arg3)
    ccall((:ggml_fp32_to_fp16_row, libwhisper), Cvoid, (Ptr{Cfloat}, Ptr{ggml_fp16_t}, Int64), arg1, arg2, arg3)
end

struct ggml_bf16_t
    bits::UInt16
end

function ggml_fp32_to_bf16(arg1)
    ccall((:ggml_fp32_to_bf16, libwhisper), ggml_bf16_t, (Cfloat,), arg1)
end

function ggml_bf16_to_fp32(arg1)
    ccall((:ggml_bf16_to_fp32, libwhisper), Cfloat, (ggml_bf16_t,), arg1)
end

function ggml_bf16_to_fp32_row(arg1, arg2, arg3)
    ccall((:ggml_bf16_to_fp32_row, libwhisper), Cvoid, (Ptr{ggml_bf16_t}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

function ggml_fp32_to_bf16_row_ref(arg1, arg2, arg3)
    ccall((:ggml_fp32_to_bf16_row_ref, libwhisper), Cvoid, (Ptr{Cfloat}, Ptr{ggml_bf16_t}, Int64), arg1, arg2, arg3)
end

function ggml_fp32_to_bf16_row(arg1, arg2, arg3)
    ccall((:ggml_fp32_to_bf16_row, libwhisper), Cvoid, (Ptr{Cfloat}, Ptr{ggml_bf16_t}, Int64), arg1, arg2, arg3)
end

mutable struct ggml_object end

@cenum ggml_type::UInt32 begin
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S = 19
    GGML_TYPE_IQ4_NL = 20
    GGML_TYPE_IQ3_S = 21
    GGML_TYPE_IQ2_S = 22
    GGML_TYPE_IQ4_XS = 23
    GGML_TYPE_I8 = 24
    GGML_TYPE_I16 = 25
    GGML_TYPE_I32 = 26
    GGML_TYPE_I64 = 27
    GGML_TYPE_F64 = 28
    GGML_TYPE_IQ1_M = 29
    GGML_TYPE_BF16 = 30
    GGML_TYPE_TQ1_0 = 34
    GGML_TYPE_TQ2_0 = 35
    GGML_TYPE_MXFP4 = 39
    GGML_TYPE_NVFP4 = 40
    GGML_TYPE_Q1_0 = 41
    GGML_TYPE_Q2_0 = 42
    GGML_TYPE_COUNT = 43
end

@cenum ggml_prec::UInt32 begin
    GGML_PREC_DEFAULT = 0
    GGML_PREC_F32 = 10
end

@cenum ggml_op_hint::UInt32 begin
    GGML_HINT_NONE = 0
    GGML_HINT_SRC0_IS_HADAMARD = 1
end

@cenum ggml_ftype::Int32 begin
    GGML_FTYPE_UNKNOWN = -1
    GGML_FTYPE_ALL_F32 = 0
    GGML_FTYPE_MOSTLY_F16 = 1
    GGML_FTYPE_MOSTLY_Q4_0 = 2
    GGML_FTYPE_MOSTLY_Q4_1 = 3
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    GGML_FTYPE_MOSTLY_Q8_0 = 7
    GGML_FTYPE_MOSTLY_Q5_0 = 8
    GGML_FTYPE_MOSTLY_Q5_1 = 9
    GGML_FTYPE_MOSTLY_Q2_K = 10
    GGML_FTYPE_MOSTLY_Q3_K = 11
    GGML_FTYPE_MOSTLY_Q4_K = 12
    GGML_FTYPE_MOSTLY_Q5_K = 13
    GGML_FTYPE_MOSTLY_Q6_K = 14
    GGML_FTYPE_MOSTLY_IQ2_XXS = 15
    GGML_FTYPE_MOSTLY_IQ2_XS = 16
    GGML_FTYPE_MOSTLY_IQ3_XXS = 17
    GGML_FTYPE_MOSTLY_IQ1_S = 18
    GGML_FTYPE_MOSTLY_IQ4_NL = 19
    GGML_FTYPE_MOSTLY_IQ3_S = 20
    GGML_FTYPE_MOSTLY_IQ2_S = 21
    GGML_FTYPE_MOSTLY_IQ4_XS = 22
    GGML_FTYPE_MOSTLY_IQ1_M = 23
    GGML_FTYPE_MOSTLY_BF16 = 24
    GGML_FTYPE_MOSTLY_MXFP4 = 25
    GGML_FTYPE_MOSTLY_NVFP4 = 26
    GGML_FTYPE_MOSTLY_Q1_0 = 27
    GGML_FTYPE_MOSTLY_Q2_0 = 28
end

@cenum ggml_op::UInt32 begin
    GGML_OP_NONE = 0
    GGML_OP_DUP = 1
    GGML_OP_ADD = 2
    GGML_OP_ADD_ID = 3
    GGML_OP_ADD1 = 4
    GGML_OP_ACC = 5
    GGML_OP_SUB = 6
    GGML_OP_MUL = 7
    GGML_OP_DIV = 8
    GGML_OP_SQR = 9
    GGML_OP_SQRT = 10
    GGML_OP_LOG = 11
    GGML_OP_SIN = 12
    GGML_OP_COS = 13
    GGML_OP_SUM = 14
    GGML_OP_SUM_ROWS = 15
    GGML_OP_CUMSUM = 16
    GGML_OP_MEAN = 17
    GGML_OP_ARGMAX = 18
    GGML_OP_COUNT_EQUAL = 19
    GGML_OP_REPEAT = 20
    GGML_OP_REPEAT_BACK = 21
    GGML_OP_CONCAT = 22
    GGML_OP_SILU_BACK = 23
    GGML_OP_NORM = 24
    GGML_OP_RMS_NORM = 25
    GGML_OP_RMS_NORM_BACK = 26
    GGML_OP_GROUP_NORM = 27
    GGML_OP_L2_NORM = 28
    GGML_OP_MUL_MAT = 29
    GGML_OP_MUL_MAT_ID = 30
    GGML_OP_OUT_PROD = 31
    GGML_OP_SCALE = 32
    GGML_OP_SET = 33
    GGML_OP_CPY = 34
    GGML_OP_CONT = 35
    GGML_OP_RESHAPE = 36
    GGML_OP_VIEW = 37
    GGML_OP_PERMUTE = 38
    GGML_OP_TRANSPOSE = 39
    GGML_OP_GET_ROWS = 40
    GGML_OP_GET_ROWS_BACK = 41
    GGML_OP_SET_ROWS = 42
    GGML_OP_DIAG = 43
    GGML_OP_DIAG_MASK_INF = 44
    GGML_OP_DIAG_MASK_ZERO = 45
    GGML_OP_SOFT_MAX = 46
    GGML_OP_SOFT_MAX_BACK = 47
    GGML_OP_ROPE = 48
    GGML_OP_ROPE_BACK = 49
    GGML_OP_CLAMP = 50
    GGML_OP_CONV_TRANSPOSE_1D = 51
    GGML_OP_IM2COL = 52
    GGML_OP_IM2COL_BACK = 53
    GGML_OP_IM2COL_3D = 54
    GGML_OP_COL2IM_1D = 55
    GGML_OP_CONV_2D = 56
    GGML_OP_CONV_3D = 57
    GGML_OP_CONV_2D_DW = 58
    GGML_OP_CONV_TRANSPOSE_2D = 59
    GGML_OP_POOL_1D = 60
    GGML_OP_POOL_2D = 61
    GGML_OP_POOL_2D_BACK = 62
    GGML_OP_UPSCALE = 63
    GGML_OP_PAD = 64
    GGML_OP_PAD_REFLECT_1D = 65
    GGML_OP_ROLL = 66
    GGML_OP_ARANGE = 67
    GGML_OP_TIMESTEP_EMBEDDING = 68
    GGML_OP_ARGSORT = 69
    GGML_OP_TOP_K = 70
    GGML_OP_LEAKY_RELU = 71
    GGML_OP_TRI = 72
    GGML_OP_FILL = 73
    GGML_OP_FLASH_ATTN_EXT = 74
    GGML_OP_FLASH_ATTN_BACK = 75
    GGML_OP_SSM_CONV = 76
    GGML_OP_SSM_SCAN = 77
    GGML_OP_WIN_PART = 78
    GGML_OP_WIN_UNPART = 79
    GGML_OP_GET_REL_POS = 80
    GGML_OP_ADD_REL_POS = 81
    GGML_OP_RWKV_WKV6 = 82
    GGML_OP_GATED_LINEAR_ATTN = 83
    GGML_OP_RWKV_WKV7 = 84
    GGML_OP_SOLVE_TRI = 85
    GGML_OP_GATED_DELTA_NET = 86
    GGML_OP_LIGHTNING_INDEXER = 87
    GGML_OP_DSV4_HC_COMB = 88
    GGML_OP_DSV4_HC_PRE = 89
    GGML_OP_DSV4_HC_POST = 90
    GGML_OP_UNARY = 91
    GGML_OP_MAP_CUSTOM1 = 92
    GGML_OP_MAP_CUSTOM2 = 93
    GGML_OP_MAP_CUSTOM3 = 94
    GGML_OP_CUSTOM = 95
    GGML_OP_CROSS_ENTROPY_LOSS = 96
    GGML_OP_CROSS_ENTROPY_LOSS_BACK = 97
    GGML_OP_OPT_STEP_ADAMW = 98
    GGML_OP_OPT_STEP_SGD = 99
    GGML_OP_GLU = 100
    GGML_OP_COUNT = 101
end

@cenum ggml_unary_op::UInt32 begin
    GGML_UNARY_OP_ABS = 0
    GGML_UNARY_OP_SGN = 1
    GGML_UNARY_OP_NEG = 2
    GGML_UNARY_OP_STEP = 3
    GGML_UNARY_OP_TANH = 4
    GGML_UNARY_OP_ELU = 5
    GGML_UNARY_OP_RELU = 6
    GGML_UNARY_OP_SIGMOID = 7
    GGML_UNARY_OP_GELU = 8
    GGML_UNARY_OP_GELU_QUICK = 9
    GGML_UNARY_OP_SILU = 10
    GGML_UNARY_OP_HARDSWISH = 11
    GGML_UNARY_OP_HARDSIGMOID = 12
    GGML_UNARY_OP_EXP = 13
    GGML_UNARY_OP_EXPM1 = 14
    GGML_UNARY_OP_SOFTPLUS = 15
    GGML_UNARY_OP_GELU_ERF = 16
    GGML_UNARY_OP_XIELU = 17
    GGML_UNARY_OP_FLOOR = 18
    GGML_UNARY_OP_CEIL = 19
    GGML_UNARY_OP_ROUND = 20
    GGML_UNARY_OP_TRUNC = 21
    GGML_UNARY_OP_COUNT = 22
end

@cenum ggml_glu_op::UInt32 begin
    GGML_GLU_OP_REGLU = 0
    GGML_GLU_OP_GEGLU = 1
    GGML_GLU_OP_SWIGLU = 2
    GGML_GLU_OP_SWIGLU_OAI = 3
    GGML_GLU_OP_GEGLU_ERF = 4
    GGML_GLU_OP_GEGLU_QUICK = 5
    GGML_GLU_OP_COUNT = 6
end

@cenum ggml_object_type::UInt32 begin
    GGML_OBJECT_TYPE_TENSOR = 0
    GGML_OBJECT_TYPE_GRAPH = 1
    GGML_OBJECT_TYPE_WORK_BUFFER = 2
end

@cenum ggml_log_level::UInt32 begin
    GGML_LOG_LEVEL_NONE = 0
    GGML_LOG_LEVEL_DEBUG = 1
    GGML_LOG_LEVEL_INFO = 2
    GGML_LOG_LEVEL_WARN = 3
    GGML_LOG_LEVEL_ERROR = 4
    GGML_LOG_LEVEL_CONT = 5
end

@cenum ggml_tensor_flag::UInt32 begin
    GGML_TENSOR_FLAG_INPUT = 1
    GGML_TENSOR_FLAG_OUTPUT = 2
    GGML_TENSOR_FLAG_PARAM = 4
    GGML_TENSOR_FLAG_LOSS = 8
    GGML_TENSOR_FLAG_COMPUTE = 16
end

@cenum ggml_tri_type::UInt32 begin
    GGML_TRI_TYPE_UPPER_DIAG = 0
    GGML_TRI_TYPE_UPPER = 1
    GGML_TRI_TYPE_LOWER_DIAG = 2
    GGML_TRI_TYPE_LOWER = 3
end

struct ggml_init_params
    mem_size::Csize_t
    mem_buffer::Ptr{Cvoid}
    no_alloc::Bool
end

mutable struct ggml_backend_buffer end

struct ggml_tensor
    type::ggml_type
    buffer::Ptr{ggml_backend_buffer}
    ne::NTuple{4, Int64}
    nb::NTuple{4, Csize_t}
    op::ggml_op
    op_params::NTuple{16, Int32}
    flags::Int32
    src::NTuple{10, Ptr{ggml_tensor}}
    view_src::Ptr{ggml_tensor}
    view_offs::Csize_t
    data::Ptr{Cvoid}
    name::NTuple{64, Cchar}
    extra::Ptr{Cvoid}
    padding::NTuple{8, Cchar}
end

const ggml_guid = NTuple{16, UInt8}

const ggml_guid_t = Ptr{ggml_guid}

function ggml_guid_matches(guid_a, guid_b)
    ccall((:ggml_guid_matches, libwhisper), Bool, (ggml_guid_t, ggml_guid_t), guid_a, guid_b)
end

function ggml_version()
    ccall((:ggml_version, libwhisper), Ptr{Cchar}, ())
end

function ggml_commit()
    ccall((:ggml_commit, libwhisper), Ptr{Cchar}, ())
end

function ggml_time_init()
    ccall((:ggml_time_init, libwhisper), Cvoid, ())
end

function ggml_time_ms()
    ccall((:ggml_time_ms, libwhisper), Int64, ())
end

function ggml_time_us()
    ccall((:ggml_time_us, libwhisper), Int64, ())
end

function ggml_cycles()
    ccall((:ggml_cycles, libwhisper), Int64, ())
end

function ggml_cycles_per_ms()
    ccall((:ggml_cycles_per_ms, libwhisper), Int64, ())
end

function ggml_fopen(fname, mode)
    ccall((:ggml_fopen, libwhisper), Ptr{Libc.FILE}, (Ptr{Cchar}, Ptr{Cchar}), fname, mode)
end

function ggml_print_object(obj)
    ccall((:ggml_print_object, libwhisper), Cvoid, (Ptr{ggml_object},), obj)
end

mutable struct ggml_context end

function ggml_print_objects(ctx)
    ccall((:ggml_print_objects, libwhisper), Cvoid, (Ptr{ggml_context},), ctx)
end

function ggml_nelements(tensor)
    ccall((:ggml_nelements, libwhisper), Int64, (Ptr{ggml_tensor},), tensor)
end

function ggml_nrows(tensor)
    ccall((:ggml_nrows, libwhisper), Int64, (Ptr{ggml_tensor},), tensor)
end

function ggml_nbytes(tensor)
    ccall((:ggml_nbytes, libwhisper), Csize_t, (Ptr{ggml_tensor},), tensor)
end

function ggml_nbytes_pad(tensor)
    ccall((:ggml_nbytes_pad, libwhisper), Csize_t, (Ptr{ggml_tensor},), tensor)
end

function ggml_blck_size(type)
    ccall((:ggml_blck_size, libwhisper), Int64, (ggml_type,), type)
end

function ggml_type_size(type)
    ccall((:ggml_type_size, libwhisper), Csize_t, (ggml_type,), type)
end

function ggml_row_size(type, ne)
    ccall((:ggml_row_size, libwhisper), Csize_t, (ggml_type, Int64), type, ne)
end

function ggml_type_sizef(type)
    ccall((:ggml_type_sizef, libwhisper), Cdouble, (ggml_type,), type)
end

function ggml_type_name(type)
    ccall((:ggml_type_name, libwhisper), Ptr{Cchar}, (ggml_type,), type)
end

function ggml_op_name(op)
    ccall((:ggml_op_name, libwhisper), Ptr{Cchar}, (ggml_op,), op)
end

function ggml_op_symbol(op)
    ccall((:ggml_op_symbol, libwhisper), Ptr{Cchar}, (ggml_op,), op)
end

function ggml_unary_op_name(op)
    ccall((:ggml_unary_op_name, libwhisper), Ptr{Cchar}, (ggml_unary_op,), op)
end

function ggml_glu_op_name(op)
    ccall((:ggml_glu_op_name, libwhisper), Ptr{Cchar}, (ggml_glu_op,), op)
end

function ggml_op_desc(t)
    ccall((:ggml_op_desc, libwhisper), Ptr{Cchar}, (Ptr{ggml_tensor},), t)
end

function ggml_element_size(tensor)
    ccall((:ggml_element_size, libwhisper), Csize_t, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_quantized(type)
    ccall((:ggml_is_quantized, libwhisper), Bool, (ggml_type,), type)
end

function ggml_ftype_to_ggml_type(ftype)
    ccall((:ggml_ftype_to_ggml_type, libwhisper), ggml_type, (ggml_ftype,), ftype)
end

function ggml_is_transposed(tensor)
    ccall((:ggml_is_transposed, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_permuted(tensor)
    ccall((:ggml_is_permuted, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_empty(tensor)
    ccall((:ggml_is_empty, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_view(tensor)
    ccall((:ggml_is_view, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_scalar(tensor)
    ccall((:ggml_is_scalar, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_vector(tensor)
    ccall((:ggml_is_vector, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_matrix(tensor)
    ccall((:ggml_is_matrix, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_3d(tensor)
    ccall((:ggml_is_3d, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_n_dims(tensor)
    ccall((:ggml_n_dims, libwhisper), Cint, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous(tensor)
    ccall((:ggml_is_contiguous, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_0(tensor)
    ccall((:ggml_is_contiguous_0, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_1(tensor)
    ccall((:ggml_is_contiguous_1, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_2(tensor)
    ccall((:ggml_is_contiguous_2, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_to_1(tensor)
    ccall((:ggml_is_contiguous_to_1, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_to_2(tensor)
    ccall((:ggml_is_contiguous_to_2, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_to_3(tensor)
    ccall((:ggml_is_contiguous_to_3, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguously_allocated(tensor)
    ccall((:ggml_is_contiguously_allocated, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_channels(tensor)
    ccall((:ggml_is_contiguous_channels, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_rows(tensor)
    ccall((:ggml_is_contiguous_rows, libwhisper), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_are_same_shape(t0, t1)
    ccall((:ggml_are_same_shape, libwhisper), Bool, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), t0, t1)
end

function ggml_are_same_stride(t0, t1)
    ccall((:ggml_are_same_stride, libwhisper), Bool, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), t0, t1)
end

function ggml_can_repeat(t0, t1)
    ccall((:ggml_can_repeat, libwhisper), Bool, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), t0, t1)
end

function ggml_tensor_overhead()
    ccall((:ggml_tensor_overhead, libwhisper), Csize_t, ())
end

function ggml_validate_row_data(type, data, nbytes)
    ccall((:ggml_validate_row_data, libwhisper), Bool, (ggml_type, Ptr{Cvoid}, Csize_t), type, data, nbytes)
end

function ggml_init(params)
    ccall((:ggml_init, libwhisper), Ptr{ggml_context}, (ggml_init_params,), params)
end

function ggml_reset(ctx)
    ccall((:ggml_reset, libwhisper), Cvoid, (Ptr{ggml_context},), ctx)
end

function ggml_free(ctx)
    ccall((:ggml_free, libwhisper), Cvoid, (Ptr{ggml_context},), ctx)
end

function ggml_used_mem(ctx)
    ccall((:ggml_used_mem, libwhisper), Csize_t, (Ptr{ggml_context},), ctx)
end

function ggml_get_no_alloc(ctx)
    ccall((:ggml_get_no_alloc, libwhisper), Bool, (Ptr{ggml_context},), ctx)
end

function ggml_set_no_alloc(ctx, no_alloc)
    ccall((:ggml_set_no_alloc, libwhisper), Cvoid, (Ptr{ggml_context}, Bool), ctx, no_alloc)
end

function ggml_get_mem_buffer(ctx)
    ccall((:ggml_get_mem_buffer, libwhisper), Ptr{Cvoid}, (Ptr{ggml_context},), ctx)
end

function ggml_get_mem_size(ctx)
    ccall((:ggml_get_mem_size, libwhisper), Csize_t, (Ptr{ggml_context},), ctx)
end

function ggml_get_max_tensor_size(ctx)
    ccall((:ggml_get_max_tensor_size, libwhisper), Csize_t, (Ptr{ggml_context},), ctx)
end

function ggml_new_tensor(ctx, type, n_dims, ne)
    ccall((:ggml_new_tensor, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Cint, Ptr{Int64}), ctx, type, n_dims, ne)
end

function ggml_new_tensor_1d(ctx, type, ne0)
    ccall((:ggml_new_tensor_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64), ctx, type, ne0)
end

function ggml_new_tensor_2d(ctx, type, ne0, ne1)
    ccall((:ggml_new_tensor_2d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64), ctx, type, ne0, ne1)
end

function ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)
    ccall((:ggml_new_tensor_3d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64, Int64), ctx, type, ne0, ne1, ne2)
end

function ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3)
    ccall((:ggml_new_tensor_4d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64, Int64, Int64), ctx, type, ne0, ne1, ne2, ne3)
end

function ggml_new_buffer(ctx, nbytes)
    ccall((:ggml_new_buffer, libwhisper), Ptr{Cvoid}, (Ptr{ggml_context}, Csize_t), ctx, nbytes)
end

function ggml_dup_tensor(ctx, src)
    ccall((:ggml_dup_tensor, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, src)
end

function ggml_view_tensor(ctx, src)
    ccall((:ggml_view_tensor, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, src)
end

function ggml_get_first_tensor(ctx)
    ccall((:ggml_get_first_tensor, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context},), ctx)
end

function ggml_get_next_tensor(ctx, tensor)
    ccall((:ggml_get_next_tensor, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, tensor)
end

function ggml_get_tensor(ctx, name)
    ccall((:ggml_get_tensor, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{Cchar}), ctx, name)
end

function ggml_unravel_index(tensor, i, i0, i1, i2, i3)
    ccall((:ggml_unravel_index, libwhisper), Cvoid, (Ptr{ggml_tensor}, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}), tensor, i, i0, i1, i2, i3)
end

function ggml_get_unary_op(tensor)
    ccall((:ggml_get_unary_op, libwhisper), ggml_unary_op, (Ptr{ggml_tensor},), tensor)
end

function ggml_get_glu_op(tensor)
    ccall((:ggml_get_glu_op, libwhisper), ggml_glu_op, (Ptr{ggml_tensor},), tensor)
end

function ggml_get_data(tensor)
    ccall((:ggml_get_data, libwhisper), Ptr{Cvoid}, (Ptr{ggml_tensor},), tensor)
end

function ggml_get_data_f32(tensor)
    ccall((:ggml_get_data_f32, libwhisper), Ptr{Cfloat}, (Ptr{ggml_tensor},), tensor)
end

function ggml_get_name(tensor)
    ccall((:ggml_get_name, libwhisper), Ptr{Cchar}, (Ptr{ggml_tensor},), tensor)
end

function ggml_set_name(tensor, name)
    ccall((:ggml_set_name, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Ptr{Cchar}), tensor, name)
end

function ggml_set_input(tensor)
    ccall((:ggml_set_input, libwhisper), Cvoid, (Ptr{ggml_tensor},), tensor)
end

function ggml_set_output(tensor)
    ccall((:ggml_set_output, libwhisper), Cvoid, (Ptr{ggml_tensor},), tensor)
end

function ggml_set_param(tensor)
    ccall((:ggml_set_param, libwhisper), Cvoid, (Ptr{ggml_tensor},), tensor)
end

function ggml_set_loss(tensor)
    ccall((:ggml_set_loss, libwhisper), Cvoid, (Ptr{ggml_tensor},), tensor)
end

function ggml_dup(ctx, a)
    ccall((:ggml_dup, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_dup_inplace(ctx, a)
    ccall((:ggml_dup_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_add(ctx, a, b)
    ccall((:ggml_add, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_add_inplace(ctx, a, b)
    ccall((:ggml_add_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_add_cast(ctx, a, b, type)
    ccall((:ggml_add_cast, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_type), ctx, a, b, type)
end

function ggml_add_id(ctx, a, b, ids)
    ccall((:ggml_add_id, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, ids)
end

function ggml_add1(ctx, a, b)
    ccall((:ggml_add1, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_add1_inplace(ctx, a, b)
    ccall((:ggml_add1_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_acc(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_acc, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

function ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_acc_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

function ggml_sub(ctx, a, b)
    ccall((:ggml_sub, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_sub_inplace(ctx, a, b)
    ccall((:ggml_sub_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_mul(ctx, a, b)
    ccall((:ggml_mul, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_mul_inplace(ctx, a, b)
    ccall((:ggml_mul_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_div(ctx, a, b)
    ccall((:ggml_div, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_div_inplace(ctx, a, b)
    ccall((:ggml_div_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_sqr(ctx, a)
    ccall((:ggml_sqr, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sqr_inplace(ctx, a)
    ccall((:ggml_sqr_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sqrt(ctx, a)
    ccall((:ggml_sqrt, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sqrt_inplace(ctx, a)
    ccall((:ggml_sqrt_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_log(ctx, a)
    ccall((:ggml_log, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_log_inplace(ctx, a)
    ccall((:ggml_log_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_expm1(ctx, a)
    ccall((:ggml_expm1, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_expm1_inplace(ctx, a)
    ccall((:ggml_expm1_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_softplus(ctx, a)
    ccall((:ggml_softplus, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_softplus_inplace(ctx, a)
    ccall((:ggml_softplus_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sin(ctx, a)
    ccall((:ggml_sin, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sin_inplace(ctx, a)
    ccall((:ggml_sin_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_cos(ctx, a)
    ccall((:ggml_cos, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_cos_inplace(ctx, a)
    ccall((:ggml_cos_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sum(ctx, a)
    ccall((:ggml_sum, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sum_rows(ctx, a)
    ccall((:ggml_sum_rows, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_cumsum(ctx, a)
    ccall((:ggml_cumsum, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_mean(ctx, a)
    ccall((:ggml_mean, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_argmax(ctx, a)
    ccall((:ggml_argmax, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_count_equal(ctx, a, b)
    ccall((:ggml_count_equal, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_repeat(ctx, a, b)
    ccall((:ggml_repeat, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_repeat_4d(ctx, a, ne0, ne1, ne2, ne3)
    ccall((:ggml_repeat_4d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2, ne3)
end

function ggml_repeat_back(ctx, a, b)
    ccall((:ggml_repeat_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_concat(ctx, a, b, dim)
    ccall((:ggml_concat, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint), ctx, a, b, dim)
end

function ggml_abs(ctx, a)
    ccall((:ggml_abs, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_abs_inplace(ctx, a)
    ccall((:ggml_abs_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sgn(ctx, a)
    ccall((:ggml_sgn, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sgn_inplace(ctx, a)
    ccall((:ggml_sgn_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_neg(ctx, a)
    ccall((:ggml_neg, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_neg_inplace(ctx, a)
    ccall((:ggml_neg_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_step(ctx, a)
    ccall((:ggml_step, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_step_inplace(ctx, a)
    ccall((:ggml_step_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_tanh(ctx, a)
    ccall((:ggml_tanh, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_tanh_inplace(ctx, a)
    ccall((:ggml_tanh_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_elu(ctx, a)
    ccall((:ggml_elu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_elu_inplace(ctx, a)
    ccall((:ggml_elu_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_relu(ctx, a)
    ccall((:ggml_relu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_leaky_relu(ctx, a, negative_slope, inplace)
    ccall((:ggml_leaky_relu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Bool), ctx, a, negative_slope, inplace)
end

function ggml_relu_inplace(ctx, a)
    ccall((:ggml_relu_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sigmoid(ctx, a)
    ccall((:ggml_sigmoid, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sigmoid_inplace(ctx, a)
    ccall((:ggml_sigmoid_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu(ctx, a)
    ccall((:ggml_gelu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_inplace(ctx, a)
    ccall((:ggml_gelu_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_erf(ctx, a)
    ccall((:ggml_gelu_erf, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_erf_inplace(ctx, a)
    ccall((:ggml_gelu_erf_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_quick(ctx, a)
    ccall((:ggml_gelu_quick, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_quick_inplace(ctx, a)
    ccall((:ggml_gelu_quick_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_silu(ctx, a)
    ccall((:ggml_silu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_silu_inplace(ctx, a)
    ccall((:ggml_silu_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_silu_back(ctx, a, b)
    ccall((:ggml_silu_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_hardswish(ctx, a)
    ccall((:ggml_hardswish, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_hardsigmoid(ctx, a)
    ccall((:ggml_hardsigmoid, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_exp(ctx, a)
    ccall((:ggml_exp, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_exp_inplace(ctx, a)
    ccall((:ggml_exp_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_floor(ctx, a)
    ccall((:ggml_floor, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_floor_inplace(ctx, a)
    ccall((:ggml_floor_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_ceil(ctx, a)
    ccall((:ggml_ceil, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_ceil_inplace(ctx, a)
    ccall((:ggml_ceil_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_round(ctx, a)
    ccall((:ggml_round, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_round_inplace(ctx, a)
    ccall((:ggml_round_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_trunc(ctx, a)
    ccall((:ggml_trunc, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_trunc_inplace(ctx, a)
    ccall((:ggml_trunc_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_xielu(ctx, a, alpha_n, alpha_p, beta, eps)
    ccall((:ggml_xielu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, alpha_n, alpha_p, beta, eps)
end

function ggml_glu(ctx, a, op, swapped)
    ccall((:ggml_glu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_glu_op, Bool), ctx, a, op, swapped)
end

function ggml_reglu(ctx, a)
    ccall((:ggml_reglu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_reglu_swapped(ctx, a)
    ccall((:ggml_reglu_swapped, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu(ctx, a)
    ccall((:ggml_geglu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_swapped(ctx, a)
    ccall((:ggml_geglu_swapped, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_swiglu(ctx, a)
    ccall((:ggml_swiglu, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_swiglu_swapped(ctx, a)
    ccall((:ggml_swiglu_swapped, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_erf(ctx, a)
    ccall((:ggml_geglu_erf, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_erf_swapped(ctx, a)
    ccall((:ggml_geglu_erf_swapped, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_quick(ctx, a)
    ccall((:ggml_geglu_quick, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_quick_swapped(ctx, a)
    ccall((:ggml_geglu_quick_swapped, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_glu_split(ctx, a, b, op)
    ccall((:ggml_glu_split, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_glu_op), ctx, a, b, op)
end

function ggml_reglu_split(ctx, a, b)
    ccall((:ggml_reglu_split, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_geglu_split(ctx, a, b)
    ccall((:ggml_geglu_split, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_swiglu_split(ctx, a, b)
    ccall((:ggml_swiglu_split, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_geglu_erf_split(ctx, a, b)
    ccall((:ggml_geglu_erf_split, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_geglu_quick_split(ctx, a, b)
    ccall((:ggml_geglu_quick_split, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_swiglu_oai(ctx, a, b, alpha, limit)
    ccall((:ggml_swiglu_oai, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, b, alpha, limit)
end

function ggml_norm(ctx, a, eps)
    ccall((:ggml_norm, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_norm_inplace(ctx, a, eps)
    ccall((:ggml_norm_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_rms_norm(ctx, a, eps)
    ccall((:ggml_rms_norm, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_rms_norm_inplace(ctx, a, eps)
    ccall((:ggml_rms_norm_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_group_norm(ctx, a, n_groups, eps)
    ccall((:ggml_group_norm, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cfloat), ctx, a, n_groups, eps)
end

function ggml_group_norm_inplace(ctx, a, n_groups, eps)
    ccall((:ggml_group_norm_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cfloat), ctx, a, n_groups, eps)
end

function ggml_l2_norm(ctx, a, eps)
    ccall((:ggml_l2_norm, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_l2_norm_inplace(ctx, a, eps)
    ccall((:ggml_l2_norm_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_rms_norm_back(ctx, a, b, eps)
    ccall((:ggml_rms_norm_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat), ctx, a, b, eps)
end

function ggml_mul_mat(ctx, a, b)
    ccall((:ggml_mul_mat, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_mul_mat_set_prec(a, prec)
    ccall((:ggml_mul_mat_set_prec, libwhisper), Cvoid, (Ptr{ggml_tensor}, ggml_prec), a, prec)
end

function ggml_mul_mat_set_hint(a, hint)
    ccall((:ggml_mul_mat_set_hint, libwhisper), Cvoid, (Ptr{ggml_tensor}, ggml_op_hint), a, hint)
end

function ggml_mul_mat_id(ctx, as, b, ids)
    ccall((:ggml_mul_mat_id, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, as, b, ids)
end

function ggml_out_prod(ctx, a, b)
    ccall((:ggml_out_prod, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_scale(ctx, a, s)
    ccall((:ggml_scale, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, s)
end

function ggml_scale_inplace(ctx, a, s)
    ccall((:ggml_scale_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, s)
end

function ggml_scale_bias(ctx, a, s, b)
    ccall((:ggml_scale_bias, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, s, b)
end

function ggml_scale_bias_inplace(ctx, a, s, b)
    ccall((:ggml_scale_bias_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, s, b)
end

function ggml_set(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_set, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

function ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_set_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

function ggml_set_1d(ctx, a, b, offset)
    ccall((:ggml_set_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t), ctx, a, b, offset)
end

function ggml_set_1d_inplace(ctx, a, b, offset)
    ccall((:ggml_set_1d_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t), ctx, a, b, offset)
end

function ggml_set_2d(ctx, a, b, nb1, offset)
    ccall((:ggml_set_2d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t), ctx, a, b, nb1, offset)
end

function ggml_set_2d_inplace(ctx, a, b, nb1, offset)
    ccall((:ggml_set_2d_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t), ctx, a, b, nb1, offset)
end

function ggml_cpy(ctx, a, b)
    ccall((:ggml_cpy, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_cast(ctx, a, type)
    ccall((:ggml_cast, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_type), ctx, a, type)
end

function ggml_cont(ctx, a)
    ccall((:ggml_cont, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_cont_1d(ctx, a, ne0)
    ccall((:ggml_cont_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64), ctx, a, ne0)
end

function ggml_cont_2d(ctx, a, ne0, ne1)
    ccall((:ggml_cont_2d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64), ctx, a, ne0, ne1)
end

function ggml_cont_3d(ctx, a, ne0, ne1, ne2)
    ccall((:ggml_cont_3d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2)
end

function ggml_cont_4d(ctx, a, ne0, ne1, ne2, ne3)
    ccall((:ggml_cont_4d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2, ne3)
end

function ggml_reshape(ctx, a, b)
    ccall((:ggml_reshape, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_reshape_1d(ctx, a, ne0)
    ccall((:ggml_reshape_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64), ctx, a, ne0)
end

function ggml_reshape_2d(ctx, a, ne0, ne1)
    ccall((:ggml_reshape_2d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64), ctx, a, ne0, ne1)
end

function ggml_reshape_3d(ctx, a, ne0, ne1, ne2)
    ccall((:ggml_reshape_3d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2)
end

function ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)
    ccall((:ggml_reshape_4d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2, ne3)
end

function ggml_view_1d(ctx, a, ne0, offset)
    ccall((:ggml_view_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Csize_t), ctx, a, ne0, offset)
end

function ggml_view_2d(ctx, a, ne0, ne1, nb1, offset)
    ccall((:ggml_view_2d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Csize_t, Csize_t), ctx, a, ne0, ne1, nb1, offset)
end

function ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset)
    ccall((:ggml_view_3d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Csize_t, Csize_t, Csize_t), ctx, a, ne0, ne1, ne2, nb1, nb2, offset)
end

function ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)
    ccall((:ggml_view_4d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)
end

function ggml_permute(ctx, a, axis0, axis1, axis2, axis3)
    ccall((:ggml_permute, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, axis0, axis1, axis2, axis3)
end

function ggml_transpose(ctx, a)
    ccall((:ggml_transpose, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_get_rows(ctx, a, b)
    ccall((:ggml_get_rows, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_get_rows_back(ctx, a, b, c)
    ccall((:ggml_get_rows_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, c)
end

function ggml_set_rows(ctx, a, b, c)
    ccall((:ggml_set_rows, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, c)
end

function ggml_diag(ctx, a)
    ccall((:ggml_diag, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_diag_mask_inf(ctx, a, n_past)
    ccall((:ggml_diag_mask_inf, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

function ggml_diag_mask_inf_inplace(ctx, a, n_past)
    ccall((:ggml_diag_mask_inf_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

function ggml_diag_mask_zero(ctx, a, n_past)
    ccall((:ggml_diag_mask_zero, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

function ggml_diag_mask_zero_inplace(ctx, a, n_past)
    ccall((:ggml_diag_mask_zero_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

function ggml_soft_max(ctx, a)
    ccall((:ggml_soft_max, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_soft_max_inplace(ctx, a)
    ccall((:ggml_soft_max_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_soft_max_ext(ctx, a, mask, scale, max_bias)
    ccall((:ggml_soft_max_ext, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, mask, scale, max_bias)
end

function ggml_soft_max_ext_inplace(ctx, a, mask, scale, max_bias)
    ccall((:ggml_soft_max_ext_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, mask, scale, max_bias)
end

function ggml_soft_max_add_sinks(a, sinks)
    ccall((:ggml_soft_max_add_sinks, libwhisper), Cvoid, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), a, sinks)
end

function ggml_soft_max_ext_back(ctx, a, b, scale, max_bias)
    ccall((:ggml_soft_max_ext_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, b, scale, max_bias)
end

function ggml_soft_max_ext_back_inplace(ctx, a, b, scale, max_bias)
    ccall((:ggml_soft_max_ext_back_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, b, scale, max_bias)
end

function ggml_rope(ctx, a, b, n_dims, mode)
    ccall((:ggml_rope, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, n_dims, mode)
end

function ggml_rope_inplace(ctx, a, b, n_dims, mode)
    ccall((:ggml_rope_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, n_dims, mode)
end

function ggml_rope_ext(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_ext, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_multi(ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_multi, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Ptr{Cint}, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_ext_inplace(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_ext_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_multi_inplace(ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_multi_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Ptr{Cint}, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_custom(ctx, a, b, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_custom, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_custom_inplace(ctx, a, b, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_custom_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, dims)
    ccall((:ggml_rope_yarn_corr_dims, libwhisper), Cvoid, (Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}), n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, dims)
end

function ggml_rope_ext_back(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_ext_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_multi_back(ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_multi_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Ptr{Cint}, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_clamp(ctx, a, min, max)
    ccall((:ggml_clamp, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, min, max)
end

function ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D, dst_type)
    ccall((:ggml_im2col, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Bool, ggml_type), ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D, dst_type)
end

function ggml_im2col_back(ctx, a, b, ne, s0, s1, p0, p1, d0, d1, is_2D)
    ccall((:ggml_im2col_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{Int64}, Cint, Cint, Cint, Cint, Cint, Cint, Bool), ctx, a, b, ne, s0, s1, p0, p1, d0, d1, is_2D)
end

function ggml_col2im_1d(ctx, a, s0, oc, p0)
    ccall((:ggml_col2im_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, s0, oc, p0)
end

function ggml_conv_1d(ctx, a, b, s0, p0, d0)
    ccall((:ggml_conv_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, s0, p0, d0)
end

function ggml_conv_1d_ph(ctx, a, b, s, d)
    ccall((:ggml_conv_1d_ph, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, s, d)
end

function ggml_conv_1d_dw(ctx, a, b, s0, p0, d0)
    ccall((:ggml_conv_1d_dw, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, s0, p0, d0)
end

function ggml_conv_1d_dw_ph(ctx, a, b, s0, d0)
    ccall((:ggml_conv_1d_dw_ph, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, s0, d0)
end

function ggml_conv_transpose_1d(ctx, a, b, s0, p0, d0)
    ccall((:ggml_conv_transpose_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, s0, p0, d0)
end

function ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1)
    ccall((:ggml_conv_2d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, p0, p1, d0, d1)
end

function ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, dst_type)
    ccall((:ggml_im2col_3d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Int64, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, ggml_type), ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, dst_type)
end

function ggml_conv_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2)
    ccall((:ggml_conv_3d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Int64, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2)
end

function ggml_conv_2d_sk_p0(ctx, a, b)
    ccall((:ggml_conv_2d_sk_p0, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_conv_2d_s1_ph(ctx, a, b)
    ccall((:ggml_conv_2d_s1_ph, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_conv_2d_dw(ctx, a, b, s0, s1, p0, p1, d0, d1)
    ccall((:ggml_conv_2d_dw, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, p0, p1, d0, d1)
end

function ggml_conv_2d_dw_direct(ctx, a, b, stride0, stride1, pad0, pad1, dilation0, dilation1)
    ccall((:ggml_conv_2d_dw_direct, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, stride0, stride1, pad0, pad1, dilation0, dilation1)
end

function ggml_conv_transpose_2d_p0(ctx, a, b, stride)
    ccall((:ggml_conv_transpose_2d_p0, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint), ctx, a, b, stride)
end

function ggml_conv_2d_direct(ctx, a, b, s0, s1, p0, p1, d0, d1)
    ccall((:ggml_conv_2d_direct, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, p0, p1, d0, d1)
end

function ggml_conv_3d_direct(ctx, a, b, s0, s1, s2, p0, p1, p2, d0, d1, d2, n_channels, n_batch, n_channels_out)
    ccall((:ggml_conv_3d_direct, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, s2, p0, p1, p2, d0, d1, d2, n_channels, n_batch, n_channels_out)
end

@cenum ggml_op_pool::UInt32 begin
    GGML_OP_POOL_MAX = 0
    GGML_OP_POOL_AVG = 1
    GGML_OP_POOL_COUNT = 2
end

function ggml_pool_1d(ctx, a, op, k0, s0, p0)
    ccall((:ggml_pool_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint), ctx, a, op, k0, s0, p0)
end

function ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)
    ccall((:ggml_pool_2d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint, Cint, Cfloat, Cfloat), ctx, a, op, k0, k1, s0, s1, p0, p1)
end

function ggml_pool_2d_back(ctx, a, af, op, k0, k1, s0, s1, p0, p1)
    ccall((:ggml_pool_2d_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint, Cint, Cfloat, Cfloat), ctx, a, af, op, k0, k1, s0, s1, p0, p1)
end

@cenum ggml_scale_mode::UInt32 begin
    GGML_SCALE_MODE_NEAREST = 0
    GGML_SCALE_MODE_BILINEAR = 1
    GGML_SCALE_MODE_BICUBIC = 2
    GGML_SCALE_MODE_COUNT = 3
end

@cenum ggml_scale_flag::UInt32 begin
    GGML_SCALE_FLAG_ALIGN_CORNERS = 256
    GGML_SCALE_FLAG_ANTIALIAS = 512
end

function ggml_upscale(ctx, a, scale_factor, mode)
    ccall((:ggml_upscale, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, ggml_scale_mode), ctx, a, scale_factor, mode)
end

function ggml_upscale_ext(ctx, a, ne0, ne1, ne2, ne3, mode)
    ccall((:ggml_upscale_ext, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, ggml_scale_mode), ctx, a, ne0, ne1, ne2, ne3, mode)
end

function ggml_interpolate(ctx, a, ne0, ne1, ne2, ne3, mode)
    ccall((:ggml_interpolate, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64, UInt32), ctx, a, ne0, ne1, ne2, ne3, mode)
end

function ggml_pad(ctx, a, p0, p1, p2, p3)
    ccall((:ggml_pad, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, p0, p1, p2, p3)
end

function ggml_pad_circular(ctx, a, p0, p1, p2, p3)
    ccall((:ggml_pad_circular, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, p0, p1, p2, p3)
end

function ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    ccall((:ggml_pad_ext, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
end

function ggml_pad_ext_circular(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    ccall((:ggml_pad_ext_circular, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
end

function ggml_pad_reflect_1d(ctx, a, p0, p1)
    ccall((:ggml_pad_reflect_1d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, p0, p1)
end

function ggml_roll(ctx, a, shift0, shift1, shift2, shift3)
    ccall((:ggml_roll, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, shift0, shift1, shift2, shift3)
end

function ggml_tri(ctx, a, type)
    ccall((:ggml_tri, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_tri_type), ctx, a, type)
end

function ggml_fill(ctx, a, c)
    ccall((:ggml_fill, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, c)
end

function ggml_fill_inplace(ctx, a, c)
    ccall((:ggml_fill_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, c)
end

function ggml_timestep_embedding(ctx, timesteps, dim, max_period)
    ccall((:ggml_timestep_embedding, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint), ctx, timesteps, dim, max_period)
end

@cenum ggml_sort_order::UInt32 begin
    GGML_SORT_ORDER_ASC = 0
    GGML_SORT_ORDER_DESC = 1
end

function ggml_argsort(ctx, a, order)
    ccall((:ggml_argsort, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_sort_order), ctx, a, order)
end

function ggml_argsort_top_k(ctx, a, k)
    ccall((:ggml_argsort_top_k, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, k)
end

function ggml_top_k(ctx, a, k)
    ccall((:ggml_top_k, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, k)
end

function ggml_arange(ctx, start, stop, step)
    ccall((:ggml_arange, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Cfloat, Cfloat, Cfloat), ctx, start, stop, step)
end

function ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)
    ccall((:ggml_flash_attn_ext, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat, Cfloat), ctx, q, k, v, mask, scale, max_bias, logit_softcap)
end

function ggml_flash_attn_ext_set_prec(a, prec)
    ccall((:ggml_flash_attn_ext_set_prec, libwhisper), Cvoid, (Ptr{ggml_tensor}, ggml_prec), a, prec)
end

function ggml_flash_attn_ext_get_prec(a)
    ccall((:ggml_flash_attn_ext_get_prec, libwhisper), ggml_prec, (Ptr{ggml_tensor},), a)
end

function ggml_flash_attn_ext_add_sinks(a, sinks)
    ccall((:ggml_flash_attn_ext_add_sinks, libwhisper), Cvoid, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), a, sinks)
end

function ggml_flash_attn_back(ctx, q, k, v, d, masked)
    ccall((:ggml_flash_attn_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Bool), ctx, q, k, v, d, masked)
end

function ggml_ssm_conv(ctx, sx, c)
    ccall((:ggml_ssm_conv, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, sx, c)
end

function ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, K)
    ccall((:ggml_ssm_scan, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Int64), ctx, s, x, dt, A, B, C, ids, K)
end

function ggml_win_part(ctx, a, w)
    ccall((:ggml_win_part, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, w)
end

function ggml_win_unpart(ctx, a, w0, h0, w)
    ccall((:ggml_win_unpart, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, w0, h0, w)
end

function ggml_unary(ctx, a, op)
    ccall((:ggml_unary, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op), ctx, a, op)
end

function ggml_unary_inplace(ctx, a, op)
    ccall((:ggml_unary_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op), ctx, a, op)
end

function ggml_get_rel_pos(ctx, a, qh, kh)
    ccall((:ggml_get_rel_pos, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, qh, kh)
end

function ggml_add_rel_pos(ctx, a, pw, ph)
    ccall((:ggml_add_rel_pos, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, pw, ph)
end

function ggml_add_rel_pos_inplace(ctx, a, pw, ph)
    ccall((:ggml_add_rel_pos_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, pw, ph)
end

function ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state)
    ccall((:ggml_rwkv_wkv6, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, k, v, r, tf, td, state)
end

function ggml_gated_linear_attn(ctx, k, v, q, g, state, scale)
    ccall((:ggml_gated_linear_attn, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat), ctx, k, v, q, g, state, scale)
end

function ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, state)
    ccall((:ggml_rwkv_wkv7, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, r, w, k, v, a, b, state)
end

function ggml_solve_tri(ctx, a, b, left, lower, uni)
    ccall((:ggml_solve_tri, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Bool, Bool, Bool), ctx, a, b, left, lower, uni)
end

function ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K)
    ccall((:ggml_gated_delta_net, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Int64), ctx, q, k, v, g, beta, state, K)
end

function ggml_lightning_indexer(ctx, q, k, weights, mask)
    ccall((:ggml_lightning_indexer, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, q, k, weights, mask)
end

function ggml_dsv4_hc_comb(ctx, mixes, scale, base, eps, n_iter)
    ccall((:ggml_dsv4_hc_comb, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Int32), ctx, mixes, scale, base, eps, n_iter)
end

function ggml_dsv4_hc_pre(ctx, x, weights)
    ccall((:ggml_dsv4_hc_pre, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, x, weights)
end

function ggml_dsv4_hc_post(ctx, x, residual, post, comb)
    ccall((:ggml_dsv4_hc_post, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, x, residual, post, comb)
end

# typedef void ( * ggml_custom1_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , int ith , int nth , void * userdata )
const ggml_custom1_op_t = Ptr{Cvoid}

# typedef void ( * ggml_custom2_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , const struct ggml_tensor * b , int ith , int nth , void * userdata )
const ggml_custom2_op_t = Ptr{Cvoid}

# typedef void ( * ggml_custom3_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , const struct ggml_tensor * b , const struct ggml_tensor * c , int ith , int nth , void * userdata )
const ggml_custom3_op_t = Ptr{Cvoid}

function ggml_map_custom1(ctx, a, fun, n_tasks, userdata)
    ccall((:ggml_map_custom1, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_t, Cint, Ptr{Cvoid}), ctx, a, fun, n_tasks, userdata)
end

function ggml_map_custom1_inplace(ctx, a, fun, n_tasks, userdata)
    ccall((:ggml_map_custom1_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_t, Cint, Ptr{Cvoid}), ctx, a, fun, n_tasks, userdata)
end

function ggml_map_custom2(ctx, a, b, fun, n_tasks, userdata)
    ccall((:ggml_map_custom2, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_t, Cint, Ptr{Cvoid}), ctx, a, b, fun, n_tasks, userdata)
end

function ggml_map_custom2_inplace(ctx, a, b, fun, n_tasks, userdata)
    ccall((:ggml_map_custom2_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_t, Cint, Ptr{Cvoid}), ctx, a, b, fun, n_tasks, userdata)
end

function ggml_map_custom3(ctx, a, b, c, fun, n_tasks, userdata)
    ccall((:ggml_map_custom3, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_t, Cint, Ptr{Cvoid}), ctx, a, b, c, fun, n_tasks, userdata)
end

function ggml_map_custom3_inplace(ctx, a, b, c, fun, n_tasks, userdata)
    ccall((:ggml_map_custom3_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_t, Cint, Ptr{Cvoid}), ctx, a, b, c, fun, n_tasks, userdata)
end

# typedef void ( * ggml_custom_op_t ) ( struct ggml_tensor * dst , int ith , int nth , void * userdata )
const ggml_custom_op_t = Ptr{Cvoid}

function ggml_custom_4d(ctx, type, ne0, ne1, ne2, ne3, args, n_args, fun, n_tasks, userdata)
    ccall((:ggml_custom_4d, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64, Int64, Int64, Ptr{Ptr{ggml_tensor}}, Cint, ggml_custom_op_t, Cint, Ptr{Cvoid}), ctx, type, ne0, ne1, ne2, ne3, args, n_args, fun, n_tasks, userdata)
end

function ggml_custom_inplace(ctx, a, args, n_args, fun, n_tasks, userdata)
    ccall((:ggml_custom_inplace, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{Ptr{ggml_tensor}}, Cint, ggml_custom_op_t, Cint, Ptr{Cvoid}), ctx, a, args, n_args, fun, n_tasks, userdata)
end

function ggml_cross_entropy_loss(ctx, a, b)
    ccall((:ggml_cross_entropy_loss, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_cross_entropy_loss_back(ctx, a, b, c)
    ccall((:ggml_cross_entropy_loss_back, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, c)
end

function ggml_opt_step_adamw(ctx, a, grad, m, v, adamw_params)
    ccall((:ggml_opt_step_adamw, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, grad, m, v, adamw_params)
end

function ggml_opt_step_sgd(ctx, a, grad, sgd_params)
    ccall((:ggml_opt_step_sgd, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, grad, sgd_params)
end

mutable struct ggml_cgraph end

function ggml_build_forward_select(cgraph, tensors, n_tensors, idx)
    ccall((:ggml_build_forward_select, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{Ptr{ggml_tensor}}, Cint, Cint), cgraph, tensors, n_tensors, idx)
end

function ggml_build_forward_expand(cgraph, tensor)
    ccall((:ggml_build_forward_expand, libwhisper), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, tensor)
end

function ggml_build_forward_order(cgraph, tensor)
    ccall((:ggml_build_forward_order, libwhisper), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, tensor)
end

function ggml_build_backward_expand(ctx, cgraph, grad_accs)
    ccall((:ggml_build_backward_expand, libwhisper), Cvoid, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Ptr{Ptr{ggml_tensor}}), ctx, cgraph, grad_accs)
end

function ggml_new_graph(ctx)
    ccall((:ggml_new_graph, libwhisper), Ptr{ggml_cgraph}, (Ptr{ggml_context},), ctx)
end

function ggml_new_graph_custom(ctx, size, grads)
    ccall((:ggml_new_graph_custom, libwhisper), Ptr{ggml_cgraph}, (Ptr{ggml_context}, Csize_t, Bool), ctx, size, grads)
end

function ggml_graph_dup(ctx, cgraph, force_grads)
    ccall((:ggml_graph_dup, libwhisper), Ptr{ggml_cgraph}, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Bool), ctx, cgraph, force_grads)
end

function ggml_graph_cpy(src, dst)
    ccall((:ggml_graph_cpy, libwhisper), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_cgraph}), src, dst)
end

function ggml_graph_reset(cgraph)
    ccall((:ggml_graph_reset, libwhisper), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_clear(cgraph)
    ccall((:ggml_graph_clear, libwhisper), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_size(cgraph)
    ccall((:ggml_graph_size, libwhisper), Cint, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_node(cgraph, i)
    ccall((:ggml_graph_node, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Cint), cgraph, i)
end

function ggml_graph_nodes(cgraph)
    ccall((:ggml_graph_nodes, libwhisper), Ptr{Ptr{ggml_tensor}}, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_n_nodes(cgraph)
    ccall((:ggml_graph_n_nodes, libwhisper), Cint, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_add_node(cgraph, tensor)
    ccall((:ggml_graph_add_node, libwhisper), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, tensor)
end

function ggml_graph_overhead()
    ccall((:ggml_graph_overhead, libwhisper), Csize_t, ())
end

function ggml_graph_overhead_custom(size, grads)
    ccall((:ggml_graph_overhead_custom, libwhisper), Csize_t, (Csize_t, Bool), size, grads)
end

function ggml_graph_get_tensor(cgraph, name)
    ccall((:ggml_graph_get_tensor, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{Cchar}), cgraph, name)
end

function ggml_graph_get_grad(cgraph, node)
    ccall((:ggml_graph_get_grad, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, node)
end

function ggml_graph_get_grad_acc(cgraph, node)
    ccall((:ggml_graph_get_grad_acc, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, node)
end

function ggml_graph_print(cgraph)
    ccall((:ggml_graph_print, libwhisper), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_dump_dot(gb, cgraph, filename)
    ccall((:ggml_graph_dump_dot, libwhisper), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_cgraph}, Ptr{Cchar}), gb, cgraph, filename)
end

function ggml_log_get(log_callback, user_data)
    ccall((:ggml_log_get, libwhisper), Cvoid, (Ptr{ggml_log_callback}, Ptr{Ptr{Cvoid}}), log_callback, user_data)
end

function ggml_log_set(log_callback, user_data)
    ccall((:ggml_log_set, libwhisper), Cvoid, (ggml_log_callback, Ptr{Cvoid}), log_callback, user_data)
end

function ggml_set_zero(tensor)
    ccall((:ggml_set_zero, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_tensor},), tensor)
end

function ggml_quantize_init(type)
    ccall((:ggml_quantize_init, libwhisper), Cvoid, (ggml_type,), type)
end

function ggml_quantize_free()
    ccall((:ggml_quantize_free, libwhisper), Cvoid, ())
end

function ggml_quantize_requires_imatrix(type)
    ccall((:ggml_quantize_requires_imatrix, libwhisper), Bool, (ggml_type,), type)
end

function ggml_quantize_chunk(type, src, dst, start, nrows, n_per_row, imatrix)
    ccall((:ggml_quantize_chunk, libwhisper), Csize_t, (ggml_type, Ptr{Cfloat}, Ptr{Cvoid}, Int64, Int64, Int64, Ptr{Cfloat}), type, src, dst, start, nrows, n_per_row, imatrix)
end

# typedef void ( * ggml_to_float_t ) ( const void * GGML_RESTRICT x , float * GGML_RESTRICT y , int64_t k )
const ggml_to_float_t = Ptr{Cvoid}

# typedef void ( * ggml_from_float_t ) ( const float * GGML_RESTRICT x , void * GGML_RESTRICT y , int64_t k )
const ggml_from_float_t = Ptr{Cvoid}

struct ggml_type_traits
    type_name::Ptr{Cchar}
    blck_size::Int64
    blck_size_interleave::Int64
    type_size::Csize_t
    is_quantized::Bool
    to_float::ggml_to_float_t
    from_float_ref::ggml_from_float_t
end

function ggml_get_type_traits(type)
    ccall((:ggml_get_type_traits, libwhisper), Ptr{ggml_type_traits}, (ggml_type,), type)
end

@cenum ggml_sched_priority::Int32 begin
    GGML_SCHED_PRIO_LOW = -1
    GGML_SCHED_PRIO_NORMAL = 0
    GGML_SCHED_PRIO_MEDIUM = 1
    GGML_SCHED_PRIO_HIGH = 2
    GGML_SCHED_PRIO_REALTIME = 3
end

struct ggml_threadpool_params
    cpumask::NTuple{512, Bool}
    n_threads::Cint
    prio::ggml_sched_priority
    poll::UInt32
    strict_cpu::Bool
    paused::Bool
end

mutable struct ggml_threadpool end

const ggml_threadpool_t = Ptr{ggml_threadpool}

function ggml_threadpool_params_default(n_threads)
    ccall((:ggml_threadpool_params_default, libwhisper), ggml_threadpool_params, (Cint,), n_threads)
end

function ggml_threadpool_params_init(p, n_threads)
    ccall((:ggml_threadpool_params_init, libwhisper), Cvoid, (Ptr{ggml_threadpool_params}, Cint), p, n_threads)
end

function ggml_threadpool_params_match(p0, p1)
    ccall((:ggml_threadpool_params_match, libwhisper), Bool, (Ptr{ggml_threadpool_params}, Ptr{ggml_threadpool_params}), p0, p1)
end

mutable struct ggml_backend_buffer_type end

const ggml_backend_buffer_type_t = Ptr{ggml_backend_buffer_type}

const ggml_backend_buffer_t = Ptr{ggml_backend_buffer}

mutable struct ggml_backend end

const ggml_backend_t = Ptr{ggml_backend}

struct ggml_tallocr
    buffer::ggml_backend_buffer_t
    base::Ptr{Cvoid}
    alignment::Csize_t
    offset::Csize_t
end

function ggml_tallocr_new(buffer)
    ccall((:ggml_tallocr_new, libwhisper), ggml_tallocr, (ggml_backend_buffer_t,), buffer)
end

function ggml_tallocr_alloc(talloc, tensor)
    ccall((:ggml_tallocr_alloc, libwhisper), ggml_status, (Ptr{ggml_tallocr}, Ptr{ggml_tensor}), talloc, tensor)
end

mutable struct ggml_gallocr end

const ggml_gallocr_t = Ptr{ggml_gallocr}

function ggml_gallocr_new(buft)
    ccall((:ggml_gallocr_new, libwhisper), ggml_gallocr_t, (ggml_backend_buffer_type_t,), buft)
end

function ggml_gallocr_new_n(bufts, n_bufs)
    ccall((:ggml_gallocr_new_n, libwhisper), ggml_gallocr_t, (Ptr{ggml_backend_buffer_type_t}, Cint), bufts, n_bufs)
end

function ggml_gallocr_free(galloc)
    ccall((:ggml_gallocr_free, libwhisper), Cvoid, (ggml_gallocr_t,), galloc)
end

function ggml_gallocr_reserve(galloc, graph)
    ccall((:ggml_gallocr_reserve, libwhisper), Bool, (ggml_gallocr_t, Ptr{ggml_cgraph}), galloc, graph)
end

function ggml_gallocr_reserve_n_size(galloc, graph, node_buffer_ids, leaf_buffer_ids, sizes)
    ccall((:ggml_gallocr_reserve_n_size, libwhisper), Cvoid, (ggml_gallocr_t, Ptr{ggml_cgraph}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}), galloc, graph, node_buffer_ids, leaf_buffer_ids, sizes)
end

function ggml_gallocr_reserve_n(galloc, graph, node_buffer_ids, leaf_buffer_ids)
    ccall((:ggml_gallocr_reserve_n, libwhisper), Bool, (ggml_gallocr_t, Ptr{ggml_cgraph}, Ptr{Cint}, Ptr{Cint}), galloc, graph, node_buffer_ids, leaf_buffer_ids)
end

function ggml_gallocr_alloc_graph(galloc, graph)
    ccall((:ggml_gallocr_alloc_graph, libwhisper), Bool, (ggml_gallocr_t, Ptr{ggml_cgraph}), galloc, graph)
end

function ggml_gallocr_get_buffer_size(galloc, buffer_id)
    ccall((:ggml_gallocr_get_buffer_size, libwhisper), Csize_t, (ggml_gallocr_t, Cint), galloc, buffer_id)
end

function ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft)
    ccall((:ggml_backend_alloc_ctx_tensors_from_buft_size, libwhisper), Csize_t, (Ptr{ggml_context}, ggml_backend_buffer_type_t), ctx, buft)
end

function ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)
    ccall((:ggml_backend_alloc_ctx_tensors_from_buft, libwhisper), Ptr{ggml_backend_buffer}, (Ptr{ggml_context}, ggml_backend_buffer_type_t), ctx, buft)
end

function ggml_backend_alloc_ctx_tensors(ctx, backend)
    ccall((:ggml_backend_alloc_ctx_tensors, libwhisper), Ptr{ggml_backend_buffer}, (Ptr{ggml_context}, ggml_backend_t), ctx, backend)
end

mutable struct ggml_backend_event end

const ggml_backend_event_t = Ptr{ggml_backend_event}

const ggml_backend_graph_plan_t = Ptr{Cvoid}

mutable struct ggml_backend_reg end

const ggml_backend_reg_t = Ptr{ggml_backend_reg}

mutable struct ggml_backend_device end

const ggml_backend_dev_t = Ptr{ggml_backend_device}

function ggml_backend_buft_name(buft)
    ccall((:ggml_backend_buft_name, libwhisper), Ptr{Cchar}, (ggml_backend_buffer_type_t,), buft)
end

function ggml_backend_buft_alloc_buffer(buft, size)
    ccall((:ggml_backend_buft_alloc_buffer, libwhisper), ggml_backend_buffer_t, (ggml_backend_buffer_type_t, Csize_t), buft, size)
end

function ggml_backend_buft_get_alignment(buft)
    ccall((:ggml_backend_buft_get_alignment, libwhisper), Csize_t, (ggml_backend_buffer_type_t,), buft)
end

function ggml_backend_buft_get_max_size(buft)
    ccall((:ggml_backend_buft_get_max_size, libwhisper), Csize_t, (ggml_backend_buffer_type_t,), buft)
end

function ggml_backend_buft_get_alloc_size(buft, tensor)
    ccall((:ggml_backend_buft_get_alloc_size, libwhisper), Csize_t, (ggml_backend_buffer_type_t, Ptr{ggml_tensor}), buft, tensor)
end

function ggml_backend_buft_is_host(buft)
    ccall((:ggml_backend_buft_is_host, libwhisper), Bool, (ggml_backend_buffer_type_t,), buft)
end

function ggml_backend_buft_get_device(buft)
    ccall((:ggml_backend_buft_get_device, libwhisper), ggml_backend_dev_t, (ggml_backend_buffer_type_t,), buft)
end

@cenum ggml_backend_buffer_usage::UInt32 begin
    GGML_BACKEND_BUFFER_USAGE_ANY = 0
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1
    GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2
end

function ggml_backend_buffer_name(buffer)
    ccall((:ggml_backend_buffer_name, libwhisper), Ptr{Cchar}, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_free(buffer)
    ccall((:ggml_backend_buffer_free, libwhisper), Cvoid, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_base(buffer)
    ccall((:ggml_backend_buffer_get_base, libwhisper), Ptr{Cvoid}, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_size(buffer)
    ccall((:ggml_backend_buffer_get_size, libwhisper), Csize_t, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_init_tensor(buffer, tensor)
    ccall((:ggml_backend_buffer_init_tensor, libwhisper), ggml_status, (ggml_backend_buffer_t, Ptr{ggml_tensor}), buffer, tensor)
end

function ggml_backend_buffer_get_alignment(buffer)
    ccall((:ggml_backend_buffer_get_alignment, libwhisper), Csize_t, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_max_size(buffer)
    ccall((:ggml_backend_buffer_get_max_size, libwhisper), Csize_t, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_alloc_size(buffer, tensor)
    ccall((:ggml_backend_buffer_get_alloc_size, libwhisper), Csize_t, (ggml_backend_buffer_t, Ptr{ggml_tensor}), buffer, tensor)
end

function ggml_backend_buffer_clear(buffer, value)
    ccall((:ggml_backend_buffer_clear, libwhisper), Cvoid, (ggml_backend_buffer_t, UInt8), buffer, value)
end

function ggml_backend_buffer_is_host(buffer)
    ccall((:ggml_backend_buffer_is_host, libwhisper), Bool, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_set_usage(buffer, usage)
    ccall((:ggml_backend_buffer_set_usage, libwhisper), Cvoid, (ggml_backend_buffer_t, ggml_backend_buffer_usage), buffer, usage)
end

function ggml_backend_buffer_get_usage(buffer)
    ccall((:ggml_backend_buffer_get_usage, libwhisper), ggml_backend_buffer_usage, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_type(buffer)
    ccall((:ggml_backend_buffer_get_type, libwhisper), ggml_backend_buffer_type_t, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_reset(buffer)
    ccall((:ggml_backend_buffer_reset, libwhisper), Cvoid, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_tensor_copy(src, dst)
    ccall((:ggml_backend_tensor_copy, libwhisper), Cvoid, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), src, dst)
end

function ggml_backend_guid(backend)
    ccall((:ggml_backend_guid, libwhisper), ggml_guid_t, (ggml_backend_t,), backend)
end

function ggml_backend_name(backend)
    ccall((:ggml_backend_name, libwhisper), Ptr{Cchar}, (ggml_backend_t,), backend)
end

function ggml_backend_free(backend)
    ccall((:ggml_backend_free, libwhisper), Cvoid, (ggml_backend_t,), backend)
end

function ggml_backend_get_default_buffer_type(backend)
    ccall((:ggml_backend_get_default_buffer_type, libwhisper), ggml_backend_buffer_type_t, (ggml_backend_t,), backend)
end

function ggml_backend_alloc_buffer(backend, size)
    ccall((:ggml_backend_alloc_buffer, libwhisper), ggml_backend_buffer_t, (ggml_backend_t, Csize_t), backend, size)
end

function ggml_backend_get_alignment(backend)
    ccall((:ggml_backend_get_alignment, libwhisper), Csize_t, (ggml_backend_t,), backend)
end

function ggml_backend_get_max_size(backend)
    ccall((:ggml_backend_get_max_size, libwhisper), Csize_t, (ggml_backend_t,), backend)
end

function ggml_backend_tensor_set_async(backend, tensor, data, offset, size)
    ccall((:ggml_backend_tensor_set_async, libwhisper), Cvoid, (ggml_backend_t, Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t), backend, tensor, data, offset, size)
end

function ggml_backend_tensor_get_async(backend, tensor, data, offset, size)
    ccall((:ggml_backend_tensor_get_async, libwhisper), Cvoid, (ggml_backend_t, Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t), backend, tensor, data, offset, size)
end

function ggml_backend_tensor_set_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data)
    ccall((:ggml_backend_tensor_set_2d_async, libwhisper), Cvoid, (ggml_backend_t, Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data)
end

function ggml_backend_tensor_get_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data)
    ccall((:ggml_backend_tensor_get_2d_async, libwhisper), Cvoid, (ggml_backend_t, Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data)
end

function ggml_backend_tensor_set(tensor, data, offset, size)
    ccall((:ggml_backend_tensor_set, libwhisper), Cvoid, (Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t), tensor, data, offset, size)
end

function ggml_backend_tensor_get(tensor, data, offset, size)
    ccall((:ggml_backend_tensor_get, libwhisper), Cvoid, (Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t), tensor, data, offset, size)
end

function ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data)
    ccall((:ggml_backend_tensor_set_2d, libwhisper), Cvoid, (Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), tensor, data, offset, size, n_copies, stride_tensor, stride_data)
end

function ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data)
    ccall((:ggml_backend_tensor_get_2d, libwhisper), Cvoid, (Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), tensor, data, offset, size, n_copies, stride_tensor, stride_data)
end

function ggml_backend_tensor_memset(tensor, value, offset, size)
    ccall((:ggml_backend_tensor_memset, libwhisper), Cvoid, (Ptr{ggml_tensor}, UInt8, Csize_t, Csize_t), tensor, value, offset, size)
end

function ggml_backend_synchronize(backend)
    ccall((:ggml_backend_synchronize, libwhisper), Cvoid, (ggml_backend_t,), backend)
end

function ggml_backend_graph_plan_create(backend, cgraph)
    ccall((:ggml_backend_graph_plan_create, libwhisper), ggml_backend_graph_plan_t, (ggml_backend_t, Ptr{ggml_cgraph}), backend, cgraph)
end

function ggml_backend_graph_plan_free(backend, plan)
    ccall((:ggml_backend_graph_plan_free, libwhisper), Cvoid, (ggml_backend_t, ggml_backend_graph_plan_t), backend, plan)
end

function ggml_backend_graph_plan_compute(backend, plan)
    ccall((:ggml_backend_graph_plan_compute, libwhisper), ggml_status, (ggml_backend_t, ggml_backend_graph_plan_t), backend, plan)
end

function ggml_backend_graph_compute(backend, cgraph)
    ccall((:ggml_backend_graph_compute, libwhisper), ggml_status, (ggml_backend_t, Ptr{ggml_cgraph}), backend, cgraph)
end

function ggml_backend_graph_compute_async(backend, cgraph)
    ccall((:ggml_backend_graph_compute_async, libwhisper), ggml_status, (ggml_backend_t, Ptr{ggml_cgraph}), backend, cgraph)
end

function ggml_backend_supports_op(backend, op)
    ccall((:ggml_backend_supports_op, libwhisper), Bool, (ggml_backend_t, Ptr{ggml_tensor}), backend, op)
end

function ggml_backend_supports_buft(backend, buft)
    ccall((:ggml_backend_supports_buft, libwhisper), Bool, (ggml_backend_t, ggml_backend_buffer_type_t), backend, buft)
end

function ggml_backend_offload_op(backend, op)
    ccall((:ggml_backend_offload_op, libwhisper), Bool, (ggml_backend_t, Ptr{ggml_tensor}), backend, op)
end

function ggml_backend_tensor_copy_async(backend_src, backend_dst, src, dst)
    ccall((:ggml_backend_tensor_copy_async, libwhisper), Cvoid, (ggml_backend_t, ggml_backend_t, Ptr{ggml_tensor}, Ptr{ggml_tensor}), backend_src, backend_dst, src, dst)
end

function ggml_backend_get_device(backend)
    ccall((:ggml_backend_get_device, libwhisper), ggml_backend_dev_t, (ggml_backend_t,), backend)
end

function ggml_backend_event_new(device)
    ccall((:ggml_backend_event_new, libwhisper), ggml_backend_event_t, (ggml_backend_dev_t,), device)
end

function ggml_backend_event_free(event)
    ccall((:ggml_backend_event_free, libwhisper), Cvoid, (ggml_backend_event_t,), event)
end

function ggml_backend_event_record(event, backend)
    ccall((:ggml_backend_event_record, libwhisper), Cvoid, (ggml_backend_event_t, ggml_backend_t), event, backend)
end

function ggml_backend_event_synchronize(event)
    ccall((:ggml_backend_event_synchronize, libwhisper), Cvoid, (ggml_backend_event_t,), event)
end

function ggml_backend_event_wait(backend, event)
    ccall((:ggml_backend_event_wait, libwhisper), Cvoid, (ggml_backend_t, ggml_backend_event_t), backend, event)
end

@cenum ggml_backend_dev_type::UInt32 begin
    GGML_BACKEND_DEVICE_TYPE_CPU = 0
    GGML_BACKEND_DEVICE_TYPE_GPU = 1
    GGML_BACKEND_DEVICE_TYPE_IGPU = 2
    GGML_BACKEND_DEVICE_TYPE_ACCEL = 3
    GGML_BACKEND_DEVICE_TYPE_META = 4
end

struct ggml_backend_dev_caps
    async::Bool
    host_buffer::Bool
    buffer_from_host_ptr::Bool
    events::Bool
    mmap_support::Bool
end

struct ggml_backend_dev_props
    name::Ptr{Cchar}
    description::Ptr{Cchar}
    memory_free::Csize_t
    memory_total::Csize_t
    type::ggml_backend_dev_type
    device_id::Ptr{Cchar}
    caps::ggml_backend_dev_caps
end

function ggml_backend_dev_name(device)
    ccall((:ggml_backend_dev_name, libwhisper), Ptr{Cchar}, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_description(device)
    ccall((:ggml_backend_dev_description, libwhisper), Ptr{Cchar}, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_memory(device, free, total)
    ccall((:ggml_backend_dev_memory, libwhisper), Cvoid, (ggml_backend_dev_t, Ptr{Csize_t}, Ptr{Csize_t}), device, free, total)
end

function ggml_backend_dev_type(device)
    ccall((:ggml_backend_dev_type, libwhisper), ggml_backend_dev_type, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_get_props(device, props)
    ccall((:ggml_backend_dev_get_props, libwhisper), Cvoid, (ggml_backend_dev_t, Ptr{ggml_backend_dev_props}), device, props)
end

function ggml_backend_dev_backend_reg(device)
    ccall((:ggml_backend_dev_backend_reg, libwhisper), ggml_backend_reg_t, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_init(device, params)
    ccall((:ggml_backend_dev_init, libwhisper), ggml_backend_t, (ggml_backend_dev_t, Ptr{Cchar}), device, params)
end

function ggml_backend_dev_buffer_type(device)
    ccall((:ggml_backend_dev_buffer_type, libwhisper), ggml_backend_buffer_type_t, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_host_buffer_type(device)
    ccall((:ggml_backend_dev_host_buffer_type, libwhisper), ggml_backend_buffer_type_t, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_buffer_from_host_ptr(device, ptr, size, max_tensor_size)
    ccall((:ggml_backend_dev_buffer_from_host_ptr, libwhisper), ggml_backend_buffer_t, (ggml_backend_dev_t, Ptr{Cvoid}, Csize_t, Csize_t), device, ptr, size, max_tensor_size)
end

function ggml_backend_dev_supports_op(device, op)
    ccall((:ggml_backend_dev_supports_op, libwhisper), Bool, (ggml_backend_dev_t, Ptr{ggml_tensor}), device, op)
end

function ggml_backend_dev_supports_buft(device, buft)
    ccall((:ggml_backend_dev_supports_buft, libwhisper), Bool, (ggml_backend_dev_t, ggml_backend_buffer_type_t), device, buft)
end

function ggml_backend_dev_offload_op(device, op)
    ccall((:ggml_backend_dev_offload_op, libwhisper), Bool, (ggml_backend_dev_t, Ptr{ggml_tensor}), device, op)
end

function ggml_backend_reg_name(reg)
    ccall((:ggml_backend_reg_name, libwhisper), Ptr{Cchar}, (ggml_backend_reg_t,), reg)
end

function ggml_backend_reg_dev_count(reg)
    ccall((:ggml_backend_reg_dev_count, libwhisper), Csize_t, (ggml_backend_reg_t,), reg)
end

function ggml_backend_reg_dev_get(reg, index)
    ccall((:ggml_backend_reg_dev_get, libwhisper), ggml_backend_dev_t, (ggml_backend_reg_t, Csize_t), reg, index)
end

function ggml_backend_reg_get_proc_address(reg, name)
    ccall((:ggml_backend_reg_get_proc_address, libwhisper), Ptr{Cvoid}, (ggml_backend_reg_t, Ptr{Cchar}), reg, name)
end

# typedef void * ( * ggml_backend_comm_init_t ) ( ggml_backend_t * backends , size_t n_backends )
const ggml_backend_comm_init_t = Ptr{Cvoid}

# typedef void ( * ggml_backend_comm_free_t ) ( void * comm_ctx )
const ggml_backend_comm_free_t = Ptr{Cvoid}

# typedef bool ( * ggml_backend_comm_allreduce_tensor_t ) ( void * comm_ctx , struct ggml_tensor * * tensors )
const ggml_backend_comm_allreduce_tensor_t = Ptr{Cvoid}

# typedef ggml_backend_buffer_type_t ( * ggml_backend_split_buffer_type_t ) ( int main_device , const float * tensor_split )
const ggml_backend_split_buffer_type_t = Ptr{Cvoid}

# typedef void ( * ggml_backend_set_n_threads_t ) ( ggml_backend_t backend , int n_threads )
const ggml_backend_set_n_threads_t = Ptr{Cvoid}

# typedef ggml_backend_buffer_type_t * ( * ggml_backend_dev_get_extra_bufts_t ) ( ggml_backend_dev_t device )
const ggml_backend_dev_get_extra_bufts_t = Ptr{Cvoid}

# typedef void ( * ggml_backend_set_abort_callback_t ) ( ggml_backend_t backend , ggml_abort_callback abort_callback , void * abort_callback_data )
const ggml_backend_set_abort_callback_t = Ptr{Cvoid}

struct ggml_backend_feature
    name::Ptr{Cchar}
    value::Ptr{Cchar}
end

# typedef struct ggml_backend_feature * ( * ggml_backend_get_features_t ) ( ggml_backend_reg_t reg )
const ggml_backend_get_features_t = Ptr{Cvoid}

function ggml_backend_register(reg)
    ccall((:ggml_backend_register, libwhisper), Cvoid, (ggml_backend_reg_t,), reg)
end

function ggml_backend_device_register(device)
    ccall((:ggml_backend_device_register, libwhisper), Cvoid, (ggml_backend_dev_t,), device)
end

function ggml_backend_reg_count()
    ccall((:ggml_backend_reg_count, libwhisper), Csize_t, ())
end

function ggml_backend_reg_get(index)
    ccall((:ggml_backend_reg_get, libwhisper), ggml_backend_reg_t, (Csize_t,), index)
end

function ggml_backend_reg_by_name(name)
    ccall((:ggml_backend_reg_by_name, libwhisper), ggml_backend_reg_t, (Ptr{Cchar},), name)
end

function ggml_backend_dev_count()
    ccall((:ggml_backend_dev_count, libwhisper), Csize_t, ())
end

function ggml_backend_dev_get(index)
    ccall((:ggml_backend_dev_get, libwhisper), ggml_backend_dev_t, (Csize_t,), index)
end

function ggml_backend_dev_by_name(name)
    ccall((:ggml_backend_dev_by_name, libwhisper), ggml_backend_dev_t, (Ptr{Cchar},), name)
end

function ggml_backend_dev_by_type(type)
    ccall((:ggml_backend_dev_by_type, libwhisper), ggml_backend_dev_t, (ggml_backend_dev_type,), type)
end

function ggml_backend_init_by_name(name, params)
    ccall((:ggml_backend_init_by_name, libwhisper), ggml_backend_t, (Ptr{Cchar}, Ptr{Cchar}), name, params)
end

function ggml_backend_init_by_type(type, params)
    ccall((:ggml_backend_init_by_type, libwhisper), ggml_backend_t, (ggml_backend_dev_type, Ptr{Cchar}), type, params)
end

function ggml_backend_init_best()
    ccall((:ggml_backend_init_best, libwhisper), ggml_backend_t, ())
end

function ggml_backend_load(path)
    ccall((:ggml_backend_load, libwhisper), ggml_backend_reg_t, (Ptr{Cchar},), path)
end

function ggml_backend_unload(reg)
    ccall((:ggml_backend_unload, libwhisper), Cvoid, (ggml_backend_reg_t,), reg)
end

function ggml_backend_load_all()
    ccall((:ggml_backend_load_all, libwhisper), Cvoid, ())
end

function ggml_backend_load_all_from_path(dir_path)
    ccall((:ggml_backend_load_all_from_path, libwhisper), Cvoid, (Ptr{Cchar},), dir_path)
end

mutable struct ggml_backend_sched end

const ggml_backend_sched_t = Ptr{ggml_backend_sched}

# typedef bool ( * ggml_backend_sched_eval_callback ) ( struct ggml_tensor * t , bool ask , void * user_data )
const ggml_backend_sched_eval_callback = Ptr{Cvoid}

function ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel, op_offload)
    ccall((:ggml_backend_sched_new, libwhisper), ggml_backend_sched_t, (Ptr{ggml_backend_t}, Ptr{ggml_backend_buffer_type_t}, Cint, Csize_t, Bool, Bool), backends, bufts, n_backends, graph_size, parallel, op_offload)
end

function ggml_backend_sched_free(sched)
    ccall((:ggml_backend_sched_free, libwhisper), Cvoid, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_reserve_size(sched, measure_graph, sizes)
    ccall((:ggml_backend_sched_reserve_size, libwhisper), Cvoid, (ggml_backend_sched_t, Ptr{ggml_cgraph}, Ptr{Csize_t}), sched, measure_graph, sizes)
end

function ggml_backend_sched_reserve(sched, measure_graph)
    ccall((:ggml_backend_sched_reserve, libwhisper), Bool, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, measure_graph)
end

function ggml_backend_sched_get_n_backends(sched)
    ccall((:ggml_backend_sched_get_n_backends, libwhisper), Cint, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_get_backend(sched, i)
    ccall((:ggml_backend_sched_get_backend, libwhisper), ggml_backend_t, (ggml_backend_sched_t, Cint), sched, i)
end

function ggml_backend_sched_get_n_splits(sched)
    ccall((:ggml_backend_sched_get_n_splits, libwhisper), Cint, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_get_n_copies(sched)
    ccall((:ggml_backend_sched_get_n_copies, libwhisper), Cint, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_get_buffer_type(sched, backend)
    ccall((:ggml_backend_sched_get_buffer_type, libwhisper), ggml_backend_buffer_type_t, (ggml_backend_sched_t, ggml_backend_t), sched, backend)
end

function ggml_backend_sched_get_buffer_size(sched, backend)
    ccall((:ggml_backend_sched_get_buffer_size, libwhisper), Csize_t, (ggml_backend_sched_t, ggml_backend_t), sched, backend)
end

function ggml_backend_sched_set_tensor_backend(sched, node, backend)
    ccall((:ggml_backend_sched_set_tensor_backend, libwhisper), Cvoid, (ggml_backend_sched_t, Ptr{ggml_tensor}, ggml_backend_t), sched, node, backend)
end

function ggml_backend_sched_get_tensor_backend(sched, node)
    ccall((:ggml_backend_sched_get_tensor_backend, libwhisper), ggml_backend_t, (ggml_backend_sched_t, Ptr{ggml_tensor}), sched, node)
end

function ggml_backend_sched_split_graph(sched, graph)
    ccall((:ggml_backend_sched_split_graph, libwhisper), Cvoid, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, graph)
end

function ggml_backend_sched_alloc_graph(sched, graph)
    ccall((:ggml_backend_sched_alloc_graph, libwhisper), Bool, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, graph)
end

function ggml_backend_sched_graph_compute(sched, graph)
    ccall((:ggml_backend_sched_graph_compute, libwhisper), ggml_status, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, graph)
end

function ggml_backend_sched_graph_compute_async(sched, graph)
    ccall((:ggml_backend_sched_graph_compute_async, libwhisper), ggml_status, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, graph)
end

function ggml_backend_sched_synchronize(sched)
    ccall((:ggml_backend_sched_synchronize, libwhisper), Cvoid, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_reset(sched)
    ccall((:ggml_backend_sched_reset, libwhisper), Cvoid, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_set_eval_callback(sched, callback, user_data)
    ccall((:ggml_backend_sched_set_eval_callback, libwhisper), Cvoid, (ggml_backend_sched_t, ggml_backend_sched_eval_callback, Ptr{Cvoid}), sched, callback, user_data)
end

@cenum ggml_backend_meta_split_axis::UInt32 begin
    GGML_BACKEND_SPLIT_AXIS_0 = 0
    GGML_BACKEND_SPLIT_AXIS_1 = 1
    GGML_BACKEND_SPLIT_AXIS_2 = 2
    GGML_BACKEND_SPLIT_AXIS_3 = 3
    GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10
    GGML_BACKEND_SPLIT_AXIS_PARTIAL = 11
    GGML_BACKEND_SPLIT_AXIS_NONE = 98
    GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99
end

function ggml_backend_meta_split_axis_name(split_axis)
    ccall((:ggml_backend_meta_split_axis_name, libwhisper), Ptr{Cchar}, (ggml_backend_meta_split_axis,), split_axis)
end

struct ggml_backend_meta_split_state
    axis::ggml_backend_meta_split_axis
    ne::NTuple{256, Int64}
    nr::NTuple{16, UInt32}
    n_segments::UInt32
end

# typedef struct ggml_backend_meta_split_state ( * ggml_backend_meta_get_split_state_t ) ( const struct ggml_tensor * tensor , void * userdata )
const ggml_backend_meta_get_split_state_t = Ptr{Cvoid}

function ggml_backend_meta_device(devs, n_devs, get_split_state, get_split_state_ud)
    ccall((:ggml_backend_meta_device, libwhisper), ggml_backend_dev_t, (Ptr{ggml_backend_dev_t}, Csize_t, ggml_backend_meta_get_split_state_t, Ptr{Cvoid}), devs, n_devs, get_split_state, get_split_state_ud)
end

struct ggml_backend_graph_copy
    buffer::ggml_backend_buffer_t
    ctx_allocated::Ptr{ggml_context}
    ctx_unallocated::Ptr{ggml_context}
    graph::Ptr{ggml_cgraph}
end

function ggml_backend_graph_copy(backend, graph)
    ccall((:ggml_backend_graph_copy, libwhisper), ggml_backend_graph_copy, (ggml_backend_t, Ptr{ggml_cgraph}), backend, graph)
end

function ggml_backend_graph_copy_free(copy)
    ccall((:ggml_backend_graph_copy_free, libwhisper), Cvoid, (ggml_backend_graph_copy,), copy)
end

# typedef bool ( * ggml_backend_eval_callback ) ( int node_index , struct ggml_tensor * t1 , struct ggml_tensor * t2 , void * user_data )
const ggml_backend_eval_callback = Ptr{Cvoid}

function ggml_backend_compare_graph_backend(backend1, backend2, graph, callback, user_data, test_nodes, num_test_nodes)
    ccall((:ggml_backend_compare_graph_backend, libwhisper), Bool, (ggml_backend_t, ggml_backend_t, Ptr{ggml_cgraph}, ggml_backend_eval_callback, Ptr{Cvoid}, Ptr{Ptr{ggml_tensor}}, Csize_t), backend1, backend2, graph, callback, user_data, test_nodes, num_test_nodes)
end

function ggml_backend_tensor_alloc(buffer, tensor, addr)
    ccall((:ggml_backend_tensor_alloc, libwhisper), ggml_status, (ggml_backend_buffer_t, Ptr{ggml_tensor}, Ptr{Cvoid}), buffer, tensor, addr)
end

function ggml_backend_view_init(tensor)
    ccall((:ggml_backend_view_init, libwhisper), ggml_status, (Ptr{ggml_tensor},), tensor)
end

function ggml_backend_cpu_buffer_from_ptr(ptr, size)
    ccall((:ggml_backend_cpu_buffer_from_ptr, libwhisper), ggml_backend_buffer_t, (Ptr{Cvoid}, Csize_t), ptr, size)
end

function ggml_backend_cpu_buffer_type()
    ccall((:ggml_backend_cpu_buffer_type, libwhisper), ggml_backend_buffer_type_t, ())
end

struct ggml_cplan
    work_size::Csize_t
    work_data::Ptr{UInt8}
    n_threads::Cint
    threadpool::Ptr{ggml_threadpool}
    abort_callback::ggml_abort_callback
    abort_callback_data::Ptr{Cvoid}
    use_ref::Bool
end

@cenum ggml_numa_strategy::UInt32 begin
    GGML_NUMA_STRATEGY_DISABLED = 0
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1
    GGML_NUMA_STRATEGY_ISOLATE = 2
    GGML_NUMA_STRATEGY_NUMACTL = 3
    GGML_NUMA_STRATEGY_MIRROR = 4
    GGML_NUMA_STRATEGY_COUNT = 5
end

function ggml_numa_init(numa)
    ccall((:ggml_numa_init, libwhisper), Cvoid, (ggml_numa_strategy,), numa)
end

function ggml_is_numa()
    ccall((:ggml_is_numa, libwhisper), Bool, ())
end

function ggml_new_i32(ctx, value)
    ccall((:ggml_new_i32, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Int32), ctx, value)
end

function ggml_new_f32(ctx, value)
    ccall((:ggml_new_f32, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_context}, Cfloat), ctx, value)
end

function ggml_set_i32(tensor, value)
    ccall((:ggml_set_i32, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Int32), tensor, value)
end

function ggml_set_f32(tensor, value)
    ccall((:ggml_set_f32, libwhisper), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Cfloat), tensor, value)
end

function ggml_get_i32_1d(tensor, i)
    ccall((:ggml_get_i32_1d, libwhisper), Int32, (Ptr{ggml_tensor}, Cint), tensor, i)
end

function ggml_set_i32_1d(tensor, i, value)
    ccall((:ggml_set_i32_1d, libwhisper), Cvoid, (Ptr{ggml_tensor}, Cint, Int32), tensor, i, value)
end

function ggml_get_i32_nd(tensor, i0, i1, i2, i3)
    ccall((:ggml_get_i32_nd, libwhisper), Int32, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), tensor, i0, i1, i2, i3)
end

function ggml_set_i32_nd(tensor, i0, i1, i2, i3, value)
    ccall((:ggml_set_i32_nd, libwhisper), Cvoid, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Int32), tensor, i0, i1, i2, i3, value)
end

function ggml_get_f32_1d(tensor, i)
    ccall((:ggml_get_f32_1d, libwhisper), Cfloat, (Ptr{ggml_tensor}, Cint), tensor, i)
end

function ggml_set_f32_1d(tensor, i, value)
    ccall((:ggml_set_f32_1d, libwhisper), Cvoid, (Ptr{ggml_tensor}, Cint, Cfloat), tensor, i, value)
end

function ggml_get_f32_nd(tensor, i0, i1, i2, i3)
    ccall((:ggml_get_f32_nd, libwhisper), Cfloat, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), tensor, i0, i1, i2, i3)
end

function ggml_set_f32_nd(tensor, i0, i1, i2, i3, value)
    ccall((:ggml_set_f32_nd, libwhisper), Cvoid, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cfloat), tensor, i0, i1, i2, i3, value)
end

function ggml_threadpool_new(params)
    ccall((:ggml_threadpool_new, libwhisper), Ptr{ggml_threadpool}, (Ptr{ggml_threadpool_params},), params)
end

function ggml_threadpool_free(threadpool)
    ccall((:ggml_threadpool_free, libwhisper), Cvoid, (Ptr{ggml_threadpool},), threadpool)
end

function ggml_threadpool_get_n_threads(threadpool)
    ccall((:ggml_threadpool_get_n_threads, libwhisper), Cint, (Ptr{ggml_threadpool},), threadpool)
end

function ggml_threadpool_pause(threadpool)
    ccall((:ggml_threadpool_pause, libwhisper), Cvoid, (Ptr{ggml_threadpool},), threadpool)
end

function ggml_threadpool_resume(threadpool)
    ccall((:ggml_threadpool_resume, libwhisper), Cvoid, (Ptr{ggml_threadpool},), threadpool)
end

function ggml_graph_plan(cgraph, n_threads, threadpool)
    ccall((:ggml_graph_plan, libwhisper), ggml_cplan, (Ptr{ggml_cgraph}, Cint, Ptr{ggml_threadpool}), cgraph, n_threads, threadpool)
end

function ggml_graph_compute(cgraph, cplan)
    ccall((:ggml_graph_compute, libwhisper), ggml_status, (Ptr{ggml_cgraph}, Ptr{ggml_cplan}), cgraph, cplan)
end

function ggml_graph_compute_with_ctx(ctx, cgraph, n_threads)
    ccall((:ggml_graph_compute_with_ctx, libwhisper), ggml_status, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Cint), ctx, cgraph, n_threads)
end

function ggml_cpu_has_sse3()
    ccall((:ggml_cpu_has_sse3, libwhisper), Cint, ())
end

function ggml_cpu_has_ssse3()
    ccall((:ggml_cpu_has_ssse3, libwhisper), Cint, ())
end

function ggml_cpu_has_avx()
    ccall((:ggml_cpu_has_avx, libwhisper), Cint, ())
end

function ggml_cpu_has_avx_vnni()
    ccall((:ggml_cpu_has_avx_vnni, libwhisper), Cint, ())
end

function ggml_cpu_has_avx2()
    ccall((:ggml_cpu_has_avx2, libwhisper), Cint, ())
end

function ggml_cpu_has_bmi2()
    ccall((:ggml_cpu_has_bmi2, libwhisper), Cint, ())
end

function ggml_cpu_has_f16c()
    ccall((:ggml_cpu_has_f16c, libwhisper), Cint, ())
end

function ggml_cpu_has_fma()
    ccall((:ggml_cpu_has_fma, libwhisper), Cint, ())
end

function ggml_cpu_has_avx512()
    ccall((:ggml_cpu_has_avx512, libwhisper), Cint, ())
end

function ggml_cpu_has_avx512_vbmi()
    ccall((:ggml_cpu_has_avx512_vbmi, libwhisper), Cint, ())
end

function ggml_cpu_has_avx512_vnni()
    ccall((:ggml_cpu_has_avx512_vnni, libwhisper), Cint, ())
end

function ggml_cpu_has_avx512_bf16()
    ccall((:ggml_cpu_has_avx512_bf16, libwhisper), Cint, ())
end

function ggml_cpu_has_amx_int8()
    ccall((:ggml_cpu_has_amx_int8, libwhisper), Cint, ())
end

function ggml_cpu_has_neon()
    ccall((:ggml_cpu_has_neon, libwhisper), Cint, ())
end

function ggml_cpu_has_arm_fma()
    ccall((:ggml_cpu_has_arm_fma, libwhisper), Cint, ())
end

function ggml_cpu_has_fp16_va()
    ccall((:ggml_cpu_has_fp16_va, libwhisper), Cint, ())
end

function ggml_cpu_has_dotprod()
    ccall((:ggml_cpu_has_dotprod, libwhisper), Cint, ())
end

function ggml_cpu_has_matmul_int8()
    ccall((:ggml_cpu_has_matmul_int8, libwhisper), Cint, ())
end

function ggml_cpu_has_sve()
    ccall((:ggml_cpu_has_sve, libwhisper), Cint, ())
end

function ggml_cpu_get_sve_cnt()
    ccall((:ggml_cpu_get_sve_cnt, libwhisper), Cint, ())
end

function ggml_cpu_has_sme()
    ccall((:ggml_cpu_has_sme, libwhisper), Cint, ())
end

function ggml_cpu_has_sme2()
    ccall((:ggml_cpu_has_sme2, libwhisper), Cint, ())
end

function ggml_cpu_has_riscv_v()
    ccall((:ggml_cpu_has_riscv_v, libwhisper), Cint, ())
end

function ggml_cpu_get_rvv_vlen()
    ccall((:ggml_cpu_get_rvv_vlen, libwhisper), Cint, ())
end

function ggml_cpu_has_vsx()
    ccall((:ggml_cpu_has_vsx, libwhisper), Cint, ())
end

function ggml_cpu_has_vxe()
    ccall((:ggml_cpu_has_vxe, libwhisper), Cint, ())
end

function ggml_cpu_has_wasm_simd()
    ccall((:ggml_cpu_has_wasm_simd, libwhisper), Cint, ())
end

function ggml_cpu_has_llamafile()
    ccall((:ggml_cpu_has_llamafile, libwhisper), Cint, ())
end

# typedef void ( * ggml_vec_dot_t ) ( int n , float * GGML_RESTRICT s , size_t bs , const void * GGML_RESTRICT x , size_t bx , const void * GGML_RESTRICT y , size_t by , int nrc )
const ggml_vec_dot_t = Ptr{Cvoid}

struct ggml_type_traits_cpu
    from_float::ggml_from_float_t
    vec_dot::ggml_vec_dot_t
    vec_dot_type::ggml_type
    nrows::Int64
end

function ggml_get_type_traits_cpu(type)
    ccall((:ggml_get_type_traits_cpu, libwhisper), Ptr{ggml_type_traits_cpu}, (ggml_type,), type)
end

function ggml_cpu_init()
    ccall((:ggml_cpu_init, libwhisper), Cvoid, ())
end

function ggml_backend_cpu_init()
    ccall((:ggml_backend_cpu_init, libwhisper), ggml_backend_t, ())
end

function ggml_backend_is_cpu(backend)
    ccall((:ggml_backend_is_cpu, libwhisper), Bool, (ggml_backend_t,), backend)
end

function ggml_backend_cpu_set_n_threads(backend_cpu, n_threads)
    ccall((:ggml_backend_cpu_set_n_threads, libwhisper), Cvoid, (ggml_backend_t, Cint), backend_cpu, n_threads)
end

function ggml_backend_cpu_set_threadpool(backend_cpu, threadpool)
    ccall((:ggml_backend_cpu_set_threadpool, libwhisper), Cvoid, (ggml_backend_t, ggml_threadpool_t), backend_cpu, threadpool)
end

function ggml_backend_cpu_set_abort_callback(backend_cpu, abort_callback, abort_callback_data)
    ccall((:ggml_backend_cpu_set_abort_callback, libwhisper), Cvoid, (ggml_backend_t, ggml_abort_callback, Ptr{Cvoid}), backend_cpu, abort_callback, abort_callback_data)
end

function ggml_backend_cpu_set_use_ref(backend_cpu, use_ref)
    ccall((:ggml_backend_cpu_set_use_ref, libwhisper), Cvoid, (ggml_backend_t, Bool), backend_cpu, use_ref)
end

function ggml_backend_cpu_reg()
    ccall((:ggml_backend_cpu_reg, libwhisper), ggml_backend_reg_t, ())
end

function ggml_cpu_fp32_to_fp32(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp32_to_fp32, libwhisper), Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_fp32_to_i32(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp32_to_i32, libwhisper), Cvoid, (Ptr{Cfloat}, Ptr{Int32}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_fp32_to_fp16(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp32_to_fp16, libwhisper), Cvoid, (Ptr{Cfloat}, Ptr{ggml_fp16_t}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_fp16_to_fp32(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp16_to_fp32, libwhisper), Cvoid, (Ptr{ggml_fp16_t}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_fp32_to_bf16(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp32_to_bf16, libwhisper), Cvoid, (Ptr{Cfloat}, Ptr{ggml_bf16_t}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_bf16_to_fp32(arg1, arg2, arg3)
    ccall((:ggml_cpu_bf16_to_fp32, libwhisper), Cvoid, (Ptr{ggml_bf16_t}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

const WHISPER_SAMPLE_RATE = 16000

const WHISPER_N_FFT = 400

const WHISPER_HOP_LENGTH = 160

const WHISPER_CHUNK_SIZE = 30

const GGML_FILE_MAGIC = 0x67676d6c

const GGML_FILE_VERSION = 2

const GGML_QNT_VERSION = 2

const GGML_QNT_VERSION_FACTOR = 1000

const GGML_MAX_DIMS = 4

const GGML_MAX_PARAMS = 2048

const GGML_MAX_SRC = 10

const GGML_MAX_N_THREADS = 512

const GGML_MAX_OP_PARAMS = 64

const GGML_MAX_NAME = 64

const GGML_DEFAULT_N_THREADS = 4

const GGML_DEFAULT_GRAPH_SIZE = 2048

const GGML_MEM_ALIGN = 16

const GGML_EXIT_SUCCESS = 0

const GGML_EXIT_ABORTED = 1

const GGML_ROPE_TYPE_NORMAL = 0

const GGML_ROPE_TYPE_NEOX = 2

const GGML_ROPE_TYPE_MROPE = 8

const GGML_ROPE_TYPE_VISION = 24

const GGML_ROPE_TYPE_IMROPE = 40

const GGML_N_TASKS_MAX = -1

# Skipping MacroDefinition: GGML_BACKEND_API extern

const GGML_BACKEND_META_MAX_DEVICES = 16

# exports
const PREFIXES = ["whisper_", "WHISPER_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
