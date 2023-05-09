module Whisper
using DataDeps


include("LibWhisper.jl")
using Whisper.LibWhisper

include("models.jl")

export transcribe

function __init__()
    ENV["DATADEPS_ALWAYS_ACCEPT"]="true"
    register_datadeps()
end

function create_ctx(model)
    whisper_init_from_file(DataDeps.resolve("whisper-ggml-$model/ggml-$model.bin", "__FILE__"))
end

struct WhisperParams
    ref::Base.RefValue{whisper_full_params}
end
Base.unsafe_convert(::Type{Ptr{whisper_full_params}}, wparams::WhisperParams) =
    Base.unsafe_convert(Ptr{whisper_full_params}, getfield(wparams, :ref))
Base.convert(::Type{whisper_full_params}, wparams::WhisperParams) =
    getfield(wparams, :ref)[]
Base.getproperty(wparams::WhisperParams, field::Symbol) =
    getproperty(Base.unsafe_convert(Ptr{whisper_full_params}, wparams), field)
Base.setproperty!(wparams::WhisperParams, field::Symbol, value) =
    setproperty!(Base.unsafe_convert(Ptr{whisper_full_params}, wparams), field, value)
function create_params(flag=LibWhisper.WHISPER_SAMPLING_GREEDY)
    raw_params = whisper_full_default_params(flag)
    return WhisperParams(Ref(raw_params))
end

"""
    transcribe(model, data) -> String

Run inference using the Whisper speech-to-text model. The model file is
automatically downloaded from HuggingFace on first use.

- `model`: Whisper model name (such as "base" or "medium.en")
- `data`: `Vector{Float32}` containing 16kHz sampled audio
"""
function transcribe(model, data)
    ctx = create_ctx(model)
    wparams = create_params()

    ret = transcribe(ctx, wparams, data)

    whisper_free(ctx)

    return ret
end

function transcribe(ctx, wparams, data)
    ret = whisper_full_parallel(ctx, wparams, data, length(data), 1)

    if ret != 0
        error("Error running whisper model: $ret")
    end

    n_segments = whisper_full_n_segments(ctx)

    result = ""
    for i in 0:(n_segments - 1)
        txt = whisper_full_get_segment_text(ctx, i)
        result = result * unsafe_string(txt)
        t0 = whisper_full_get_segment_t0(ctx, i)
        t1 = whisper_full_get_segment_t1(ctx, i)
        @debug "Time for inference: ", t0-t1
    end

    return result
end

end
