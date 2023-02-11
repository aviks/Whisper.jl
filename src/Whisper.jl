module Whisper
using DataDeps


# Write your package code here.

include("LibWhisper.jl")
using Whisper.LibWhisper

include("models.jl")

export transcribe

function __init__()
    ENV["DATADEPS_ALWAYS_ACCEPT"]="true"
    register_datadeps()
end

# run inference using the whisper model
# 
# `full_infer(model, data)`
# 
#   data: array of floats containing 16kHz sampled audio
#   model: model name 
function transcribe(model, data)
    ctx = whisper_init_from_file(DataDeps.resolve("whisper-ggml-$model/ggml-$model.bin", "__FILE__") )
    wparams = whisper_full_default_params(LibWhisper.WHISPER_SAMPLING_GREEDY)
    
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
    end
    return result
end

end
