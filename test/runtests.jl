using Whisper
using Whisper.LibWhisper
using Test
using DataDeps

using FileIO
using LibSndFile
using SampledSignals


@testset "model downloads" begin
    @test isdir(datadep"whisper-ggml-tiny.en")
    @test isfile(datadep"whisper-ggml-tiny.en/ggml-tiny.en.bin")
end

@testset "basic infer" begin
    ctx = whisper_init(datadep"whisper-ggml-base.en/ggml-base.en.bin" )
    @test ctx != C_NULL

    wparams = whisper_full_default_params(LibWhisper.WHISPER_SAMPLING_GREEDY)

    s = load(joinpath(@__DIR__,"en-ref-1.ogg"))  # This loads float32 data, which is what we need
    samplerate!(s, 16000)                        # Whisper expects 16kHz sample rate
    sd = s.data
    ss = [sd[i,1] + sd[i,2] for i in 1:size(sd)[1]] #convert stereo to mono

    @time ret = whisper_full_parallel(ctx, wparams, ss, length(ss), 1)
    @test ret == 0 

    n_segments = whisper_full_n_segments(ctx)

    result = ""
    for i in 0:(n_segments - 1)
        txt = whisper_full_get_segment_text(ctx, i)
        result = result * unsafe_string(txt)
        t0 = whisper_full_get_segment_t0(ctx, i)
        t1 = whisper_full_get_segment_t1(ctx, i)
    end

    expected = readlines(joinpath(@__DIR__, "en-ref-1.txt"))[1]
    @test expected == result
end