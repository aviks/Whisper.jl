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

@testset "model load" begin
    ctx = whisper_init_from_file(datadep"whisper-ggml-base.en/ggml-base.en.bin" )
    @test ctx != C_NULL

    wparams = whisper_full_default_params(LibWhisper.WHISPER_SAMPLING_GREEDY)
    @test wparams != C_NULL
end

@testset "basic infer" begin

    s = load(joinpath(@__DIR__,"en-ref-1.ogg"))  # This loads float32 data, which is what we need
    samplerate!(s, 16000)                        # Whisper expects 16kHz sample rate
    sd = s.data
    ss = [sd[i,1] + sd[i,2] for i in 1:size(sd)[1]] #convert stereo to mono

    result = transcribe("base.en", ss)

    expected = readlines(joinpath(@__DIR__, "en-ref-1.txt"))[1]
    @test expected == result
end