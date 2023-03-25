using Whisper
using Whisper.LibWhisper
using Test
using DataDeps

using FileIO
using LibSndFile
using SampledSignals
using DataDeps
using StringDistances


register(DataDep(
    "WhisperSamples",
    "Whisper sample test files",
    [ 
        "https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga", #gb0
        "https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg",                                     #gb1
        "https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg",                                                     #hp0
        "https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav",                                                        #mm1
        "https://upload.wikimedia.org/wikipedia/commons/c/c1/La_contaminacion_del_agua.ogg"                                         #es1
    ],
    [
        "b844a36b9b0c0d777c64f1d62356bf0b6cad6a0f753627f5a7e7abd17c843f0c",
        "97a6384767e2fc3fb27c7593831aa19115d909fcdb85a1e389359ecc4b92a1e8",
        "753014d9f365a3d49989aecb3d2416f2aa6644909fc2d6289e5b986ee1324472",
        "37de21902b32aa2fc147ccbfdcc0566cc7061fffb2c0b10874f05147c0b9de0f",
        "43ee99686d75fd2976128450cec95a621a70a99b4dbd1c224fb9b35c6549daae",
    ],
    post_fetch_method = [
        file->mv(file, "gb0.oga"), 
        file->mv(file, "gb1.ogg"), 
        file->mv(file, "hp0.ogg"), 
        file->mv(file, "mm1.wav"),
        file->mv(file, "es1.ogg")
    ]
))


@testset "model downloads" begin
    @test isdir(datadep"whisper-ggml-tiny.en")
    @test isfile(datadep"whisper-ggml-tiny.en/ggml-tiny.en.bin")
end

@testset "model load" begin
    ctx = whisper_init_from_file(datadep"whisper-ggml-tiny.en/ggml-tiny.en.bin" )
    @test ctx != C_NULL

    wparams = whisper_full_default_params(LibWhisper.WHISPER_SAMPLING_GREEDY)
    @test wparams != C_NULL
end

function transcribe_test(audio_file, txt_file; model="base.en", accuracy=.99) 
    audio_file_path = DataDeps.resolve("WhisperSamples/$audio_file", "__FILE__")
    s = load(audio_file_path)  # This loads float32 data, which is what we need
    sout = SampleBuf(Float32, 16000, round(Int, length(s)*(16000/samplerate(s))), nchannels(s))  # Whisper expects 16kHz sample rate
    write(SampleBufSink(sout), SampleBufSource(s))  # Resample
    if nchannels(sout) == 1
        ss = sout.data
    elseif nchannels(sout) == 2
        sd = sout.data
        ss = [sd[i,1] + sd[i,2] for i in 1:size(sd)[1]] #convert stereo to mono
    else 
        @test false
    end
    result = transcribe(model, ss)
    result = replace(result, "[BLANK_AUDIO]"=>"")
    result = replace(result, "[ Silence ]"=>"")
    expected = readlines(joinpath(@__DIR__, txt_file))[1]
    # println(result)
    # println(expected)
    @test compare(expected, result, Levenshtein()) > accuracy
end

@testset "Transcription with base.en" begin
    transcribe_test("gb0.oga", "gb0.txt")
    transcribe_test("gb1.ogg", "gb1.txt")
    transcribe_test("hp0.ogg", "hp0.txt")
    transcribe_test("mm1.wav", "mm1.txt", accuracy=.90)
end

@testset "Transcription with tiny.en" begin
    transcribe_test("gb0.oga", "gb0.txt", model="tiny.en", accuracy=.90)
    transcribe_test("gb1.ogg", "gb1.txt", model="tiny.en", accuracy=.90)
    transcribe_test("hp0.ogg", "hp0.txt", model="tiny.en", accuracy=.90)
    transcribe_test("mm1.wav", "mm1.txt", model="tiny.en", accuracy=.80)
end


