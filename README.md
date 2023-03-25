## Whisper

[![Build Status](https://github.com/aviks/Whisper.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/aviks/Whisper.jl/actions/workflows/CI.yml?query=branch%3Amain)


Whisper.jl is a Julia package for automatic speech recognition, based on [OpenAI's Whisper](https://github.com/openai/whisper) model. This package wraps the [whisper.cpp](https://github.com/ggerganov/whisper.cpp) code, which is a C/C++ implementation of the model. It uses the model weights which were published by OpenAI. The weights are downloaded on demand. 

### Quick Start Guide

The `transcribe` function is the simplest way to run speech recognition on an 
audio signal. It takes as input the model name, and a float array with the sound signal. 
The sound signal is expected to be sampled at 16kHz. 

```
    # Load audio file using FileIO and LibSndFile, returns SampledSignals
    using Whisper, LibSndFile, FileIO, SampledSignals
    s = load("/path/to/audio_file.ogg")  
    # Whisper expects 16kHz sample rate and Float32 data
    sout = SampleBuf(Float32, 16000, round(Int, length(s)*(16000/samplerate(s))), nchannels(s))  
    write(SampleBufSink(sout), SampleBufSource(s))  # Resample
    if nchannels(sout) == 1
        data  = sout.data
    elseif nchannels(sout) == 2
        sd = sout.data
        data = [sd[i,1] + sd[i,2] for i in 1:size(sd)[1]] #convert stereo to mono
    end

    result = transcribe("base.en", data)
```

For more control, the [whisper.cpp C interface](https://github.com/ggerganov/whisper.cpp/blob/master/whisper.h) is available within the [`Whisper.LibWhisper` module](https://github.com/aviks/Whisper.jl/blob/main/src/LibWhisper.jl). 

### TODO

The biggest missing functionality is a streaming interface. 