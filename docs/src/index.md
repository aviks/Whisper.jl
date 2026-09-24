```@meta
CurrentModule = Whisper
```

# Whisper

Speech recognition in Julia with [OpenAI's Whisper](https://github.com/openai/whisper),
through [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## Quick start

```julia
using Whisper

# audio: Vector{Float32}, 16 kHz, mono, values in [-1, 1]
text = transcribe("base.en", audio)

# reuse a loaded model
ctx = WhisperContext("large-v3-turbo")
text = transcribe(ctx, audio; language = "auto", sampling = :beam)
for s in segments(ctx)
    println(s.t0, " – ", s.t1, ": ", s.text)
end
close(ctx)
```

Models are downloaded on first use; see [`available_models`](@ref). GPU builds of
whisper.cpp (CUDA on Linux, Metal on Apple Silicon) are picked automatically by the
package manager when available and used by default.

## API

```@docs
transcribe
WhisperContext
segments
available_models
detected_language
is_multilingual
model_path
log_level!
version
```

## The C API

The complete whisper.cpp C API is available as `Whisper.LibWhisper`. It is generated
from `whisper.h` with Clang.jl; regenerate it with `gen/generator.jl` after bumping
`whisper_cpp_jll`.

```@index
Modules = [Whisper.LibWhisper]
Order = [:function]
```
