# Whisper model weights, in whisper.cpp's ggml format, fetched on first use from
# HuggingFace through DataDeps.
#
# Sizes are measured Content-Length values. sha1 values were computed from the
# downloaded files. `large.en` is kept for backwards compatibility only: there is
# no English-only large model, and the file it fetches is large-v2.

const MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

const MODELS = (
    (name = "tiny.en",        size = "74 MB",  sha1 = "c78c86eb1a8faa21b369bcd33207cc90d64ae9df", note = "English-only"),
    (name = "tiny",           size = "74 MB",  sha1 = "bd577a113a864445d4c299885e0cb97d4ba92b5f", note = "multilingual"),
    (name = "base.en",        size = "141 MB", sha1 = "137c40403d78fd54d454da0f9bd998f78703390c", note = "English-only"),
    (name = "base",           size = "141 MB", sha1 = "465707469ff3a37a2b9b8d8f89f2f99de7299dac", note = "multilingual"),
    (name = "small.en",       size = "465 MB", sha1 = "db8a495a91d927739e50b3fc1cc4c6b8f6c2d022", note = "English-only"),
    (name = "small",          size = "465 MB", sha1 = "55356645c2b361a969dfd0ef2c5a50d530afd8d5", note = "multilingual"),
    (name = "medium.en",      size = "1.4 GB", sha1 = "8c30f0e44ce9560643ebd10bbe50cd20eafd3723", note = "English-only"),
    (name = "medium",         size = "1.4 GB", sha1 = "fd9727b6e1217c2f614f9b698455c4ffd82463b4", note = "multilingual"),
    (name = "large-v1",       size = "2.9 GB", sha1 = "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299", note = "multilingual"),
    (name = "large-v2",       size = "2.9 GB", sha1 = "0f4c8e34f21cf1a914c59d8b3ce882345ad349d6", note = "multilingual"),
    (name = "large-v3",       size = "2.9 GB", sha1 = "ad82bf6a9043ceed055076d0fd39f5f186ff8062", note = "multilingual"),
    (name = "large-v3-turbo", size = "1.5 GB", sha1 = "4af2b29d7ec73d781377bfd1758ca957a807e941", note = "multilingual, ~8x faster than large-v3 at similar quality"),
)

"""
    available_models() -> Vector{String}

Names accepted by [`WhisperContext`](@ref) and [`transcribe`](@ref). The weights are
downloaded on first use. Models with an `.en` suffix are English-only and slightly
more accurate for English than the multilingual model of the same size.
"""
available_models() = String[m.name for m in MODELS]

"""
    model_path(model) -> String

Local path of the ggml weights for `model`, downloading them first if needed.
`model` may also be a path to an existing `.bin` file, which is returned as is.
"""
function model_path(model::AbstractString)
    isfile(model) && return String(model)
    name = String(model)
    if name == "large.en"
        Base.depwarn("model \"large.en\" is a misnomer (it is large-v2 and not English-only); use \"large-v2\"",
                     :model_path)
    elseif !(name in available_models())
        throw(ArgumentError("unknown model \"$name\" (not an existing file either); " *
                            "available models: " * join(available_models(), ", ")))
    end
    return DataDeps.resolve("whisper-ggml-$name/ggml-$name.bin", "__FILE__")
end

function register_datadeps()
    for m in MODELS
        register(DataDep(
            "whisper-ggml-$(m.name)",
            "Whisper $(m.name) model (ggml format, ~$(m.size), $(m.note)) from $MODEL_BASE_URL",
            "$MODEL_BASE_URL/ggml-$(m.name).bin",
            (sha1, m.sha1),
        ))
    end

    # Deprecated alias kept so existing `transcribe("large.en", ...)` calls work.
    # ggml-large.bin is upstream's alias for large-v2 (identical sha1).
    register(DataDep(
        "whisper-ggml-large.en",
        "DEPRECATED alias for large-v2 (~2.9 GB, multilingual; not English-only)",
        "$MODEL_BASE_URL/ggml-large.bin",
        (sha1, "0f4c8e34f21cf1a914c59d8b3ce882345ad349d6"),
    ))
end
