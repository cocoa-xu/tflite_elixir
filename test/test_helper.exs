# The LiteRT API is a build option of tflite_beam and is off by default. Rather
# than every LiteRT test failing with "not compiled into this build", they are
# excluded unless the build has it.
litert_exclusions =
  if TFLiteElixir.LiteRT.CompiledModel.available?(), do: [], else: [litert: true]

# armv6 and armv7l build with XNNPACK off, and two delegate cases need it. The
# tag was already on them and was never excluded anywhere, so on such a build
# they failed rather than stepping aside.
xnnpack_exclusions =
  if :xnnpack in :tflite_beam_delegate.available(), do: [], else: [require_xnnpack: true]

ExUnit.configure(
  exclude:
    [
      # exclude all tests that require a physical TPU by default
      require_tpu: true
    ] ++ litert_exclusions ++ xnnpack_exclusions
)

ExUnit.start()
