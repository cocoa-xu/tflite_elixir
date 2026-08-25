defmodule TFLiteElixir.Version.Test do
  use ExUnit.Case

  # The runtime comes from LiteRT now, and the Elixir side can load delegate
  # plugins through TFLiteElixir.Delegate.external/1 without upstream offering
  # any binary stable interface between them. Matching versions is the whole of
  # what makes that safe, so the number has to be reachable from Elixir rather
  # than only from :tflite_beam.
  test "the version a plugin has to match is LiteRT's" do
    version = TFLiteElixir.tflite_version()
    assert is_binary(version)
    assert version == :tflite_beam.tflite_version()

    # not TensorFlow's, which is a separate line and a separate number
    tensorflow = TFLiteElixir.tensorflow_version()
    assert is_binary(tensorflow)
    assert tensorflow == :tflite_beam.tensorflow_version()
    refute version == tensorflow
  end

  # A stale precompiled artifact looks exactly like a fresh one from the outside,
  # so this asks rather than trusting the build. A release from before the move
  # has no such function and fails here with undef rather than a wrong answer.
  test "the loaded object came from LiteRT" do
    assert :litert == TFLiteElixir.source_tree()
  end
end
