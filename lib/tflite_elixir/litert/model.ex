defmodule TFLiteElixir.LiteRT.Model do
  @moduledoc """
  Experimental LiteRT graph inspection and operator option rewrites.

  `inspect/2` reports subgraph connectivity and supported OneHot, SplitV and
  StableHLO composite options. Other operators retain their numeric opcode.
  `rewrite/3` returns a new TFLite binary and preserves the source, tensor shapes
  and connectivity. Callers must keep edited options consistent with those
  shapes. This uses LiteRT's experimental builder API and requires a newer
  upstream dependency; older versions return an error.
  """

  import Kernel, except: [inspect: 2]
  import TFLiteElixir.Errorize, only: [deferror: 1]

  alias TFLiteElixir.LiteRT.Compatibility

  @type source :: String.t() | {:buffer, binary()}
  @type edit :: %{subgraph: non_neg_integer(), operator: non_neg_integer(), options: map()}
  @erl :tflite_beam_litert_model

  @doc "Inspect subgraph connectivity and supported operator options."
  @spec inspect(reference(), source()) :: {:ok, [map()]} | {:error, String.t()}
  def inspect(env, source), do: Compatibility.call(@erl, :inspect, [env, source])

  deferror(inspect(env, source))

  @doc "Return a rewritten TFLite binary without modifying the source."
  @spec rewrite(reference(), source(), [edit()]) :: {:ok, binary()} | {:error, String.t()}
  def rewrite(env, source, edits), do: Compatibility.call(@erl, :rewrite, [env, source, edits])

  deferror(rewrite(env, source, edits))
end
