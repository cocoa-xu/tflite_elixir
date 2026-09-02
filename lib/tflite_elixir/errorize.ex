defmodule TFLiteElixir.Errorize do
  @moduledoc false

  # Original version by José
  # https://gist.github.com/josevalim/7a5ed50ed86a2260d907603ca8223448
  # modified a tiny bit by Cocoa
  #
  # The decision of what to raise lives in classify/1 rather than in the
  # generated function. Written out in every bang variant, the charlist branch
  # and the fallthrough were dead wherever the wrapped function only ever
  # answers a binary reason, and dialyzer said so 170 times, which took an
  # ignore file covering three warning classes in eleven files to silence and
  # hid two warnings about hand-written code with them.
  defmacro deferror(fun) do
    {name, args} = Macro.decompose_call(fun)

    doc = """
    Raising version of `#{name}/#{length(args)}`.
    """

    quote do
      @doc unquote(doc)
      def unquote(:"#{name}!")(unquote_splicing(args)) do
        case TFLiteElixir.Errorize.classify(unquote(fun)) do
          {:ok, res} -> res
          {:raise, message} -> raise message
        end
      end
    end
  end

  @doc false
  @spec classify(term()) :: {:ok, term()} | {:raise, String.t()}
  def classify({:ok, res}), do: {:ok, res}
  def classify({:error, message}) when is_list(message), do: {:raise, List.to_string(message)}
  def classify({:error, message}) when is_binary(message), do: {:raise, message}
  def classify(res), do: {:ok, res}
end
