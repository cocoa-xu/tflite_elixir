defmodule TFLiteElixir.BangVariantsTest do
  use ExUnit.Case

  # Every raising variant comes from a deferror/1 line written by hand next to
  # the function it wraps, so a function that later grew an arity kept a bang
  # variant that no longer covered it. build_from_file/1 had no
  # build_from_file!/1 at all: the plain one got its arity from a default
  # argument the bang one did not have, and calling it raised undef rather than
  # the model's own error.
  test "a bang variant covers every arity of the function it wraps" do
    gaps =
      Path.wildcard("#{:code.lib_dir(:tflite_elixir)}/ebin/Elixir.TFLiteElixir*.beam")
      |> Enum.map(&(&1 |> Path.basename(".beam") |> String.to_atom()))
      |> Enum.flat_map(fn module ->
        Code.ensure_loaded(module)
        by_name = Enum.group_by(module.__info__(:functions), &elem(&1, 0), &elem(&1, 1))

        for {name, arities} <- by_name,
            text = Atom.to_string(name),
            String.ends_with?(text, "!"),
            plain = text |> String.trim_trailing("!") |> String.to_atom(),
            plain_arities = by_name[plain],
            plain_arities != nil,
            missing = Enum.sort(plain_arities) -- Enum.sort(arities),
            missing != [] do
          "#{inspect(module)}.#{name} is missing #{inspect(missing)}, " <>
            "which #{plain}/#{Enum.join(Enum.sort(plain_arities), ",")} has"
        end
      end)

    assert gaps == [], Enum.join(gaps, "\n")
  end

  test "a bang variant raises the message the plain one returns" do
    missing = "/definitely/not/a/model.tflite"

    assert {:error, reason} = TFLiteElixir.FlatBufferModel.build_from_file(missing)

    assert_raise RuntimeError, to_string(reason), fn ->
      TFLiteElixir.FlatBufferModel.build_from_file!(missing)
    end

    assert {:error, reason} = TFLiteElixir.Interpreter.new(missing)

    assert_raise RuntimeError, to_string(reason), fn ->
      TFLiteElixir.Interpreter.new!(missing)
    end
  end
end
