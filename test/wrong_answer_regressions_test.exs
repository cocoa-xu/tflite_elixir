defmodule TFLiteElixir.WrongAnswerRegressionsTest do
  @moduledoc """
  Each case here stands for a way this library used to answer a question it had
  not actually asked the model. They are grouped because they share one shape:
  a result was computed, not looked at, and something plausible was returned in
  its place.
  """
  use ExUnit.Case

  alias TFLiteElixir.{Interpreter, TFLiteTensor}

  @quantised "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
  @unquantised "ssd_mobilenet_v2_coco_quant_postprocess.tflite"

  defp model(name), do: Path.join([__DIR__, "test_data", name])

  defp interpreter(name) do
    i = Interpreter.new!(model(name))
    {:ok, [index | _]} = Interpreter.inputs(i)
    {i, index}
  end

  describe "predict/2 answers only for the run it performed" do
    test "a short input is refused rather than truncated and answered" do
      {i, _} = interpreter(@quantised)

      assert {:error, reason} = Interpreter.predict(i, [<<1, 2, 3>>])
      assert reason =~ "150528"
      assert reason =~ "got 3"
    end

    test "a short input named in a map is refused too" do
      {i, index} = interpreter(@quantised)
      name = Interpreter.tensor(i, index).name

      assert {:error, reason} = Interpreter.predict(i, %{name => <<1, 2, 3>>})
      assert reason =~ "150528"
    end

    test "a bare binary is accepted, as the spec has always said" do
      {i, _} = interpreter(@quantised)
      whole = :binary.copy(<<1>>, 1 * 224 * 224 * 3)

      assert [%Nx.Tensor{}] = Interpreter.predict(i, whole)
    end

    test "a wrong type reports one error, not a list with an error inside it" do
      {i, _} = interpreter(@quantised)
      wrong = Nx.broadcast(Nx.tensor(0.0, type: :f32), {1, 224, 224, 3})

      # a caller matching {:error, _} used to miss this entirely, because the
      # length-mismatch case returned a tuple and this one returned a bare list
      assert {:error, reason} = Interpreter.predict(i, [wrong])
      assert reason =~ "does not match the data type"
    end
  end

  describe "a tensor that is not quantised" do
    test "reading its quantisation parameters does not raise" do
      i = Interpreter.new!(model(@unquantised))
      {:ok, [output | _]} = Interpreter.outputs(i)

      # an empty scale list is what "not quantised" looks like, and [scale] = []
      # raised on it in four places, including two on the input side
      assert [] == Interpreter.tensor(i, output).quantization_params.scale
    end
  end

  describe "handles and types that Nx cannot take" do
    test "to_nx on a dead reference answers instead of raising" do
      assert {:error, reason} = TFLiteTensor.to_nx(make_ref())
      assert reason =~ "cannot access"
    end

    test "reset_variable_tensor accepts the struct every other function takes" do
      {i, index} = interpreter(@quantised)

      # it passed the struct where the NIF wants the reference, so it could
      # never succeed and its @spec of `any` hid that
      assert :ok == TFLiteElixir.reset_variable_tensor(Interpreter.tensor(i, index))
    end
  end
end
