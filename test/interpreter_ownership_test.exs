defmodule TFLiteElixir.InterpreterOwnershipTest do
  use ExUnit.Case

  alias TFLiteElixir.Interpreter

  @quantised Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
  @tiny Path.join([__DIR__, "test_data", "fp8_types.bin"])

  defp whole_input, do: :binary.copy(<<1>>, 1 * 224 * 224 * 3)

  defp every_shape(interpreter, whole) do
    {:ok, [index]} = Interpreter.inputs(interpreter)
    name = Interpreter.tensor(interpreter, index).name
    as_nx = whole |> Nx.from_binary(:u8) |> Nx.reshape({1, 224, 224, 3})
    [whole, [whole], %{name => whole}, as_nx, [as_nx], %{name => as_nx}]
  end

  test "an interpreter belongs to nobody until claimed, and can be given back" do
    interpreter = Interpreter.new!(@quantised)
    me = self()

    assert :undefined == Interpreter.controlling_process(interpreter)
    assert :ok == Interpreter.controlling_process(interpreter, me)
    assert {:ok, ^me} = Interpreter.controlling_process(interpreter)
    assert :ok == Interpreter.controlling_process(interpreter, :undefined)
    assert :undefined == Interpreter.controlling_process(interpreter)
  end

  test "a claimed interpreter refuses another process by name, whatever shape the input takes" do
    interpreter = Interpreter.new!(@quantised)
    whole = whole_input()
    shapes = every_shape(interpreter, whole)
    :ok = Interpreter.controlling_process(interpreter, self())

    for input <- shapes do
      assert {:error, reason} =
               Task.await(Task.async(fn -> Interpreter.predict(interpreter, input) end))

      assert reason =~ "belongs to another process"
    end

    assert {:error, reason} =
             Task.await(
               Task.async(fn -> Interpreter.controlling_process(interpreter, self()) end)
             )

    assert reason =~ "belongs to another process"

    # and the owner is still answered
    assert [%Nx.Tensor{}] = Interpreter.predict(interpreter, whole)
  end

  # The Erlang predict/2 answers every one of these by name. The port carried
  # the happy clauses and not the refusals, so each of them was a
  # FunctionClauseError from a private function, and a map value that was an Nx
  # tensor of the wrong type but the right byte count was written unchecked.
  test "input that is not tensor data is refused by name, whichever way it arrives" do
    interpreter = Interpreter.new!(@quantised)
    {:ok, [index]} = Interpreter.inputs(interpreter)
    name = Interpreter.tensor(interpreter, index).name

    assert {:error, reason} = Interpreter.predict(interpreter, :nope)
    assert reason =~ "input must be"
    assert reason =~ ":nope"

    assert {:error, reason} = Interpreter.predict(interpreter, [[1.0, 2.0]])
    assert reason =~ "tensor index 0"
    assert reason =~ "[1.0, 2.0]"

    assert {:error, reason} = Interpreter.predict(interpreter, %{name => :nope})
    assert reason =~ "tensor index 0"
    assert reason =~ ":nope"

    same_bytes_wrong_type = Nx.broadcast(Nx.tensor(1, type: :s8), {1, 224, 224, 3})

    assert {:error, reason} = Interpreter.predict(interpreter, %{name => same_bytes_wrong_type})
    assert reason =~ "does not match the data type of the tensor, {:u, 8}"

    whole = whole_input()
    assert [%Nx.Tensor{}] = Interpreter.predict(interpreter, %{name => whole})

    assert [%Nx.Tensor{}] =
             Interpreter.predict(
               interpreter,
               whole |> Nx.from_binary(:u8) |> Nx.reshape({1, 224, 224, 3})
             )
  end

  test "printing the state of a claimed interpreter from elsewhere is refused, not swallowed" do
    interpreter = Interpreter.new!(@tiny)
    :ok = Interpreter.controlling_process(interpreter, self())

    assert {:error, reason} =
             Task.await(Task.async(fn -> TFLiteElixir.print_interpreter_state(interpreter) end))

    assert reason =~ "belongs to another process"
    assert nil == TFLiteElixir.print_interpreter_state(interpreter)
  end

  # TfLite defines -1 as "let the runtime choose" and 0 as 1, the Erlang side
  # takes both and says so, and the guards here refused everything below 1.
  test "set_num_threads/2 takes the -1 that asks the runtime to choose" do
    interpreter = Interpreter.new!(@quantised)
    assert :ok == Interpreter.set_num_threads(interpreter, -1)
    assert :ok == Interpreter.set_num_threads(interpreter, 0)

    model = TFLiteElixir.FlatBufferModel.build_from_file!(@quantised)
    resolver = TFLiteElixir.Ops.Builtin.BuiltinResolver.new!()
    builder = TFLiteElixir.InterpreterBuilder.new!(model, resolver)
    assert :ok == TFLiteElixir.InterpreterBuilder.set_num_threads(builder, -1)
    # below -1 the NIF says what it expects, where the guard here refused with
    # a FunctionClauseError of its own
    assert {:error, reason} = Interpreter.set_num_threads(interpreter, -2)
    assert reason =~ "-1"
    assert {:error, _} = TFLiteElixir.InterpreterBuilder.set_num_threads(builder, -2)
  end
end
