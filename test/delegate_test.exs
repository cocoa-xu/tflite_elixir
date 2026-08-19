defmodule TFLiteElixir.Delegate.Test do
  use ExUnit.Case

  alias TFLiteElixir.Delegate
  alias TFLiteElixir.FlatBufferModel
  alias TFLiteElixir.Interpreter
  alias TFLiteElixir.InterpreterBuilder
  alias TFLiteElixir.Ops.Builtin.BuiltinResolver

  @model Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
  @unbuildable Path.join([__DIR__, "test_data", "0_subgraphs.bin"])

  test "available/0 says what this build can construct" do
    available = Delegate.available()
    assert is_list(available)
    # loading a plugin needs only the dynamic loader, so this one is everywhere
    assert :external in available
  end

  test "xnnpack/1 rejects an unknown flag by name" do
    assert {:error, reason} = Delegate.xnnpack(flags: [:not_a_flag])
    assert reason =~ "not_a_flag"
  end

  test "external/1 on a path that is not there is an error, and external!/1 raises" do
    assert {:error, reason} = Delegate.external("/nowhere/libnothing.so")
    assert reason =~ "no such delegate library"

    assert_raise RuntimeError, ~r/no such delegate library/, fn ->
      Delegate.external!("/nowhere/libnothing.so")
    end
  end

  test "external/1 on a library that is not a delegate plugin is an error" do
    # a real file that the loader will refuse, rather than a missing one
    assert {:error, reason} = Delegate.external(@unbuildable)
    assert reason =~ "cannot load delegate library"
  end

  @tag :require_xnnpack
  test "the default path and TfLite's own lazy delegation land in the same place" do
    default = run(BuiltinResolver.new!())
    lazy = run(BuiltinResolver.new!(apply_default_delegates: true))

    assert default == lazy
  end

  @tag :require_xnnpack
  test "an explicitly attached delegate suppresses the default one" do
    {:ok, delegate} = Delegate.xnnpack(num_threads: 2)

    explicit =
      run(BuiltinResolver.new!(), fn builder ->
        InterpreterBuilder.add_delegate!(builder, delegate)
      end)

    assert run(BuiltinResolver.new!()) == explicit
  end

  test "build/2 reports a build that failed, and build!/2 raises on it" do
    model = FlatBufferModel.build_from_file!(@unbuildable, [])
    resolver = BuiltinResolver.new!()
    builder = InterpreterBuilder.new!(model, resolver)
    interpreter = Interpreter.new!()

    assert {:error, _} = InterpreterBuilder.build(builder, interpreter)

    builder2 = InterpreterBuilder.new!(model, BuiltinResolver.new!())
    interpreter2 = Interpreter.new!()

    assert_raise RuntimeError, fn ->
      InterpreterBuilder.build!(builder2, interpreter2)
    end
  end

  test "add_delegate/3 refuses an unknown on_decline value before reaching the NIF" do
    model = FlatBufferModel.build_from_file!(@model, [])
    builder = InterpreterBuilder.new!(model, BuiltinResolver.new!())
    {:ok, delegate} = Delegate.xnnpack()

    assert {:error, reason} =
             InterpreterBuilder.add_delegate(builder, delegate, on_decline: :perhaps)

    assert reason =~ "on_decline"
  end

  @tag :require_tpu
  test "an Edge TPU delegate matches make_edge_tpu_interpreter/2" do
    model =
      Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"])

    assert through_edge_tpu_context(model) == through_edge_tpu_delegate(model)
  end

  defp through_edge_tpu_context(model_path) do
    model = FlatBufferModel.build_from_file!(model_path, [])
    context = TFLiteElixir.Coral.get_edge_tpu_context!()
    interpreter = TFLiteElixir.Coral.make_edge_tpu_interpreter!(model, context)
    :ok = Interpreter.allocate_tensors(interpreter)
    Interpreter.predict(interpreter, [tpu_input()])
  end

  defp through_edge_tpu_delegate(model_path) do
    model = FlatBufferModel.build_from_file!(model_path, [])
    builder = InterpreterBuilder.new!(model, BuiltinResolver.new!())
    {:ok, delegate} = TFLiteElixir.Coral.edge_tpu_delegate()
    :ok = InterpreterBuilder.add_delegate!(builder, delegate)
    interpreter = Interpreter.new!()
    :ok = InterpreterBuilder.build!(builder, interpreter)
    :ok = Interpreter.allocate_tensors(interpreter)
    Interpreter.predict(interpreter, [tpu_input()])
  end

  defp tpu_input, do: :binary.copy(<<7>>, 224 * 224 * 3)

  defp run(resolver, configure \\ fn _ -> :ok end) do
    model = FlatBufferModel.build_from_file!(@model, [])
    builder = InterpreterBuilder.new!(model, resolver)
    interpreter = Interpreter.new!()
    :ok = configure.(builder)
    :ok = InterpreterBuilder.build!(builder, interpreter)
    :ok = Interpreter.allocate_tensors(interpreter)

    input = :binary.copy(<<7>>, 224 * 224 * 3)

    {Interpreter.predict(interpreter, [input]), length(Interpreter.execution_plan(interpreter)),
     Interpreter.nodes_size(interpreter)}
  end
end
