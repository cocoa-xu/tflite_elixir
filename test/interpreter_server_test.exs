defmodule TFLiteElixir.Interpreter.Server.Test do
  use ExUnit.Case

  alias TFLiteElixir.Interpreter
  alias TFLiteElixir.Interpreter.Server

  @model Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])

  defp input(byte), do: :binary.copy(<<byte>>, 224 * 224 * 3)

  test "predict/2 feeds, runs and reads back" do
    {:ok, server} = Server.start(@model)
    assert [<<_::binary>>] = Server.predict(server, [input(7)])
    :ok = Server.stop(server)
  end

  # The reason this module exists: the same thing done directly gets two
  # processes each other's answers.
  test "concurrent callers each get the answer to their own input" do
    {:ok, server} = Server.start(@model)
    quiet = Server.predict(server, [input(7)])
    loud = Server.predict(server, [input(200)])
    assert quiet != loud

    rounds = 100
    parent = self()

    hammer = fn byte, expected ->
      fn ->
        wrong =
          Enum.count(1..rounds, fn _ -> Server.predict(server, [input(byte)]) != expected end)

        send(parent, {self(), wrong})
      end
    end

    first = spawn(hammer.(7, quiet))
    second = spawn(hammer.(200, loud))

    assert_receive {^first, 0}, 120_000
    assert_receive {^second, 0}, 120_000

    :ok = Server.stop(server)
  end

  test "the interpreter cannot be reached from outside the server" do
    {:ok, server} = Server.start(@model)
    interpreter = Server.run(server, & &1)

    assert {:error, _} = Interpreter.invoke(interpreter)
    assert [<<_::binary>>] = Server.predict(server, [input(7)])

    :ok = Server.stop(server)
  end

  test "run/2 covers what predict/2 does not" do
    {:ok, server} = Server.start(@model)
    assert Server.run(server, &Interpreter.tensors_size/1) > 0
    :ok = Server.stop(server)
  end
end
