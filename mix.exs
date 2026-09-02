defmodule TFLiteElixir.MixProject do
  use Mix.Project

  @app :tflite_elixir
  @version "1.0.0-rc4"
  @github_url "https://github.com/cocoa-xu/tflite_elixir"

  def project do
    [
      app: @app,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      source_url: @github_url,
      description: description(),
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Set TFLITE_BEAM_PATH to develop against a checkout of tflite_beam rather
  # than the published version. The two repositories move together often enough
  # that doing this by editing mix.exs invites committing the edit.
  defp tflite_beam_dep do
    case System.get_env("TFLITE_BEAM_PATH") do
      nil -> {:tflite_beam, "1.0.0-rc5"}
      path -> {:tflite_beam, path: path, override: true}
    end
  end

  defp deps do
    [
      tflite_beam_dep(),
      {:nx, "~> 0.11"},
      {:stb_image, "~> 0.6"},
      {:ex_doc, "~> 0.27", only: :docs, runtime: false}
    ]
  end

  defp description() do
    "TensorFlow Lite Elixir binding with optional TPU support."
  end

  defp package() do
    [
      name: to_string(@app),
      files: ~w(
        lib
        .formatter.exs
        mix.exs
        README*
        LICENSE*
      ),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @github_url}
    ]
  end
end
