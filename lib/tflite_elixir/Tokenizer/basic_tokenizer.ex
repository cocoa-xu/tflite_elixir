defmodule TFLiteElixir.Tokenizer.BasicTokenizer do
  @moduledoc """
  Runs basic tokenization such as punctuation spliting, lower casing.
  """

  @doc """
  Split text on whitespace, dropping the runs between words.
  """
  @spec split_by_whitespace(binary()) :: [String.t()]
  def split_by_whitespace(text) when is_binary(text) do
    :tflite_beam_basic_tokenizer.split_by_whitespace(text)
  end

  @doc """
  Tokenizes a piece of text.
  """
  @spec tokenize(String.t(), boolean()) :: [String.t()]
  def tokenize(text, is_case_insensitive) do
    :tflite_beam_basic_tokenizer.tokenize(text, is_case_insensitive)
  end
end
