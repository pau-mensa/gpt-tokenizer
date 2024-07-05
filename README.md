# gpt-tokenizer
This is just a very simple implementation of the tokenizer of OpenAI's GPT. The idea was taken from https://www.youtube.com/watch?v=zduSFxRajkE

## Implemented

- A version of a simple tokenizer that applies the BPE algorithm over the bytes directly.

- A version of a regex tokenizer that applies the same pattern used in GPT4 and then applies the BPE algorithm over the splits.

## Not Implemented

There is no support for special characters. This would be pretty easy to do, the constructor should take a list of special characters and then, after the BPE algorithm, add to the merges dictionary all the special characters with consecutive id's.