# Explicit reasoning effort

Use `llms.WithReasoningEffort` on each generation call, including calls that
continue a conversation after a tool result:

```go
response, err := model.GenerateContent(ctx, messages,
    llms.WithReasoningEffort("high"),
    llms.WithMaxTokens(4096),
)
```

The option uses the client's existing endpoint and API:

| Client | Request field |
| --- | --- |
| OpenAI Chat Completions | `reasoning_effort` |
| Anthropic Messages | `output_config.effort` |

The same fields are used with compatible gateway URLs, including Bifrost.
LangChainGo does not select a different API based on effort, inspect model names,
normalize the value, or retry without effort. Choose an effort supported by the
actual endpoint and model. Unknown values are forwarded so the endpoint can
validate them. Other providers and Anthropic's legacy Text Completions API do
not implement this option.

An absent option or an empty string omits the field, preserving prior behavior.
Values such as `"none"` are explicit values, not absence. The option is per call;
reapply it when sending tool results or otherwise continuing a conversation.
Streaming uses the same request construction as non-streaming generation.

Effort is separate from `WithThinkingMode` and `WithThinkingBudget`. It does not
enable thinking or calculate a token budget. Anthropic thinking configuration
can coexist with effort and retains its existing budget conversion behavior.

The HTTP contract tests cover direct and gateway URL prefixes, unknown model
aliases, unchanged effort values, omission, streaming, and tool-result
continuations. They do not verify acceptance by live vendor services or run a
live Bifrost instance. Bifrost may perform its own provider-specific conversion.
Existing Codex transports that consume Chat Completions `reasoning_effort` and
convert it into Responses `reasoning.effort` can retain that conversion.

Provider references:
- [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort)
