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
For explicit effort, LangChainGo does not select a different API, gate the
value on model names, normalize it, or retry without it. Choose an effort supported by the
actual endpoint and model. Unknown values are forwarded so the endpoint can
validate them. Other providers and Anthropic's legacy Text Completions API do
not implement this option.

An absent option or an empty string preserves prior behavior, including existing
OpenAI ThinkingMode inference for recognized models. Without ThinkingMode, the
effort field is omitted.
Values such as `"none"` are explicit values, not absence. The option is per call;
reapply it when sending tool results or otherwise continuing a conversation.
Streaming uses the same request construction as non-streaming generation.

Explicit effort overrides OpenAI effort inferred from `WithThinkingMode`,
regardless of option order or model name. It does not calculate a token budget.
For OpenAI requests (including gateways), explicit `"none"` preserves temperature,
including zero; other nonempty efforts omit temperature. Without explicit effort,
legacy model and ThinkingMode behavior is retained. Anthropic thinking configuration
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
