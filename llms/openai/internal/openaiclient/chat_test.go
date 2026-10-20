package openaiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
)

func TestParseStreamingChatResponse_FinishReason(t *testing.T) {
	ctx := context.Background()
	t.Parallel()
	mockBody := `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}`
	r := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewBufferString(mockBody)),
	}

	req := &ChatRequest{
		StreamingFunc: func(_ context.Context, _ []byte) error {
			return nil
		},
	}

	resp, err := parseStreamingChatResponse(ctx, r, req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, FinishReason("stop"), resp.Choices[0].FinishReason)
}

func TestParseStreamingChatResponse_ReasoningContent(t *testing.T) {
	ctx := context.Background()
	t.Parallel()
	// A complete encrypted state can exceed bufio.Scanner's default 64 KiB limit.
	state := `{"provider":"openai.responses","data":[{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"` +
		strings.Repeat("opaque", 16384) +
		`"},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"one","arguments":"{}"},{"type":"function_call","id":"fc_2","call_id":"call_2","name":"two","arguments":"{}"}]}`
	mockBody := `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"final answer","reasoning_content":"step-by-step reasoning","tool_calls":[{"id":"call_1","type":"function","function":{"name":"one","arguments":"{}"}},{"id":"call_2","type":"function","function":{"name":"two","arguments":"{}"}}],"provider_state":` + state + `},"finish_reason":"tool_calls"}]}`
	r := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewBufferString(mockBody)),
	}

	req := &ChatRequest{
		StreamingFunc: func(_ context.Context, _ []byte) error {
			return nil
		},
	}

	resp, err := parseStreamingChatResponse(ctx, r, req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "final answer", resp.Choices[0].Message.Content)
	assert.Equal(t, "step-by-step reasoning", resp.Choices[0].Message.ReasoningContent)
	assert.Equal(t, FinishReason("tool_calls"), resp.Choices[0].FinishReason)
	require.Len(t, resp.Choices[0].Message.ToolCalls, 2)
	assert.Equal(t, "call_1", resp.Choices[0].Message.ToolCalls[0].ID)
	assert.Equal(t, "call_2", resp.Choices[0].Message.ToolCalls[1].ID)
	rawState, err := json.Marshal(resp.Choices[0].Message.ProviderState)
	require.NoError(t, err)
	require.JSONEq(t, state, string(rawState))
}

func TestParseStreamingChatResponse_ReasoningFunc(t *testing.T) {
	ctx := context.Background()
	t.Parallel()
	mockBody := `
data: {"id":"fa7e4fc5-a05d-4e7b-9a66-a2dd89e91a4e","object":"chat.completion.chunk","created":1738492867,"model":"deepseek-reasoner","system_fingerprint":"fp_7e73fd9a08","choices":[{"index":0,"delta":{"content":null,"reasoning_content":"Okay"},"logprobs":null,"finish_reason":null}]}
`
	r := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewBufferString(mockBody)),
	}

	req := &ChatRequest{
		StreamingReasoningFunc: func(_ context.Context, reasoningChunk, chunk []byte) error {
			t.Logf("reasoningChunk: %s", string(reasoningChunk))
			t.Logf("chunk: %s", string(chunk))
			return nil
		},
	}

	resp, err := parseStreamingChatResponse(ctx, r, req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "", resp.Choices[0].Message.Content)
	assert.Equal(t, "Okay", resp.Choices[0].Message.ReasoningContent)
	assert.Equal(t, FinishReason(""), resp.Choices[0].FinishReason)
}

func TestChatMessage_MarshalUnmarshal(t *testing.T) {
	t.Parallel()
	msg := ChatMessage{
		Role:    "assistant",
		Content: "hello",
		FunctionCall: &FunctionCall{
			Name:      "test",
			Arguments: "func",
		},
	}
	text, err := json.Marshal(msg)
	require.NoError(t, err)
	require.Equal(t, `{"role":"assistant","content":"hello","function_call":{"name":"test","arguments":"func"}}`, string(text)) // nolint: lll

	var msg2 ChatMessage
	err = json.Unmarshal(text, &msg2)
	require.NoError(t, err)
	require.Equal(t, msg, msg2)
}

func TestChatMessage_MarshalUnmarshal_WithReasoning(t *testing.T) {
	t.Parallel()
	msg := ChatMessage{
		Role:             "assistant",
		Content:          "final answer",
		ReasoningContent: "step-by-step reasoning",
		ProviderState: &llms.ProviderState{
			Provider: "openai.responses",
			Data:     json.RawMessage(`[{"type":"reasoning","summary":[],"encrypted_content":"opaque","future":{"n":9007199254740993}}]`),
		},
	}
	text, err := json.Marshal(msg)
	require.NoError(t, err)
	require.Equal(t, `{"role":"assistant","content":"final answer","reasoning_content":"step-by-step reasoning","provider_state":{"provider":"openai.responses","data":[{"type":"reasoning","summary":[],"encrypted_content":"opaque","future":{"n":9007199254740993}}]}}`, string(text))

	var msg2 ChatMessage
	err = json.Unmarshal(text, &msg2)
	require.NoError(t, err)
	require.Equal(t, msg, msg2)

	// Exercise the separate multipart marshaler without changing state.
	msg.Content = ""
	msg.MultiContent = []llms.ContentPart{llms.TextPart("one"), llms.TextPart("two")}
	text, err = json.Marshal(msg)
	require.NoError(t, err)
	var multipart struct {
		Content       []json.RawMessage   `json:"content"`
		ProviderState *llms.ProviderState `json:"provider_state"`
	}
	require.NoError(t, json.Unmarshal(text, &multipart))
	require.Len(t, multipart.Content, 2)
	require.Equal(t, msg.ProviderState, multipart.ProviderState)
}
