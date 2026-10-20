package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
)

func TestToolFromToolFunctionParameters(t *testing.T) {
	tests := []struct {
		name       string
		parameters any
		want       string
	}{
		{
			name:       "zero arguments",
			parameters: map[string]any{"type": "object"},
			want:       `{"type":"object","properties":{}}`,
		},
		{
			name: "one argument",
			parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{"type": "string"},
				},
			},
			want: `{"type":"object","properties":{"query":{"type":"string"}}}`,
		},
		{
			name: "nested argument",
			parameters: json.RawMessage(
				`{"type":"object","properties":{"filter":{"type":"object"}}}`,
			),
			want: `{"type":"object","properties":{"filter":{"type":"object"}}}`,
		},
		{
			name:       "non-object schema",
			parameters: json.RawMessage(`{"type":"string"}`),
			want:       `{"type":"string"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool, err := toolFromTool(llms.Tool{
				Type: "function",
				Function: &llms.FunctionDefinition{
					Name:       "test_tool",
					Parameters: tt.parameters,
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			got, err := json.Marshal(tool.Function.Parameters)
			if err != nil {
				t.Fatal(err)
			}
			var gotSchema, wantSchema any
			if err := json.Unmarshal(got, &gotSchema); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tt.want), &wantSchema); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(gotSchema, wantSchema) {
				t.Errorf("parameters = %s, want %s", got, tt.want)
			}
		})
	}
}

//nolint:funlen // Reuse one server and one continuation for buffered and streamed replay.
func TestGenerateContentFunctionParametersAndReplay(t *testing.T) {
	state := &llms.ProviderState{Provider: "openai.responses", Data: json.RawMessage(`[
		{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque-1","future":{"n":9007199254740993}},
		{"type":"function_call","id":"fc_1","call_id":"call_1","name":"now","arguments":"{}","status":"completed"},
		{"type":"reasoning","id":"rs_2","summary":[{"type":"summary_text","text":"summary"}],"encrypted_content":"opaque-2"},
		{"type":"message","id":"msg_1","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"checking","annotations":[]}],"status":"completed"},
		{"type":"function_call","id":"fc_2","call_id":"call_2","name":"now","arguments":"{}","status":"completed"}
	]`)}
	toolCalls := []llms.ToolCall{
		{ID: "call_1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "now", Arguments: "{}"}},
		{ID: "call_2", Type: "function", FunctionCall: &llms.FunctionCall{Name: "now", Arguments: "{}"}},
	}
	message := ChatMessage{Role: "assistant", Content: "checking", ReasoningContent: "summary",
		ToolCalls: toolCallsFromToolCalls(toolCalls), ProviderState: state}
	messageJSON, err := json.Marshal(message)
	require.NoError(t, err)
	finalState := `{"provider":"openai.responses","data":[{"type":"reasoning","id":"rs_3","summary":[],"encrypted_content":"opaque-3"},{"type":"message","id":"msg_2","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"ok","annotations":[]}],"status":"completed"}]}`
	requests := make(chan []byte, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			t.Error(readErr)
			return
		}
		requests <- body
		var req struct {
			Stream bool `json:"stream"`
		}
		if decodeErr := json.Unmarshal(body, &req); decodeErr != nil {
			t.Error(decodeErr)
			return
		}
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			// Snapshots replace earlier state; later deltas without state retain it.
			_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"o\",\"reasoning_content\":\"summary\",\"provider_state\":{\"provider\":\"openai.responses\",\"data\":[{\"type\":\"reasoning\",\"id\":\"rs_3\",\"summary\":[],\"encrypted_content\":\"opaque-3\"}]}}}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"content\":\"k\",\"provider_state\":%s}}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", finalState)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"choices":[{"index":0,"message":%s,"finish_reason":"tool_calls"}]}`, messageJSON)
	}))
	defer server.Close()

	llm, err := New(WithToken("test-token"), WithBaseURL(server.URL), WithModel("gpt-4o"))
	require.NoError(t, err)
	history := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what time is it")}
	resp, err := llm.GenerateContent(context.Background(), history,
		llms.WithFunctions([]llms.FunctionDefinition{{Name: "now", Parameters: map[string]any{"type": "object"}}}),
	)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	require.Equal(t, "checking", resp.Choices[0].Content)
	require.Equal(t, "summary", resp.Choices[0].ReasoningContent)
	require.Equal(t, toolCalls, resp.Choices[0].ToolCalls)
	require.NotNil(t, resp.Choices[0].ProviderState)
	require.Equal(t, state.Provider, resp.Choices[0].ProviderState.Provider)
	require.JSONEq(t, string(state.Data), string(resp.Choices[0].ProviderState.Data))

	var req struct {
		Messages []ChatMessage `json:"messages"`
		Tools    []struct {
			Function struct {
				Parameters any `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(<-requests, &req))
	require.Len(t, req.Tools, 1)
	require.Nil(t, req.Messages[0].ProviderState)
	want := map[string]any{"type": "object", "properties": map[string]any{}}
	require.Equal(t, want, req.Tools[0].Function.Parameters)

	// OpenAI choices have no interface-valued Parts; their state and tool calls
	// can also survive an application's persisted history, not just pointer reuse.
	saved, err := json.Marshal(resp.Choices[0])
	require.NoError(t, err)
	var restored llms.ContentChoice
	require.NoError(t, json.Unmarshal(saved, &restored))
	history = append(history, (&llms.ContentResponse{Choices: []*llms.ContentChoice{&restored}}).AssistantMessage())
	for _, call := range toolCalls {
		history = append(history, llms.MessageContent{Role: llms.ChatMessageTypeTool,
			Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: call.ID, Content: "12:00"}}})
	}
	// A new client must recover state entirely from explicitly retained history.
	llm, err = New(WithToken("test-token"), WithBaseURL(server.URL), WithModel("gpt-4o"))
	require.NoError(t, err)
	var text, reasoning string
	resp, err = llm.GenerateContent(context.Background(), history,
		llms.WithStreamingReasoningFunc(func(_ context.Context, reasoningChunk, chunk []byte) error {
			text += string(chunk)
			reasoning += string(reasoningChunk)
			return nil
		}))
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(<-requests, &req))
	require.Len(t, req.Messages, 4)
	require.NotNil(t, req.Messages[1].ProviderState)
	require.Equal(t, state.Provider, req.Messages[1].ProviderState.Provider)
	require.JSONEq(t, string(state.Data), string(req.Messages[1].ProviderState.Data))
	require.Contains(t, string(req.Messages[1].ProviderState.Data), "9007199254740993")
	require.Equal(t, message.ToolCalls, req.Messages[1].ToolCalls)
	require.Equal(t, "checking", req.Messages[1].Content)
	for i, call := range toolCalls {
		require.Equal(t, call.ID, req.Messages[i+2].ToolCallID)
		require.Nil(t, req.Messages[i+2].ProviderState)
	}
	require.Equal(t, "ok", text)
	require.Equal(t, "summary", reasoning)
	require.Equal(t, text, resp.Choices[0].Content)
	require.Equal(t, reasoning, resp.Choices[0].ReasoningContent)
	replayed, err := json.Marshal(resp.AssistantMessage().ProviderState)
	require.NoError(t, err)
	require.JSONEq(t, finalState, string(replayed))
}

func TestGetModelCapabilities(t *testing.T) {
	tests := []struct {
		model        string
		wantThinking bool
		wantSystem   bool
	}{
		{"o1-mini", true, false},
		{"o3", true, false},
		{"o4-mini", true, true},
		{"o3-2025-04-16", true, true},
		{"gpt-5", true, true},
		{"gpt-5-mini", true, true},
		{"gpt-4o", false, true},
		{"gpt-4", false, true},
		{"gpt-3.5-turbo", false, true},
	}
	for _, tt := range tests {
		caps := getModelCapabilities(tt.model)
		if caps.SupportsThinking != tt.wantThinking {
			t.Errorf("getModelCapabilities(%q).SupportsThinking = %v, want %v", tt.model, caps.SupportsThinking, tt.wantThinking)
		}
		if caps.SupportsSystem != tt.wantSystem {
			t.Errorf("getModelCapabilities(%q).SupportsSystem = %v, want %v", tt.model, caps.SupportsSystem, tt.wantSystem)
		}
	}
}

func TestClampReasoningEffort(t *testing.T) {
	tests := []struct {
		effort llms.ThinkingEffort
		want   string
	}{
		{"low", "low"},
		{"medium", "medium"},
		{"high", "high"},
		// OpenAI accepts xhigh, so it must not be downgraded.
		{"xhigh", "xhigh"},
		// max is Anthropic-only; it clamps to OpenAI's highest.
		{"max", "xhigh"},
		{"", ""},
	}
	for _, tt := range tests {
		if got := clampReasoningEffort(tt.effort); got != tt.want {
			t.Errorf("clampReasoningEffort(%q) = %q, want %q", tt.effort, got, tt.want)
		}
	}
}
