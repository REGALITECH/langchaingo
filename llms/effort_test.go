package llms_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
	"github.com/tmc/langchaingo/llms/openai"
)

// Exercise the public option through the real HTTP clients, including gateway
// URL prefixes and tool-result history. The gateway itself is not simulated.
func TestExplicitReasoningEffortHTTP(t *testing.T) {
	for _, provider := range []string{"openai", "anthropic"} {
		for _, gateway := range []bool{false, true} {
			for _, stream := range []bool{false, true} {
				for _, effort := range []string{"", "none", "minimal", "low", "medium", "high", "xhigh", "max", "future-value"} {
					name := fmt.Sprintf("%s/gateway=%t/stream=%t/effort=%s", provider, gateway, stream, effort)
					t.Run(name, func(t *testing.T) {
						model := "unknown-model-alias"
						prefix := "/v1"
						if gateway {
							prefix = "/" + provider
							model = provider + "/" + model
						}
						requests := 0
						server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							requests++
							var body map[string]any
							if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
								t.Error(err)
								w.WriteHeader(400)
								return
							}
							if body["model"] != model {
								t.Errorf("model = %v, want %s", body["model"], model)
							}
							expectedPath := prefix + "/chat/completions"
							key := "reasoning_effort"
							if provider == "anthropic" {
								expectedPath = prefix + "/messages"
								key = "output_config"
							}
							if r.URL.Path != expectedPath {
								t.Errorf("path = %s, want %s", r.URL.Path, expectedPath)
							}
							value, exists := body[key]
							if effort == "" {
								if exists {
									t.Errorf("unspecified effort emitted %s: %v", key, value)
								}
							} else if provider == "anthropic" {
								config, ok := value.(map[string]any)
								if !ok || config["effort"] != effort {
									t.Errorf("output_config = %v, want effort %s", value, effort)
								}
							} else if value != effort {
								t.Errorf("reasoning_effort = %v, want %s", value, effort)
							}
							if _, exists := body["thinking"]; exists {
								t.Error("effort unexpectedly enabled budget-based thinking")
							}
							if (body["stream"] == true) != stream {
								t.Errorf("stream = %v", body["stream"])
							}
							if requests == 2 {
								messages := body["messages"].([]any)
								if len(messages) != 3 {
									t.Errorf("tool continuation messages = %v", messages)
								}
								last := messages[len(messages)-1].(map[string]any)
								if provider == "openai" {
									if last["role"] != "tool" || last["tool_call_id"] != "call_1" {
										t.Errorf("tool result = %v", last)
									}
								} else {
									content := last["content"].([]any)[0].(map[string]any)
									if content["type"] != "tool_result" || content["tool_use_id"] != "call_1" {
										t.Errorf("tool result = %v", last)
									}
								}
							}
							writeEffortResponse(w, provider, stream, requests == 1)
						}))
						defer server.Close()
						var modelClient llms.Model
						var err error
						if provider == "openai" {
							modelClient, err = openai.New(openai.WithToken("test-key"), openai.WithModel(model), openai.WithBaseURL(server.URL+prefix))
						} else {
							modelClient, err = anthropic.New(anthropic.WithToken("test-key"), anthropic.WithModel(model), anthropic.WithBaseURL(server.URL+prefix))
						}
						require.NoError(t, err)
						options := []llms.CallOption{llms.WithMaxTokens(4096), llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{Name: "lookup", Parameters: map[string]any{"type": "object"}}}})}
						if effort != "" {
							options = append(options, llms.WithReasoningEffort(effort))
						}
						var streamed string
						if stream {
							options = append(options, llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error { streamed += string(chunk); return nil }))
						}
						messages := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "Look it up")}
						first, err := modelClient.GenerateContent(context.Background(), messages, options...)
						require.NoError(t, err)
						require.Len(t, first.Choices, 1)
						require.Len(t, first.Choices[0].ToolCalls, 1)
						messages = append(messages,
							llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{first.Choices[0].ToolCalls[0]}},
							llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{ToolCallID: "call_1", Name: "lookup", Content: "found"}}},
						)
						streamed = "" // Only inspect the final text, after tool-call streaming.
						final, err := modelClient.GenerateContent(context.Background(), messages, options...)
						require.NoError(t, err)
						require.Equal(t, "done", final.Choices[0].Content)
						require.Equal(t, 2, requests)
						if stream {
							require.Equal(t, "done", streamed)
						}
					})
				}
			}
		}
	}
}

func writeEffortResponse(w http.ResponseWriter, provider string, stream, tool bool) {
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	if provider == "openai" {
		if stream {
			delta := `{"content":"done"}`
			if tool {
				delta = `{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}`
			}
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":%s}]}\n\ndata: [DONE]\n\n", delta)
		} else if tool {
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
		} else {
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}`)
		}
		return
	}
	if !stream {
		content := `{"type":"text","text":"done"}`
		if tool {
			content = `{"type":"tool_use","id":"call_1","name":"lookup","input":{}}`
		}
		fmt.Fprintf(w, `{"id":"msg_1","type":"message","role":"assistant","content":[%s],"usage":{"input_tokens":1,"output_tokens":1}}`, content)
		return
	}
	fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":1}}}\n\n")
	if tool {
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"lookup\",\"input\":{}}}\n\n")
	} else {
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n")
	}
	fmt.Fprint(w, "data: {\"type\":\"content_block_stop\",\"index\":0}\n\ndata: {\"type\":\"message_stop\"}\n\n")
}

func TestEffortIndependentOfThinkingBudget(t *testing.T) {
	for _, effort := range []string{"", "max"} {
		t.Run(effort, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				thinking, ok := body["thinking"].(map[string]any)
				if !ok || thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(2048) {
					t.Errorf("thinking = %v", body["thinking"])
				}
				if effort == "" {
					if _, ok := body["output_config"]; ok {
						t.Error("ThinkingMode must not imply effort")
					}
				} else {
					config, ok := body["output_config"].(map[string]any)
					if !ok || config["effort"] != effort {
						t.Errorf("output_config = %v", body["output_config"])
					}
				}
				writeEffortResponse(w, "anthropic", false, false)
			}))
			defer server.Close()
			client, err := anthropic.New(anthropic.WithToken("test-key"), anthropic.WithModel("claude-sonnet-4-5"), anthropic.WithBaseURL(server.URL))
			require.NoError(t, err)
			_, err = client.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hello")}, llms.WithMaxTokens(4096), llms.WithThinkingMode(llms.ThinkingModeMedium), llms.WithReasoningEffort(effort))
			require.NoError(t, err)
		})
	}
}
