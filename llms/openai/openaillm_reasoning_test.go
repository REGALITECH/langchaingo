package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/tmc/langchaingo/llms"
)

type recordingDoer struct {
	requests chan map[string]any
}

func (d recordingDoer) Do(request *http.Request) (*http.Response, error) {
	var payload map[string]any
	if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
		return nil, err
	}
	d.requests <- payload

	return &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(bytes.NewBufferString(
			`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`,
		)),
		Header: make(http.Header),
	}, nil
}

func TestGPT54ReasoningEffortCapabilities(t *testing.T) {
	caps := getModelCapabilities("gpt-5.4")
	if !caps.SupportsThinking {
		t.Fatal("gpt-5.4 does not support thinking")
	}

	for _, effort := range []string{"none", "low", "medium", "high", "xhigh"} {
		if !supportsReasoningEffort(caps, effort) {
			t.Errorf("gpt-5.4 does not support reasoning effort %q", effort)
		}
	}
	if supportsReasoningEffort(caps, "minimal") {
		t.Error("gpt-5.4 supports unexpected reasoning effort minimal")
	}

	if supportsReasoningEffort(getModelCapabilities("gpt-5.3"), "high") {
		t.Error("gpt-5.3 unexpectedly supports reasoning_effort")
	}
}

func TestGPT54ReasoningEffortRequest(t *testing.T) {
	requests := make(chan map[string]any, 2)

	llm, err := New(WithToken("test-key"), WithModel("gpt-5.4"), WithHTTPClient(recordingDoer{requests}))
	if err != nil {
		t.Fatalf("new LLM: %v", err)
	}

	for _, test := range []struct {
		name            string
		mode            llms.ThinkingMode
		wantTemperature bool
	}{
		{name: "reasoning enabled", mode: llms.ThinkingModeHigh},
		{name: "reasoning disabled", mode: llms.ThinkingModeNone, wantTemperature: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := llm.Call(context.Background(), "hello", llms.WithThinkingMode(test.mode), llms.WithTemperature(0))
			if err != nil {
				t.Fatalf("call LLM: %v", err)
			}

			request := <-requests
			if got := request["reasoning_effort"]; got != string(test.mode) {
				t.Errorf("reasoning_effort = %#v, want %q", got, test.mode)
			}
			_, gotTemperature := request["temperature"]
			if gotTemperature != test.wantTemperature {
				t.Errorf("temperature present = %v, want %v", gotTemperature, test.wantTemperature)
			}
		})
	}
}

func TestExplicitEffortOverridesThinkingMode(t *testing.T) {
	for _, model := range []string{"gpt-5.4", "gpt-5.3", "o1", "gpt-4o", "gateway/custom-model"} {
		for _, effort := range []string{"none", "high", "xhigh", "future-value"} {
			for _, temperature := range []float64{0, 0.7} {
				for _, explicitFirst := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%s/temp=%g/explicitFirst=%t", model, effort, temperature, explicitFirst), func(t *testing.T) {
						requests := make(chan map[string]any, 1)
						client, err := New(WithToken("test-key"), WithModel(model), WithHTTPClient(recordingDoer{requests}))
						if err != nil {
							t.Fatal(err)
						}
						options := []llms.CallOption{llms.WithTemperature(temperature)}
						// Use a conflicting legacy mode to prove precedence is order independent.
						mode := llms.ThinkingModeNone
						if effort == "none" {
							mode = llms.ThinkingModeHigh
						}
						if explicitFirst {
							options = append(options, llms.WithReasoningEffort(effort), llms.WithThinkingMode(mode))
						} else {
							options = append(options, llms.WithThinkingMode(mode), llms.WithReasoningEffort(effort))
						}
						if _, err := client.Call(context.Background(), "hello", options...); err != nil {
							t.Fatal(err)
						}
						payload := <-requests
						if payload["reasoning_effort"] != effort {
							t.Errorf("effort = %v, want %s", payload["reasoning_effort"], effort)
						}
						got, exists := payload["temperature"]
						if effort == "none" {
							if !exists || got != temperature {
								t.Errorf("temperature = %v (present %t), want %g", got, exists, temperature)
							}
						} else if exists {
							t.Errorf("temperature must be omitted, got %v", got)
						}
					})
				}
			}
		}
	}
}

func TestUnspecifiedEffortPreservesLegacyThinking(t *testing.T) {
	for _, model := range []string{"gpt-5.4", "gpt-5.5", "gpt-5.3", "o1", "gpt-4o", "gateway/custom-model"} {
		for _, mode := range []llms.ThinkingMode{"", llms.ThinkingModeNone, llms.ThinkingModeHigh, llms.ThinkingModeXHigh} {
			for _, emptyOption := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/emptyOption=%t", model, mode, emptyOption), func(t *testing.T) {
					requests := make(chan map[string]any, 1)
					client, err := New(WithToken("test-key"), WithModel(model), WithHTTPClient(recordingDoer{requests}))
					if err != nil {
						t.Fatal(err)
					}
					options := []llms.CallOption{llms.WithTemperature(0.7)}
					if mode != "" {
						options = append(options, llms.WithThinkingMode(mode))
					}
					if emptyOption {
						options = append(options, llms.WithReasoningEffort(""))
					}
					if _, err := client.Call(context.Background(), "hello", options...); err != nil {
						t.Fatal(err)
					}
					payload := <-requests
					wantEffort := ""
					if model == "gpt-5.4" || model == "gpt-5.5" {
						wantEffort = string(mode)
					}
					gotEffort, exists := payload["reasoning_effort"]
					if wantEffort == "" {
						if exists {
							t.Errorf("unexpected effort %v", gotEffort)
						}
					} else if gotEffort != wantEffort {
						t.Errorf("effort = %v, want %s", gotEffort, wantEffort)
					}
					wantTemperature := model == "gpt-4o" || model == "gateway/custom-model" || wantEffort == "none"
					gotTemp, exists := payload["temperature"]
					if exists != wantTemperature || (exists && gotTemp != 0.7) {
						t.Errorf("temperature = %v (present %t), want presence %t", gotTemp, exists, wantTemperature)
					}
				})
			}
		}
	}
}
