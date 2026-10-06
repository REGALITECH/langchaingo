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
		for _, mode := range []llms.ThinkingMode{"", llms.ThinkingModeNone, llms.ThinkingModeHigh} {
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
					if effort, exists := payload["reasoning_effort"]; exists {
						t.Errorf("unexpected effort %v", effort)
					}
					wantTemperature := model == "gpt-4o" || model == "gateway/custom-model" || model == "o1"
					gotTemp, exists := payload["temperature"]
					if exists != wantTemperature || (exists && gotTemp != 0.7) {
						t.Errorf("temperature = %v (present %t), want presence %t", gotTemp, exists, wantTemperature)
					}
				})
			}
		}
	}
}

func TestEmptyReasoningEffortPreservesUpstreamTemperature(t *testing.T) {
	for _, model := range []string{"gpt-5.4", "o1", "gpt-4o"} {
		for _, effort := range []llms.ThinkingEffort{"none", llms.ThinkingEffortHigh} {
			t.Run(fmt.Sprintf("%s/%s", model, effort), func(t *testing.T) {
				requests := make(chan map[string]any, 1)
				client, err := New(WithToken("test-key"), WithModel(model), WithHTTPClient(recordingDoer{requests}))
				if err != nil {
					t.Fatal(err)
				}
				_, err = client.Call(context.Background(), "hello", llms.WithTemperature(0.7),
					llms.WithThinkingEffort(effort), llms.WithReasoningEffort(""))
				if err != nil {
					t.Fatal(err)
				}
				payload := <-requests
				got, present := payload["temperature"]
				// Preserve main's model-based behavior, including the bare o1 name.
				if want := model != "gpt-5.4"; present != want || (present && got != 0.7) {
					t.Errorf("temperature = %v (present %t), want presence %t", got, present, want)
				}
			})
		}
	}
}
