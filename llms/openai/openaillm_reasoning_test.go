package openai

import (
	"bytes"
	"context"
	"encoding/json"
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
