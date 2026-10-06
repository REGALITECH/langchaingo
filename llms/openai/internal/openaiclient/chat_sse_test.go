package openaiclient

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestParseStreamingChatResponse_SSEComments(t *testing.T) {
	ctx := context.Background()
	t.Parallel()

	// Test the key SSE comment patterns
	testCases := []struct {
		name            string
		body            string
		expectedContent string
	}{
		{
			name: "openrouter_comments",
			body: `data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
: OPENROUTER PROCESSING
: OPENROUTER PROCESSING
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]`,
			expectedContent: "Hello World",
		},
		{
			name: "comments_without_space",
			body: `data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"Test"},"finish_reason":null}]}
:comment-without-space
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]`,
			expectedContent: "Test",
		},
		{
			name: "other_sse_fields",
			body: `event: message
id: 12345
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"Data"},"finish_reason":null}]}
retry: 1000
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]`,
			expectedContent: "Data",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			r := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(tc.body)),
			}

			req := &ChatRequest{
				StreamingFunc: func(_ context.Context, _ []byte) error {
					return nil
				},
			}

			resp, err := parseStreamingChatResponse(ctx, r, req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp == nil {
				t.Fatal("response should not be nil")
			}
			if len(resp.Choices) == 0 {
				t.Fatal("expected at least one choice")
			}
			if got := resp.Choices[0].Message.Content; got != tc.expectedContent {
				t.Errorf("content mismatch: got %q, want %q", got, tc.expectedContent)
			}
		})
	}
}

func TestParseStreamingChatResponse_ErrorEnvelope(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name          string
		body          string
		wantChunks    []string
		wantErrorText string
	}{
		{
			name: "before_choices",
			body: `data: {"error":{"message":"synthetic upstream failure","type":"server_error","code":"synthetic"}}

data: [DONE]`,
			wantErrorText: "API returned streaming error: synthetic upstream failure",
		},
		{
			name: "after_partial_choice",
			body: `data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

data: {"error":{"message":"synthetic upstream failure","type":"server_error","code":"synthetic"}}

data: [DONE]`,
			wantChunks:    []string{"partial"},
			wantErrorText: "API returned streaming error: synthetic upstream failure",
		},
		{
			name: "normal_completion",
			body: `data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"complete"},"finish_reason":null}]}

data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]`,
			wantChunks: []string{"complete", ""},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var chunks []string
			response, err := parseStreamingChatResponse(context.Background(), &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(tc.body)),
			}, &ChatRequest{StreamingFunc: func(_ context.Context, chunk []byte) error {
				chunks = append(chunks, string(chunk))
				return nil
			}})

			if tc.wantErrorText == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if response == nil || response.Choices[0].Message.Content != "complete" {
					t.Fatalf("normal stream response = %#v, want complete content", response)
				}
			} else {
				if response != nil {
					t.Fatalf("response = %#v, want nil on stream error", response)
				}
				if err == nil || err.Error() != tc.wantErrorText {
					t.Fatalf("error = %v, want %q", err, tc.wantErrorText)
				}
			}
			if got := strings.Join(chunks, "|"); got != strings.Join(tc.wantChunks, "|") {
				t.Fatalf("stream chunks = %#v, want %#v", chunks, tc.wantChunks)
			}
		})
	}
}
