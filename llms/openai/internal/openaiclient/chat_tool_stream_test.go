package openaiclient

import (
	"encoding/json"
	"testing"
)

func decodeToolDeltas(t *testing.T, chunks ...string) []ToolCall {
	t.Helper()
	var tools []ToolCall
	for _, chunk := range chunks {
		var delta []*ToolCall
		if err := json.Unmarshal([]byte(chunk), &delta); err != nil {
			t.Fatalf("decode delta %s: %v", chunk, err)
		}
		_, tools = updateToolCalls(tools, delta)
	}
	return tools
}

func TestUpdateToolCallsMergesContinuationsByIndex(t *testing.T) {
	t.Parallel()
	tests := map[string]struct {
		chunks []string
		want   []ToolCall
	}{
		"gateway repeats type on argument fragments": {
			chunks: []string{
				`[{"index":0,"id":"toolu_1","type":"function","function":{"name":"get_weather","arguments":""}}]`,
				`[{"index":0,"id":"","type":"function","function":{"name":"","arguments":"{\"city\""}}]`,
				`[{"index":0,"id":"","type":"function","function":{"name":"","arguments":": \"Tokyo\"}"}}]`,
			},
			want: []ToolCall{{ID: "toolu_1", Type: "function", Function: ToolFunction{Name: "get_weather", Arguments: `{"city": "Tokyo"}`}}},
		},
		"interleaved parallel calls": {
			chunks: []string{
				`[{"index":0,"id":"call_a","type":"function","function":{"name":"a","arguments":""}}]`,
				`[{"index":1,"id":"call_b","type":"function","function":{"name":"b","arguments":""}}]`,
				`[{"index":0,"function":{"arguments":"{\"x\":1}"}}]`,
				`[{"index":1,"function":{"arguments":"{\"y\":2}"}}]`,
			},
			want: []ToolCall{
				{ID: "call_a", Type: "function", Function: ToolFunction{Name: "a", Arguments: `{"x":1}`}},
				{ID: "call_b", Type: "function", Function: ToolFunction{Name: "b", Arguments: `{"y":2}`}},
			},
		},
		"complete calls sharing an index stay separate": {
			chunks: []string{
				`[{"index":0,"id":"call_a","type":"function","function":{"name":"a","arguments":"{}"}}]`,
				`[{"index":0,"id":"call_b","type":"function","function":{"name":"b","arguments":"{}"}}]`,
			},
			want: []ToolCall{
				{ID: "call_a", Type: "function", Function: ToolFunction{Name: "a", Arguments: "{}"}},
				{ID: "call_b", Type: "function", Function: ToolFunction{Name: "b", Arguments: "{}"}},
			},
		},
		"fragments without index continue the last call": {
			chunks: []string{
				`[{"id":"call_a","type":"function","function":{"name":"a","arguments":"{\"q\""}}]`,
				`[{"function":{"arguments":":1}"}}]`,
			},
			want: []ToolCall{{ID: "call_a", Type: "function", Function: ToolFunction{Name: "a", Arguments: `{"q":1}`}}},
		},
		"canonical shape with type only on first chunk and many fragments": {
			chunks: []string{
				`[{"index":0,"id":"call_a","type":"function","function":{"name":"a","arguments":""}}]`,
				`[{"index":0,"function":{"arguments":"{\"k"}}]`,
				`[{"index":0,"function":{"arguments":"ey\":"}}]`,
				`[{"index":0,"function":{"arguments":"\"v\"}"}}]`,
			},
			want: []ToolCall{{ID: "call_a", Type: "function", Function: ToolFunction{Name: "a", Arguments: `{"key":"v"}`}}},
		},
		"router shape with interleaved parallel calls": {
			chunks: []string{
				`[{"index":0,"id":"t1","type":"function","function":{"name":"a","arguments":""}}]`,
				`[{"index":1,"id":"t2","type":"function","function":{"name":"b","arguments":""}}]`,
				`[{"index":1,"id":"","type":"function","function":{"name":"","arguments":"{\"y\":"}}]`,
				`[{"index":0,"id":"","type":"function","function":{"name":"","arguments":"{\"x\":1}"}}]`,
				`[{"index":1,"id":"","type":"function","function":{"name":"","arguments":"2}"}}]`,
			},
			want: []ToolCall{
				{ID: "t1", Type: "function", Function: ToolFunction{Name: "a", Arguments: `{"x":1}`}},
				{ID: "t2", Type: "function", Function: ToolFunction{Name: "b", Arguments: `{"y":2}`}},
			},
		},
		"orphan fragment without any call is dropped": {
			chunks: []string{
				`[{"index":0,"type":"function","function":{"arguments":"{}"}}]`,
			},
			want: nil,
		},
		"continuation by id without index": {
			chunks: []string{
				`[{"id":"call_a","type":"function","function":{"name":"a","arguments":"{"}}]`,
				`[{"id":"call_a","type":"function","function":{"arguments":"}"}}]`,
			},
			want: []ToolCall{{ID: "call_a", Type: "function", Function: ToolFunction{Name: "a", Arguments: "{}"}}},
		},
		"first chunk without index then fragments with index": {
			chunks: []string{
				`[{"id":"call_a","type":"function","function":{"name":"a","arguments":"{\"q\""}}]`,
				`[{"index":0,"function":{"arguments":":1}"}}]`,
			},
			want: []ToolCall{{ID: "call_a", Type: "function", Function: ToolFunction{Name: "a", Arguments: `{"q":1}`}}},
		},
		"name repeated on every fragment without id": {
			chunks: []string{
				`[{"index":0,"id":"t1","type":"function","function":{"name":"a","arguments":""}}]`,
				`[{"index":0,"type":"function","function":{"name":"a","arguments":"{\"x\":"}}]`,
				`[{"index":0,"type":"function","function":{"name":"a","arguments":"1}"}}]`,
			},
			want: []ToolCall{{ID: "t1", Type: "function", Function: ToolFunction{Name: "a", Arguments: `{"x":1}`}}},
		},
		"different name at a new index starts a new call": {
			chunks: []string{
				`[{"index":0,"id":"t1","type":"function","function":{"name":"a","arguments":"{}"}}]`,
				`[{"index":1,"type":"function","function":{"name":"b","arguments":"{}"}}]`,
			},
			want: []ToolCall{
				{ID: "t1", Type: "function", Function: ToolFunction{Name: "a", Arguments: "{}"}},
				{Type: "function", Function: ToolFunction{Name: "b", Arguments: "{}"}},
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			got := decodeToolDeltas(t, tc.chunks...)
			if len(got) != len(tc.want) {
				t.Fatalf("got %d tool calls, want %d: %+v", len(got), len(tc.want), got)
			}
			for i := range tc.want {
				if got[i].ID != tc.want[i].ID || got[i].Type != tc.want[i].Type || got[i].Function != tc.want[i].Function {
					t.Errorf("tool call %d = %+v, want %+v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestUpdateToolCallsReturnsRawDeltaForStreamingCallback(t *testing.T) {
	t.Parallel()
	chunks := []string{
		`[{"index":0,"id":"t1","type":"function","function":{"name":"a","arguments":""}}]`,
		`[{"index":0,"id":"","type":"function","function":{"name":"","arguments":"{\"x\":1}"}}]`,
	}
	var tools []ToolCall
	for _, c := range chunks {
		var delta []*ToolCall
		if err := json.Unmarshal([]byte(c), &delta); err != nil {
			t.Fatal(err)
		}
		want, err := json.Marshal(delta)
		if err != nil {
			t.Fatal(err)
		}
		var got []byte
		got, tools = updateToolCalls(tools, delta)
		if string(got) != string(want) {
			t.Errorf("callback chunk = %s, want raw delta %s", got, want)
		}
	}
}
