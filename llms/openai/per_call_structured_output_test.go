package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
)

func TestResponseFormatFromCallOptions_PreservesRawJSONSchema(t *testing.T) {
	t.Parallel()
	rawSchema := map[string]any{
		"$defs": map[string]any{
			"answer": map[string]any{"oneOf": []any{map[string]any{"type": "string"}}},
		},
		"$ref": "#/$defs/answer",
	}
	var opts llms.CallOptions
	llms.WithJSONSchemaResponseFormat(&llms.JSONSchemaResponseFormat{
		Name: "answer", Schema: rawSchema, Strict: true,
	})(&opts)

	responseFormat := responseFormatFromCallOptions(opts)
	require.NotNil(t, responseFormat)
	payload, err := json.Marshal(responseFormat)
	require.NoError(t, err)
	var got map[string]any
	require.NoError(t, json.Unmarshal(payload, &got))
	assert.Equal(t, map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name": "answer", "strict": true, "schema": rawSchema,
		},
	}, got)
}

func TestResponseFormatFromCallOptions_AbsentLeavesRequestUnchanged(t *testing.T) {
	t.Parallel()
	assert.Nil(t, responseFormatFromCallOptions(llms.CallOptions{}))
}
