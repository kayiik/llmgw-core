package sourceusage

import (
	"io"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

type nativeFrames struct{ frames []string }

func (s *nativeFrames) Next() ([]byte, error) {
	if len(s.frames) == 0 {
		return nil, io.EOF
	}
	frame := s.frames[0]
	s.frames = s.frames[1:]
	return []byte(frame), nil
}
func (*nativeFrames) Close() error { return nil }

func TestNativeStreamUsageLifecyclePreservesEvidence(t *testing.T) {
	for _, native := range []struct {
		api             core.UpstreamAPI
		dialect         translate.UsageDialect
		usage, terminal string
	}{
		{core.UpstreamAPIChatCompletions, translate.UsageDialectChat, `{"usage":{"prompt_tokens":11,"completion_tokens":7}}`, `data: [DONE]`},
		{core.UpstreamAPIResponses, translate.UsageDialectResponses, `{"type":"response.in_progress","response":{"usage":{"input_tokens":11,"output_tokens":7}}}`, `data: {"type":"response.completed"}`},
		{core.UpstreamAPIMessages, translate.UsageDialectMessages, `{"type":"message_delta","usage":{"input_tokens":11,"output_tokens":7}}`, `data: {"type":"message_stop"}`},
	} {
		for _, result := range []struct {
			name    string
			tail    []string
			quality translate.CompletionQuality
		}{
			{"keepalive-complete", []string{": keepalive\n\n", native.terminal}, translate.CompletionComplete},
			{"malformed-then-terminal", []string{"data: {broken}\n\n", native.terminal}, translate.CompletionFailed},
			{"native-error", []string{`data: {"type":"error","error":{"message":"failed"}}`}, translate.CompletionFailed},
			{"missing-terminal", nil, translate.CompletionTruncated},
		} {
			t.Run(string(native.api)+"/"+result.name, func(t *testing.T) {
				record := core.BeginSourceUsage(t.Context(), native.api, native.dialect)
				frames := append([]string{"data: " + native.usage + "\n\n"}, result.tail...)
				stream := Stream(t.Context(), &nativeFrames{frames: frames}, native.api, record)
				for {
					_, err := stream.Next()
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Fatal(err)
					}
				}
				if err := stream.Close(); err != nil {
					t.Fatal(err)
				}
				observation, ok := core.SourceUsageOf(stream)
				if !ok || observation.Usage.Quality != result.quality || observation.Usage.OutputTokens.Value != 7 || observation.Usage.OutputTokens.Presence != translate.CountPresent {
					t.Fatalf("native evidence/lifecycle lost: %+v", observation)
				}
			})
		}
	}
}
