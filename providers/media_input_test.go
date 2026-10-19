package providers

import (
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/translation"
)

func TestPreparedGoogleMediaInputs(t *testing.T) {
	for _, deployment := range []GoogleDeployment{GoogleAIStudio, GoogleVertexAI} {
		for _, stream := range []bool{false, true} {
			for _, surface := range []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages} {
				t.Run(string(deployment)+"/"+string(surface)+"/"+map[bool]string{false: "invoke", true: "stream"}[stream], func(t *testing.T) {
					answer := `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`
					if stream {
						answer = "data: " + answer + "\n\n"
					}
					fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, answer))
					provider := newGoogleTest(t, GoogleConfig{Deployment: deployment, Project: "fixture-project", BaseURL: base})
					body := `{"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":"auto"}},{"type":"text","text":"after"},{"type":"file","file":{"filename":"fixture.pdf","file_data":"data:application/pdf;base64,BAUG"}},{"type":"file","file":{"filename":"fixture.mp4","file_data":"data:video/mp4;base64,BwgJ"}}]}]}`
					if surface == core.ModelSurfaceResponses {
						body = `{"input":[{"role":"user","content":[{"type":"input_text","text":"before"},{"type":"input_image","image_url":"data:image/png;base64,AQID","detail":"auto"},{"type":"input_text","text":"after"},{"type":"input_file","filename":"fixture.pdf","file_data":"data:application/pdf;base64,BAUG"},{"type":"input_file","filename":"fixture.mp4","file_data":"data:video/mp4;base64,BwgJ"}]}]}`
					} else if surface == core.ModelSurfaceMessages {
						body = `{"max_tokens":32,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}},{"type":"text","text":"after"},{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"BAUG"}}]}]}`
					}
					request := googleChatRequest("fixture", body, googleKey("fixture"))
					request.Surface = surface
					mode := core.CallModeInvoke
					if stream {
						mode = core.CallModeStream
					}
					prepared, err := core.PrepareCall(t.Context(), translation.Adapter{Provider: provider}, request, core.CallPreparationOptions{Mode: mode, AllowAdaptation: true})
					if err != nil {
						t.Fatal(err)
					}
					if _, err := core.PlanPreparedCall(prepared.Description(), core.CallPolicy{AllowAdaptation: true}); err != nil {
						t.Fatalf("media not admitted: %v (%+v)", err, prepared.Description().Assessment)
					}
					if len(fake.take()) != 0 {
						t.Fatal("preparation made an upstream request")
					}
					if stream {
						iter, err := prepared.Stream(t.Context(), request.Credential)
						if err != nil {
							t.Fatal(err)
						}
						defer iter.Close()
						for {
							if _, err := iter.Next(); err != nil {
								if err != io.EOF {
									t.Fatal(err)
								}
								break
							}
						}
					} else if _, err := prepared.Invoke(t.Context(), request.Credential); err != nil {
						t.Fatal(err)
					}
					calls := fake.take()
					if len(calls) != 1 {
						t.Fatalf("requests=%d", len(calls))
					}
					var sent map[string]any
					if err := json.Unmarshal([]byte(calls[0].body), &sent); err != nil {
						t.Fatal(err)
					}
					parts := sent["contents"].([]any)[0].(map[string]any)["parts"]
					var want any
					_ = json.Unmarshal([]byte(`[{"text":"before"},{"inlineData":{"mimeType":"image/png","data":"AQID"}},{"text":"after"},{"inlineData":{"mimeType":"application/pdf","data":"BAUG"}},{"inlineData":{"mimeType":"video/mp4","data":"BwgJ"}}]`), &want)
					if surface == core.ModelSurfaceMessages {
						want = want.([]any)[:4]
					}
					if !reflect.DeepEqual(parts, want) {
						t.Fatalf("media/order changed: %s", calls[0].body)
					}
				})
			}
		}
	}
}

func TestGoogleAudioAndRemoteMediaParts(t *testing.T) {
	for _, tc := range []struct{ content, want string }{
		{`{"type":"input_audio","input_audio":{"data":"AQID","format":"wav"}}`, `{"inlineData":{"mimeType":"audio/wav","data":"AQID"}}`},
		{`{"type":"input_audio","input_audio":{"data":"AQID","format":"mp3"}}`, `{"inlineData":{"mimeType":"audio/mpeg","data":"AQID"}}`},
		{`{"type":"image_url","image_url":{"url":"https://example.invalid/fixture.png"}}`, `{"fileData":{"fileUri":"https://example.invalid/fixture.png","mimeType":"image/png"}}`},
		{`{"type":"file","file":{"filename":"fixture.pdf","file_data":"AQID"}}`, `{"inlineData":{"mimeType":"application/pdf","data":"AQID"}}`},
		{`{"type":"file","file":{"file_id":"gs://fixture-bucket/fixture.mp4"}}`, `{"fileData":{"fileUri":"gs://fixture-bucket/fixture.mp4","mimeType":"video/mp4"}}`},
	} {
		body, losses := googleSent(t, GoogleAIStudio, false, `{"messages":[{"role":"user","content":[`+tc.content+`]}]}`)
		var sent map[string]any
		var want any
		_ = json.Unmarshal([]byte(body), &sent)
		_ = json.Unmarshal([]byte(tc.want), &want)
		part := sent["contents"].([]any)[0].(map[string]any)["parts"].([]any)[0]
		if len(losses) != 0 || !reflect.DeepEqual(part, want) {
			t.Errorf("mapping=%s losses=%+v want=%s", body, losses, tc.want)
		}
	}
}

func TestOllamaInlineImagesAcrossTextSurfaces(t *testing.T) {
	for _, surface := range []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages} {
		for _, stream := range []bool{false, true} {
			provider, calls := ollamaDaemon(t, http.StatusOK, `{"message":{"content":"ok"},"done":true}`)
			body := `{"messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,BAUG"}}]}]}`
			if surface == core.ModelSurfaceResponses {
				body = `{"input":[{"role":"user","content":[{"type":"input_text","text":"describe"},{"type":"input_image","image_url":"data:image/png;base64,AQID"},{"type":"input_image","image_url":"data:image/jpeg;base64,BAUG"}]}]}`
			} else if surface == core.ModelSurfaceMessages {
				body = `{"max_tokens":32,"messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"BAUG"}}]}]}`
			}
			request := ollamaChatRequest("fixture", body)
			request.Surface = surface
			adapter := translation.Adapter{Provider: provider}
			mode := core.CallModeInvoke
			if stream {
				mode = core.CallModeStream
			}
			prepared, err := core.PrepareCall(t.Context(), adapter, request, core.CallPreparationOptions{Mode: mode, AllowAdaptation: true})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := core.PlanPreparedCall(prepared.Description(), core.CallPolicy{AllowAdaptation: true}); err != nil {
				t.Fatalf("%s media admission: %v", surface, err)
			}
			if stream {
				iter, err := prepared.Stream(t.Context(), nil)
				if err != nil {
					t.Fatal(err)
				}
				_ = iter.Close()
			} else if _, err := prepared.Invoke(t.Context(), nil); err != nil {
				t.Fatal(err)
			}
			got := calls()
			if len(got) != 1 || !strings.Contains(got[0].body, `"content":"describe","images":["AQID","BAUG"]`) {
				t.Fatalf("%s stream=%t: native images not preserved: %+v", surface, stream, got)
			}
		}
	}
}

func TestPreparedMediaInputsRefuseUnmappedCarriers(t *testing.T) {
	for _, content := range []string{
		`{"type":"image_url","image_url":{"url":"data:,"}}`,
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,not-base64"}}`,
		`{"type":"image_url","image_url":{"url":"data:application/pdf;base64,AQID"}}`,
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":"high"}}`,
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","unmapped_control":true}}`,
		`{"type":"input_audio","input_audio":{"data":"AQID"}}`,
		`{"type":"input_audio","input_audio":{"data":"not-base64","format":"wav"}}`,
		`{"type":"file","file":{"file_id":"file_fixture"}}`,
		`{"type":"file","file":{"file_data":4,"file_id":"gs://fixture-bucket/fixture.pdf"}}`,
		`{"type":"file","file":{"filename":"fixture.pdf","file_data":"AQID","unmapped_control":true}}`,
	} {
		fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
		google := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
		ollama, calls := ollamaDaemon(t, http.StatusOK, `{"message":{"content":"ok"},"done":true}`)
		for _, provider := range []core.Provider{google, ollama} {
			for _, mode := range []core.CallMode{core.CallModeInvoke, core.CallModeStream} {
				request := googleChatRequest("fixture", `{"messages":[{"role":"user","content":[`+content+`]}]}`, googleKey("fixture"))
				prepared, err := core.PrepareCall(t.Context(), provider, request, core.CallPreparationOptions{Mode: mode, AllowAdaptation: true})
				if err == nil {
					_, err = core.PlanPreparedCall(prepared.Description(), core.CallPolicy{AllowAdaptation: true})
				}
				if err == nil {
					t.Fatalf("%T silently admitted unsupported media: %s", provider, content)
				}
			}
		}
		if len(fake.take()) != 0 || len(calls()) != 0 {
			t.Fatal("refused media reached inference")
		}
	}
}
