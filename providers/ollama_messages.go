package providers

import (
	"strconv"
	"strings"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// ollamaMessages normalizes Chat messages as the gateway does for Ollama and
// reports what that loses at each message's path. A developer message is
// sent as a system one, and a message without a role as a user one.
// Strings and ordered text parts preserve their text. Unsupported content
// produces material findings checked before inference. A message with
// no content, tool calls or tool call ID is dropped, and of the others only
// the role, content, tool_calls, tool_call_id and name are sent.
func ollamaMessages(messages []map[string]any) ([]map[string]any, []core.Loss) {
	out := []map[string]any{}
	var losses []core.Loss
	for index, message := range messages {
		path := "messages." + strconv.Itoa(index)
		role, _ := message["role"].(string)
		if role == "" {
			role = "user"
		}
		content, images, contentLosses := ollamaContent(path+".content", message["content"], role == "user")
		losses = append(losses, contentLosses...)
		toolCalls := message["tool_calls"]
		toolCallID, _ := message["tool_call_id"].(string)
		if message["content"] == nil && toolCalls == nil && toolCallID == "" {
			losses = append(losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Ollama does not send a message without content, tool calls or a tool call ID"})
			continue
		}
		if role == "developer" {
			role = "system"
			losses = append(losses, core.Loss{Path: path + ".role", Class: translate.LossApproximated, Severity: translate.LossAdvisory, Detail: "Ollama receives a developer message as a system message"})
		}
		sent := map[string]any{"role": role, "content": content}
		if len(images) > 0 {
			sent["images"] = images
		}
		if toolCalls != nil {
			sent["tool_calls"] = toolCalls
		}
		if toolCallID != "" {
			sent["tool_call_id"] = toolCallID
		}
		if name, ok := message["name"].(string); ok && name != "" {
			sent["name"] = name
		}
		for field, value := range message {
			switch field {
			case "role", "content", "tool_calls", "tool_call_id", "name":
				continue
			}
			if value != nil {
				losses = append(losses, core.Loss{Path: path + "." + field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Ollama does not send this message field"})
			}
		}
		out = append(out, sent)
	}
	return out, losses
}

func ollamaContent(path string, content any, allowImages bool) (string, []string, []core.Loss) {
	parts, multipart := content.([]any)
	if !multipart || !allowImages {
		text, losses := chatText(path, content, "Ollama cannot map this content")
		return text, nil, losses
	}
	var text strings.Builder
	var images []string
	var losses []core.Loss
	for index, raw := range parts {
		partPath := path + "." + strconv.Itoa(index)
		part, _ := raw.(map[string]any)
		if part["type"] != "image_url" {
			value, findings := chatText(path, []any{raw}, "Ollama cannot map this content")
			for _, loss := range findings {
				loss.Path = partPath + strings.TrimPrefix(loss.Path, path+".0")
				losses = append(losses, loss)
			}
			text.WriteString(value)
			continue
		}
		image, _ := part["image_url"].(map[string]any)
		losses = append(losses, mediaInputFields(partPath, part, "type", "image_url")...)
		losses = append(losses, mediaInputFields(partPath+".image_url", image, "url", "detail")...)
		if detail := image["detail"]; detail != nil && detail != "auto" {
			losses = append(losses, mediaInputLoss(partPath+".image_url.detail", "Ollama does not map image detail"))
		}
		reference, _ := image["url"].(string)
		mediaType, data, valid := mediaDataURL(reference)
		if !valid || !strings.HasPrefix(mediaType, "image/") {
			losses = append(losses, mediaInputLoss(partPath, "Ollama images require an inline base64 image; remote URL fetching is not performed"))
			continue
		}
		images = append(images, data)
	}
	return text.String(), images, losses
}
