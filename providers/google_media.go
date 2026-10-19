package providers

import (
	"net/url"
	"path"
	"strconv"
	"strings"

	core "github.com/kayiik/llmgw-core"
)

func googleChatContent(path string, content any) ([]map[string]any, []core.Loss) {
	items, multipart := content.([]any)
	if !multipart {
		text, losses := chatText(path, content, "Google cannot represent this content")
		return []map[string]any{{"text": text}}, losses
	}
	var parts []map[string]any
	var losses []core.Loss
	var text strings.Builder
	textSeen := false
	flush := func() {
		if textSeen {
			parts = append(parts, map[string]any{"text": text.String()})
			text.Reset()
			textSeen = false
		}
	}
	for index, raw := range items {
		partPath := path + "." + strconv.Itoa(index)
		part, _ := raw.(map[string]any)
		if part["type"] == "text" {
			value, valid := part["text"].(string)
			if !valid {
				losses = append(losses, mediaInputLoss(partPath, "text content must be a string"))
				continue
			}
			losses = append(losses, mediaInputFields(partPath, part, "type", "text")...)
			text.WriteString(value)
			textSeen = true
			continue
		}
		flush()
		mapped, findings := googleMediaPart(partPath, part)
		losses = append(losses, findings...)
		if mapped != nil {
			parts = append(parts, mapped)
		}
	}
	flush()
	if len(parts) == 0 {
		parts = []map[string]any{{"text": ""}}
	}
	return parts, losses
}

func googleMediaPart(path string, part map[string]any) (map[string]any, []core.Loss) {
	var reference, filename, expected string
	var losses []core.Loss
	switch part["type"] {
	case "image_url":
		losses = mediaInputFields(path, part, "type", "image_url")
		image, ok := part["image_url"].(map[string]any)
		if !ok {
			return nil, append(losses, mediaInputLoss(path, "image_url must contain a URL object"))
		}
		losses = append(losses, mediaInputFields(path+".image_url", image, "url", "detail")...)
		if detail := image["detail"]; detail != nil && detail != "auto" {
			losses = append(losses, mediaInputLoss(path+".image_url.detail", "non-default image detail needs a model-specific resolution mapping"))
		}
		reference, _ = image["url"].(string)
		expected = "image/"
	case "input_audio":
		losses = mediaInputFields(path, part, "type", "input_audio")
		audio, _ := part["input_audio"].(map[string]any)
		losses = append(losses, mediaInputFields(path+".input_audio", audio, "data", "format")...)
		data, _ := audio["data"].(string)
		format, _ := audio["format"].(string)
		mediaType := map[string]string{"wav": "audio/wav", "mp3": "audio/mpeg"}[format]
		if mediaType == "" || !validMediaBase64(data) {
			return nil, append(losses, mediaInputLoss(path, "input_audio requires base64 data and an explicit wav or mp3 format"))
		}
		return map[string]any{"inlineData": map[string]any{"mimeType": mediaType, "data": data}}, losses
	case "file":
		losses = mediaInputFields(path, part, "type", "file")
		file, _ := part["file"].(map[string]any)
		losses = append(losses, mediaInputFields(path+".file", file, "filename", "file_data", "file_id")...)
		filename, _ = file["filename"].(string)
		dataValue, dataSet := file["file_data"]
		idValue, idSet := file["file_id"]
		data, dataValid := dataValue.(string)
		id, idValid := idValue.(string)
		if dataSet == idSet || (dataSet && !dataValid) || (idSet && !idValid) || (file["filename"] != nil && filename == "") {
			return nil, append(losses, mediaInputLoss(path, "file needs exactly one data carrier or native URI and a valid optional filename"))
		}
		reference = data
		if idSet {
			// Opaque file IDs belong to their original provider. Only an actual
			// Google-readable URI can be forwarded as fileData, never an ID guess.
			if !strings.HasPrefix(id, "gs://") && !strings.HasPrefix(id, "https://") {
				return nil, append(losses, mediaInputLoss(path+".file.file_id", "a provider file ID is not a Google file URI"))
			}
			reference = id
		}
	default:
		return nil, []core.Loss{mediaInputLoss(path, "Google cannot map this content-part type")}
	}
	mapped, ok := googleMediaReference(reference, filename, expected)
	if !ok {
		losses = append(losses, mediaInputLoss(path, "media requires valid base64 with a concrete MIME type or a native media URI"))
	}
	return mapped, losses
}

func googleMediaReference(reference, filename, expected string) (map[string]any, bool) {
	if strings.HasPrefix(reference, "data:") {
		mediaType, data, valid := mediaDataURL(reference)
		if !valid || !strings.HasPrefix(mediaType, expected) {
			return nil, false
		}
		return map[string]any{"inlineData": map[string]any{"mimeType": mediaType, "data": data}}, true
	}
	uri, err := url.Parse(reference)
	if err == nil && (uri.Scheme == "https" || uri.Scheme == "http" || uri.Scheme == "gs") && uri.Host != "" && uri.User == nil {
		file := map[string]any{"fileUri": reference}
		if mediaType := googleMediaMIME(uri.Path); mediaType != "" {
			if !strings.HasPrefix(mediaType, expected) {
				return nil, false
			}
			file["mimeType"] = mediaType
		}
		return map[string]any{"fileData": file}, true
	}
	if expected == "" && filename != "" && validMediaBase64(reference) {
		if mediaType := googleMediaMIME(filename); mediaType != "" {
			return map[string]any{"inlineData": map[string]any{"mimeType": mediaType, "data": reference}}, true
		}
	}
	return nil, false
}

func googleMediaMIME(filename string) string {
	return map[string]string{
		".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
		".pdf": "application/pdf", ".txt": "text/plain", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
		".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".mpeg": "video/mpeg", ".mpg": "video/mpeg",
	}[strings.ToLower(path.Ext(filename))]
}
