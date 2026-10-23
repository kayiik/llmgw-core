package providers

import (
	"encoding/base64"
	"io"
	"maps"
	"mime"
	"slices"
	"strings"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

func mediaInputLoss(path, detail string) core.Loss {
	return core.Loss{Path: path, Class: translate.LossUnsupported, Severity: translate.LossMaterial, Detail: detail}
}

func mediaInputFields(path string, object map[string]any, known ...string) []core.Loss {
	var losses []core.Loss
	for _, key := range slices.Sorted(maps.Keys(object)) {
		if object[key] != nil && !slices.Contains(known, key) {
			losses = append(losses, mediaInputLoss(path+"."+key, "this media control is not mapped"))
		}
	}
	return losses
}

func validMediaBase64(data string) bool {
	if data == "" {
		return false
	}
	count, err := io.Copy(io.Discard, base64.NewDecoder(base64.StdEncoding.Strict(), strings.NewReader(data)))
	return count > 0 && err == nil
}

// Validate the carrier, not the media codec. Providers own file/model validation;
// no image decoder, transcoder or remote fetch is needed to map these bytes.
func mediaDataURL(value string) (mediaType, data string, ok bool) {
	header, data, found := strings.Cut(value, ",")
	if !found || !strings.HasPrefix(header, "data:") || !strings.HasSuffix(header, ";base64") {
		return "", "", false
	}
	mediaType, params, err := mime.ParseMediaType(strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64"))
	if err != nil || len(params) != 0 || !strings.Contains(mediaType, "/") || strings.Contains(mediaType, "*") || !validMediaBase64(data) {
		return "", "", false
	}
	return mediaType, data, true
}
