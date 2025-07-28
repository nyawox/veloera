package aws_converse

import (
	"encoding/json"
	"fmt"
	"strings"
	"veloera/common"
	"veloera/dto"
	relaycommon "veloera/relay/common"
	"veloera/relay/helper"
	"veloera/service"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
)

func newAwsClient(info *relaycommon.RelayInfo) (*bedrockruntime.Client, error) {
	awsSecret := strings.Split(info.ApiKey, "|")
	if len(awsSecret) != 3 {
		return nil, errors.New("invalid aws secret key")
	}
	ak := awsSecret[0]
	sk := awsSecret[1]
	region := awsSecret[2]

	options := bedrockruntime.Options{
		Region:      region,
		Credentials: aws.NewCredentialsCache(credentials.NewStaticCredentialsProvider(ak, sk, "")),
	}

	if info.BaseUrl != "" {
		baseEndpoint := info.BaseUrl
		options.BaseEndpoint = &baseEndpoint
	}

	client := bedrockruntime.New(options)
	return client, nil
}

func wrapErr(err error) *dto.OpenAIErrorWithStatusCode {
	return &dto.OpenAIErrorWithStatusCode{
		StatusCode: 500,
		Error: dto.OpenAIError{
			Message: err.Error(),
		},
	}
}

func stopReasonConverse2OpenAI(reason string) string {
	switch reason {
	case "end_turn", "stop_sequence":
		return "stop"
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	case "content_filtered", "guardrail_intervened":
		return "content_filter"
	default:
		return reason
	}
}

type ConverseRequest struct {
	Messages                     []types.Message               `json:"messages"`
	System                       []types.SystemContentBlock    `json:"system,omitempty"`
	InferenceConfig              *types.InferenceConfiguration `json:"inference_config,omitempty"`
	ToolConfig                   *types.ToolConfiguration      `json:"tool_config,omitempty"`
	AdditionalModelRequestFields document.Interface            `json:"additional_model_request_fields,omitempty"`
}

func (a *Adaptor) RequestOpenAI2ConverseRequest(request *dto.GeneralOpenAIRequest) (*ConverseRequest, error) {
	var system []types.SystemContentBlock
	messages := make([]types.Message, 0, len(request.Messages))

	for _, msg := range request.Messages {
		switch msg.Role {
		case "system":
			system = []types.SystemContentBlock{&types.SystemContentBlockMemberText{Value: msg.StringContent()}}
		case "user":
			content := []types.ContentBlock{&types.ContentBlockMemberText{Value: msg.StringContent()}}
			messages = append(messages, types.Message{Role: types.ConversationRoleUser, Content: content})
		case "assistant":
			var content []types.ContentBlock
			if msg.ToolCalls != nil {
				for _, toolCall := range msg.ParseToolCalls() {
					inputObj := make(map[string]any)
					if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &inputObj); err != nil {
						common.SysError("tool call function arguments is not a map[string]any: " + fmt.Sprintf("%v", toolCall.Function.Arguments))
						continue
					}
					content = append(content, &types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: &toolCall.ID,
							Name:      &toolCall.Function.Name,
							Input:     document.NewLazyDocument(inputObj),
						},
					})
				}
			}
			if msg.StringContent() != "" {
				content = append(content, &types.ContentBlockMemberText{Value: msg.StringContent()})
			}
			if msg.ReasoningContent != "" {
				reasoningBlock := &types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{
						Value: types.ReasoningTextBlock{
							Text: &msg.ReasoningContent,
						},
					},
				}
				content = append(content, reasoningBlock)
			}
			if len(content) > 0 {
				messages = append(messages, types.Message{Role: types.ConversationRoleAssistant, Content: content})
			}
		case "tool":
			var toolResultContent types.ToolResultContentBlock
			var toolInput map[string]interface{}
			if err := json.Unmarshal([]byte(msg.StringContent()), &toolInput); err == nil {
				toolResultContent = &types.ToolResultContentBlockMemberJson{
					Value: document.NewLazyDocument(toolInput),
				}
			} else {
				toolResultContent = &types.ToolResultContentBlockMemberText{Value: msg.StringContent()}
			}

			toolResult := &types.ContentBlockMemberToolResult{
				Value: types.ToolResultBlock{
					ToolUseId: &msg.ToolCallId,
					Content:   []types.ToolResultContentBlock{toolResultContent},
				},
			}

			if len(messages) > 0 && messages[len(messages)-1].Role == types.ConversationRoleUser {
				messages[len(messages)-1].Content = append(messages[len(messages)-1].Content, toolResult)
			} else {
				messages = append(messages, types.Message{
					Role:    types.ConversationRoleUser,
					Content: []types.ContentBlock{toolResult},
				})
			}
		}
	}

	var toolConfig *types.ToolConfiguration
	if len(request.Tools) > 0 {
		tools := make([]types.Tool, len(request.Tools))
		for i, tool := range request.Tools {
			tools[i] = &types.ToolMemberToolSpec{
				Value: types.ToolSpecification{
					Name:        &tool.Function.Name,
					Description: &tool.Function.Description,
					InputSchema: &types.ToolInputSchemaMemberJson{
						Value: document.NewLazyDocument(tool.Function.Parameters),
					},
				},
			}
		}

		var toolChoice types.ToolChoice = &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
		if request.ToolChoice != nil {
			switch choice := request.ToolChoice.(type) {
			case string:
				switch choice {
				case "any", "required":
					toolChoice = &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}
				}
			case map[string]interface{}:
				if typeVal, ok := choice["type"].(string); ok && typeVal == "function" {
					if function, ok := choice["function"].(map[string]interface{}); ok {
						if name, ok := function["name"].(string); ok {
							toolChoice = &types.ToolChoiceMemberTool{
								Value: types.SpecificToolChoice{Name: &name},
							}
						}
					}
				}
			}
		}

		toolConfig = &types.ToolConfiguration{
			Tools:      tools,
			ToolChoice: toolChoice,
		}
	}

	var reasoningConfig document.Interface
	if request.ReasoningEffort != "" {
		budgetTokens := 2048
		switch request.ReasoningEffort {
		case "low":
			budgetTokens = 1024
		case "high":
			budgetTokens = 4096
		}
		if budgetTokens < 1024 {
			budgetTokens = 1024
		}
		reasoningConfig = document.NewLazyDocument(map[string]interface{}{
			"thinking": map[string]interface{}{
				"type":          "enabled",
				"budget_tokens": budgetTokens,
			},
		})
	}

	return &ConverseRequest{
		Messages:                     messages,
		System:                       system,
		ToolConfig:                   toolConfig,
		AdditionalModelRequestFields: reasoningConfig,
	}, nil
}

type ConverseResponseInfo struct {
	ResponseId         string
	Created            int64
	Model              string
	ResponseText       strings.Builder
	ReasoningText      strings.Builder
	Usage              *dto.Usage
	ContentIndex       int
	ToolCalls          []dto.ToolCallResponse
	inProgressToolCall *dto.ToolCallResponse
}

func converseStreamHandler(c *gin.Context, stream *bedrockruntime.ConverseStreamOutput, info *relaycommon.RelayInfo) (*dto.OpenAIErrorWithStatusCode, *dto.Usage) {
	converseInfo := &ConverseResponseInfo{
		ResponseId:    fmt.Sprintf("chatcmpl-%s", common.GetUUID()),
		Created:       common.GetTimestamp(),
		Model:         info.UpstreamModelName,
		ResponseText:  strings.Builder{},
		ReasoningText: strings.Builder{},
		Usage:         &dto.Usage{},
		ContentIndex:  0,
	}

	helper.SetEventStreamHeaders(c)

	for event := range stream.GetStream().Events() {
		err := handleConverseStreamEvent(c, converseInfo, event)
		if err != nil {
			return err, nil
		}
	}

	handleStreamFinalResponse(c, info, converseInfo)
	return nil, converseInfo.Usage
}

func handleConverseStreamEvent(c *gin.Context, converseInfo *ConverseResponseInfo, event interface{}) *dto.OpenAIErrorWithStatusCode {
	var response *dto.ChatCompletionsStreamResponse

	switch e := event.(type) {
	case *types.ConverseStreamOutputMemberMetadata:
		if e.Value.Usage != nil {
			converseInfo.Usage.PromptTokens = int(*e.Value.Usage.InputTokens)
			converseInfo.Usage.CompletionTokens = int(*e.Value.Usage.OutputTokens)
			converseInfo.Usage.TotalTokens = int(*e.Value.Usage.TotalTokens)
		}
	case *types.ConverseStreamOutputMemberMessageStart:
	case *types.ConverseStreamOutputMemberContentBlockStart:
		if e.Value.ContentBlockIndex != nil {
			converseInfo.ContentIndex = int(*e.Value.ContentBlockIndex)
		}
		if start, ok := e.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
			toolCall := dto.ToolCallResponse{
				Index: common.GetPointer(len(converseInfo.ToolCalls)),
				ID:    *start.Value.ToolUseId,
				Type:  "function",
				Function: dto.FunctionResponse{
					Name:      *start.Value.Name,
					Arguments: "",
				},
			}
			converseInfo.inProgressToolCall = &toolCall
			delta := dto.ChatCompletionsStreamResponseChoiceDelta{
				ToolCalls: []dto.ToolCallResponse{toolCall},
			}
			response = createOpenAIStreamChunk(delta, converseInfo)
		}
	case *types.ConverseStreamOutputMemberContentBlockDelta:
		if e.Value.Delta != nil {
			switch delta := e.Value.Delta.(type) {
			case *types.ContentBlockDeltaMemberToolUse:
				if converseInfo.inProgressToolCall != nil && delta.Value.Input != nil {
					converseInfo.inProgressToolCall.Function.Arguments += *delta.Value.Input
					toolDelta := dto.ChatCompletionsStreamResponseChoiceDelta{
						ToolCalls: []dto.ToolCallResponse{
							{
								Index: converseInfo.inProgressToolCall.Index,
								Function: dto.FunctionResponse{
									Arguments: *delta.Value.Input,
								},
							},
						},
					}
					response = createOpenAIStreamChunk(toolDelta, converseInfo)
				}
			case *types.ContentBlockDeltaMemberText:
				if delta.Value != "" {
					converseInfo.ResponseText.WriteString(delta.Value)
					textDelta := dto.ChatCompletionsStreamResponseChoiceDelta{Content: &delta.Value}
					response = createOpenAIStreamChunk(textDelta, converseInfo)
				}
			case *types.ContentBlockDeltaMemberReasoningContent:
				if reasoningDelta, ok := delta.Value.(*types.ReasoningContentBlockDeltaMemberText); ok && reasoningDelta.Value != "" {
					converseInfo.ReasoningText.WriteString(reasoningDelta.Value)
					reasoningDeltaContent := dto.ChatCompletionsStreamResponseChoiceDelta{ReasoningContent: &reasoningDelta.Value}
					response = createOpenAIStreamChunk(reasoningDeltaContent, converseInfo)
				}
			}
		}
	case *types.ConverseStreamOutputMemberContentBlockStop:
		if converseInfo.inProgressToolCall != nil {
			converseInfo.ToolCalls = append(converseInfo.ToolCalls, *converseInfo.inProgressToolCall)
			converseInfo.inProgressToolCall = nil
		}
	case *types.ConverseStreamOutputMemberMessageStop:
		finishReason := "stop"
		if len(converseInfo.ToolCalls) > 0 {
			finishReason = "tool_calls"
		} else if e.Value.StopReason != "" {
			finishReason = stopReasonConverse2OpenAI(string(e.Value.StopReason))
		}
		finalDelta := dto.ChatCompletionsStreamResponseChoiceDelta{}
		response = createOpenAIStreamChunk(finalDelta, converseInfo)
		response.Choices[0].FinishReason = &finishReason
	}

	if response != nil {
		if err := helper.ObjectData(c, response); err != nil {
			common.LogError(c, "send stream chunk failed: "+err.Error())
			return wrapErr(err)
		}
	}

	return nil
}

func createOpenAIStreamChunk(delta dto.ChatCompletionsStreamResponseChoiceDelta, converseInfo *ConverseResponseInfo) *dto.ChatCompletionsStreamResponse {
	return &dto.ChatCompletionsStreamResponse{
		Id:      converseInfo.ResponseId,
		Object:  "chat.completion.chunk",
		Created: converseInfo.Created,
		Model:   converseInfo.Model,
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{
				Index: converseInfo.ContentIndex,
				Delta: delta,
			},
		},
	}
}

func handleStreamFinalResponse(c *gin.Context, info *relaycommon.RelayInfo, converseInfo *ConverseResponseInfo) {
	if converseInfo.Usage.CompletionTokens == 0 {
		converseInfo.Usage, _ = service.ResponseText2Usage(converseInfo.ResponseText.String(), info.UpstreamModelName, converseInfo.Usage.PromptTokens)
	}
	if info.ShouldIncludeUsage {
		response := helper.GenerateFinalUsageResponse(converseInfo.ResponseId, converseInfo.Created, info.UpstreamModelName, *converseInfo.Usage)
		err := helper.ObjectData(c, response)
		if err != nil {
			common.SysError("send final response failed: " + err.Error())
		}
	}
	helper.Done(c)
}

func converseHandler(c *gin.Context, resp *bedrockruntime.ConverseOutput, info *relaycommon.RelayInfo) (*dto.OpenAIErrorWithStatusCode, *dto.Usage) {
	var usage dto.Usage
	if resp.Usage != nil {
		usage = dto.Usage{
			PromptTokens:     int(*resp.Usage.InputTokens),
			CompletionTokens: int(*resp.Usage.OutputTokens),
			TotalTokens:      int(*resp.Usage.TotalTokens),
		}
	}
	openaiResponse := &dto.OpenAITextResponse{
		Object:  "chat.completion",
		Id:      fmt.Sprintf("chatcmpl-%s", common.GetUUID()),
		Created: common.GetTimestamp(),
		Model:   info.UpstreamModelName,
		Usage:   usage,
		Choices: []dto.OpenAITextResponseChoice{
			{
				Index: 0,
				Message: dto.Message{
					Role: "assistant",
				},
			},
		},
	}

	var responseTextBuilder strings.Builder
	var reasoningTextBuilder strings.Builder
	var toolCalls []dto.ToolCallResponse

	if output, ok := resp.Output.(*types.ConverseOutputMemberMessage); ok {
		for _, content := range output.Value.Content {
			switch content := content.(type) {
			case *types.ContentBlockMemberText:
				responseTextBuilder.WriteString(content.Value)
			case *types.ContentBlockMemberToolUse:
				arguments, err := json.Marshal(content.Value.Input)
				if err != nil {
					arguments = []byte("{}")
				}
				toolCalls = append(toolCalls, dto.ToolCallResponse{
					ID:   *content.Value.ToolUseId,
					Type: "function",
					Function: dto.FunctionResponse{
						Name:      *content.Value.Name,
						Arguments: string(arguments),
					},
				})
			case *types.ContentBlockMemberReasoningContent:
				if reasoningValue, ok := content.Value.(*types.ReasoningContentBlockMemberReasoningText); ok && *reasoningValue.Value.Text != "" {
					reasoningTextBuilder.WriteString(*reasoningValue.Value.Text)
				}
			}
		}
	}

	if resp.StopReason == types.StopReasonToolUse || len(toolCalls) > 0 {
		openaiResponse.Choices[0].FinishReason = "tool_calls"
		openaiResponse.Choices[0].Message.SetToolCalls(toolCalls)
	} else {
		openaiResponse.Choices[0].FinishReason = stopReasonConverse2OpenAI(string(resp.StopReason))
		contentBytes, _ := json.Marshal(responseTextBuilder.String())
		openaiResponse.Choices[0].Message.Content = contentBytes
	}

	if reasoningTextBuilder.Len() > 0 {
		openaiResponse.Choices[0].Message.ReasoningContent = reasoningTextBuilder.String()
	}

	c.JSON(200, openaiResponse)
	return nil, &openaiResponse.Usage
}
