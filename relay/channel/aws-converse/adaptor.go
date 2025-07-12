package aws_converse

import (
	"errors"
	"net/http"

	"io"
	"veloera/dto"
	relaycommon "veloera/relay/common"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/gin-gonic/gin"
)

type ConverseResponse struct {
	dto.OpenAITextResponse
}

type Adaptor struct {
}

func (a *Adaptor) ConvertClaudeRequest(c *gin.Context, info *relaycommon.RelayInfo, request *dto.ClaudeRequest) (any, error) {
	return nil, errors.New("not implemented")
}

func (a *Adaptor) ConvertAudioRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.AudioRequest) (io.Reader, error) {
	return nil, errors.New("not implemented")
}

func (a *Adaptor) ConvertImageRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.ImageRequest) (any, error) {
	return nil, errors.New("not implemented")
}

func (a *Adaptor) Init(info *relaycommon.RelayInfo) {
}

func (a *Adaptor) GetRequestURL(info *relaycommon.RelayInfo) (string, error) {
	return "", nil
}

func (a *Adaptor) SetupRequestHeader(c *gin.Context, req *http.Header, info *relaycommon.RelayInfo) error {
	return nil
}

func (a *Adaptor) ConvertOpenAIRequest(c *gin.Context, info *relaycommon.RelayInfo, request *dto.GeneralOpenAIRequest) (any, error) {
	if request == nil {
		return nil, errors.New("request is nil")
	}
	converseRequest, err := a.RequestOpenAI2ConverseRequest(request)
	if err != nil {
		return nil, err
	}
	c.Set("request_model", request.Model)
	c.Set("converted_request", converseRequest)
	return converseRequest, nil
}

func (a *Adaptor) ConvertRerankRequest(c *gin.Context, relayMode int, request dto.RerankRequest) (any, error) {
	return nil, errors.New("not implemented")
}

func (a *Adaptor) ConvertEmbeddingRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.EmbeddingRequest) (any, error) {
	return nil, errors.New("not implemented")
}

func (a *Adaptor) ConvertOpenAIResponsesRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.OpenAIResponsesRequest) (any, error) {
	return nil, errors.New("not implemented")
}

func (a *Adaptor) DoRequest(c *gin.Context, info *relaycommon.RelayInfo, requestBody io.Reader) (any, error) {
	client, err := newAwsClient(info)
	if err != nil {
		return nil, err
	}

	converseReq, exists := c.Get("converted_request")
	if !exists {
		return nil, errors.New("converted request not found")
	}

	requestModel, exists := c.Get("request_model")
	if !exists {
		return nil, errors.New("request model not found")
	}

	modelId := requestModel.(string)

	request := converseReq.(*ConverseRequest)
	input := &bedrockruntime.ConverseInput{
		ModelId:                      &modelId,
		Messages:                     request.Messages,
		System:                       request.System,
		InferenceConfig:              request.InferenceConfig,
		ToolConfig:                   request.ToolConfig,
		AdditionalModelRequestFields: request.AdditionalModelRequestFields,
	}

	if info.IsStream {
		streamInput := &bedrockruntime.ConverseStreamInput{
			ModelId:                      input.ModelId,
			Messages:                     input.Messages,
			System:                       input.System,
			InferenceConfig:              input.InferenceConfig,
			ToolConfig:                   input.ToolConfig,
			AdditionalModelRequestFields: input.AdditionalModelRequestFields,
		}
		streamResp, err := client.ConverseStream(c, streamInput)
		if err != nil {
			var validationErr *types.ValidationException
			if errors.As(err, &validationErr) {
				return nil, validationErr
			}
			return nil, err
		}
		c.Set("aws_stream_response", streamResp)
	} else {
		resp, err := client.Converse(c, input)
		if err != nil {
			var validationErr *types.ValidationException
			if errors.As(err, &validationErr) {
				return nil, validationErr
			}
			return nil, err
		}
		c.Set("aws_response", resp)
	}

	return nil, nil
}

func (a *Adaptor) DoResponse(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (usage any, err *dto.OpenAIErrorWithStatusCode) {
	if info.IsStream {
		if streamResp, exists := c.Get("aws_stream_response"); exists {
			if stream, ok := streamResp.(*bedrockruntime.ConverseStreamOutput); ok {
				err, usage = converseStreamHandler(c, stream, info)
			} else {
				err = wrapErr(errors.New("invalid stream response type"))
			}
		} else {
			err = wrapErr(errors.New("stream response not found in context"))
		}
	} else {
		if awsResp, exists := c.Get("aws_response"); exists {
			if response, ok := awsResp.(*bedrockruntime.ConverseOutput); ok {
				err, usage = converseHandler(c, response, info)
			} else {
				err = wrapErr(errors.New("invalid response type"))
			}
		} else {
			err = wrapErr(errors.New("response not found in context"))
		}
	}
	return
}

func (a *Adaptor) GetModelList() (models []string) {
	return ModelList
}

func (a *Adaptor) GetChannelName() string {
	return ChannelName
}
