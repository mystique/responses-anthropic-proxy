# OpenAI Responses API 实现文档

整理日期：2026-05-06  
基础地址：`https://api.openai.com/v1`  
认证：所有请求使用 `Authorization: Bearer $OPENAI_API_KEY`，JSON 请求使用 `Content-Type: application/json`。

本文面向实现者，目标是让你可以据此封装 HTTP client、定义类型、处理普通响应、流式响应、工具调用和多轮状态。

参考来源：

- OpenAI API Reference, Responses: `https://developers.openai.com/api/reference/resources/responses/`
- Create a response: `https://developers.openai.com/api/reference/resources/responses/methods/create/`
- Streaming events: `https://developers.openai.com/api/reference/resources/responses/streaming-events/`
- 官方 OpenAPI spec: `https://github.com/openai/openai-openapi`

## 1. API 总览

Responses API 是 OpenAI 当前统一的模型响应接口，用于文本、图像、文件、音频输入，文本或结构化 JSON 输出，内置工具调用，以及自定义函数调用。核心资源是 `response`。

常用实现路径：

1. 单轮文本或多模态输入：`POST /responses`
2. 多轮上下文：传 `previous_response_id`，或使用 `conversation`
3. 工具调用：模型返回 `function_call` 等 output item，业务执行工具后再次 `POST /responses`，把 `function_call_output` 放入 `input`
4. 流式输出：`stream: true`，按 Server-Sent Events 解析事件

## 2. 端点定义

### 2.1 Create response

`POST /responses`

创建模型响应。非流式返回 `Response` 对象；`stream: true` 时返回 `text/event-stream`。

最小请求：

```json
{
  "model": "gpt-4.1",
  "input": "Tell me a three sentence bedtime story."
}
```

请求体类型：

```ts
type CreateResponseRequest = {
  model?: string;
  input?: string | InputItem[];
  instructions?: string | null;
  previous_response_id?: string | null;
  max_output_tokens?: number | null;
  temperature?: number | null;
  top_p?: number | null;
  parallel_tool_calls?: boolean | null;
  tools?: Tool[];
  tool_choice?: string | object | null;
  stream?: boolean | null;
};
```

## 3. 工具调用

工具调用闭环：

1. 请求中传 `tools[{type:"function"}]`
2. 模型输出 `function_call`
3. 客户端执行工具
4. 后续请求 input 带 `function_call_output`

### Web search tools

The proxy accepts `tools` entries with `type:"web_search"` and `type:"web_search_preview"`.
Both are mapped to Anthropic server-side web search.

Supported pass-through fields:

- `max_uses`
- `allowed_domains`
- `blocked_domains`
- `filters.allowed_domains`
- `filters.blocked_domains`
- `user_location.type`
- `user_location.city`
- `user_location.region`
- `user_location.country`
- `user_location.timezone`

OpenAI-only fields without an Anthropic equivalent, such as `search_context_size`, are ignored.

## 4. 流式事件

Responses API 流式响应使用 SSE。常见事件包括：

- `response.created`
- `response.in_progress`
- `response.output_text.delta`
- `response.output_text.done`
- `response.function_call_arguments.done`
- `response.completed`
- `response.failed`
