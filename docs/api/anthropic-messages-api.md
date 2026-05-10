# Anthropic Messages API 实现文档

整理日期：2026-05-06  
基础地址：`https://api.anthropic.com`  
认证：`x-api-key: $ANTHROPIC_API_KEY`  
版本头：`anthropic-version: 2023-06-01`  
JSON 请求：`content-type: application/json`

本文面向实现者，目标是让你可以据此封装 HTTP client、定义类型、处理普通响应、流式响应、工具调用、扩展思考、多模态输入和 token 计数。

参考来源：

- Messages API: `https://docs.anthropic.com/en/api/messages`
- Count message tokens: `https://docs.anthropic.com/en/api/messages-count-tokens`
- Streaming messages: `https://docs.anthropic.com/en/api/messages-streaming`
- Tool use overview: `https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview`

## 1. API 总览

Anthropic Messages API 使用结构化 `messages` 数组生成下一条 assistant 消息。它既支持单轮请求，也支持无状态多轮对话：每次请求都把历史 `user` / `assistant` turn 传入。

重要差异：

- 输入消息只有 `user` 和 `assistant` role；没有 `system` role。系统提示使用顶层 `system` 字段。
- `content` 可以是字符串，也可以是 content block 数组；字符串等价于一个 `{ "type": "text" }` block。
- 对话历史需要客户端维护；服务端不通过 `previous_response_id` 续接。
- 工具调用是 content block：模型返回 `tool_use`，业务执行后把 `tool_result` 放在下一次 `user` 消息里。
- 流式响应是 SSE，事件流为 `message_start`、content block 事件、`message_delta`、`message_stop`。

## 2. 端点定义

### 2.1 Create message

`POST /v1/messages`

创建一条模型消息。非流式返回 `Message`；`stream: true` 时返回 SSE。

最小请求：

```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 1024,
  "messages": [
    { "role": "user", "content": "Hello, Claude" }
  ]
}
```

请求体：

```ts
type CreateMessageRequest = {
  model: string;
  max_tokens: number;
  messages: MessageParam[];
  system?: string | TextBlockParam[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  tools?: ToolUnion[];
  tool_choice?: ToolChoice;
};
```

### 2.2 Count message tokens

`POST /v1/messages/count_tokens`

按 Messages 格式统计 token，不生成模型输出。可统计文本、工具、图片和文档。

## 3. 请求字段语义

```ts
type MessageParam = {
  role: "user" | "assistant";
  content: string | ContentBlockParam[];
};
```

关键语义：

- `max_tokens`：本次最多生成 token 数。
- `messages`：最多 100,000 条。连续相同 role 的消息会被合并为单个 turn。
- `system`：顶层系统提示；Messages API 不接受 role 为 `system` 的 message。
- `temperature`：0 到 1；即使为 0 也不保证完全确定。
- `top_p`：nucleus sampling，高级场景使用。

### Web search server tool

OpenAI `web_search` and `web_search_preview` map to:

```json
{"type":"web_search_20250305","name":"web_search"}
```

Anthropic response blocks `server_tool_use` and `web_search_tool_result` are preserved in stored transcript history so `previous_response_id` continuation can send valid Messages history back upstream.
