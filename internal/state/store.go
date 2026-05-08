package state

import (
	"sync"
	"time"

	"uni-api/internal/anthropic"
	"uni-api/internal/openai"
)

type ResponseRecord struct {
	ID         string
	Response   openai.Response
	Transcript []anthropic.MessageParam
	Status     string
	CreatedAt  int64
}

type ToolCallRecord struct {
	OpenAICallID       string
	AnthropicToolUseID string
	ResponseID         string
	Name               string
	Arguments          string
	OutputIndex        int
	CreatedAt          int64
	ResolvedAt         int64
}

type Store struct {
	mu        sync.RWMutex
	ttl       time.Duration
	records   map[string]ResponseRecord
	toolCalls map[string]map[string]ToolCallRecord
}

func NewStore(ttl time.Duration) *Store {
	return &Store{ttl: ttl, records: map[string]ResponseRecord{}, toolCalls: map[string]map[string]ToolCallRecord{}}
}

func (s *Store) Save(record ResponseRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	record.Response = cloneResponse(record.Response)
	record.Transcript = cloneMessages(record.Transcript)
	s.records[record.ID] = record
	s.indexToolCallsLocked(record)
}

func (s *Store) Get(id string) (ResponseRecord, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.records[id]
	if !ok {
		return ResponseRecord{}, false
	}
	record.Response = cloneResponse(record.Response)
	record.Transcript = cloneMessages(record.Transcript)
	return record, true
}

func (s *Store) Delete(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.records[id]; !ok {
		return false
	}
	delete(s.records, id)
	delete(s.toolCalls, id)
	return true
}

func (s *Store) UpdateStatus(id, status string) (ResponseRecord, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	record, ok := s.records[id]
	if !ok {
		return ResponseRecord{}, false
	}
	record.Status = status
	record.Response.Status = status
	s.records[id] = record
	record.Response = cloneResponse(record.Response)
	record.Transcript = cloneMessages(record.Transcript)
	return record, true
}

func (s *Store) Cleanup(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cutoff := now.Add(-s.ttl).Unix()
	for id, record := range s.records {
		if record.CreatedAt < cutoff {
			delete(s.records, id)
			delete(s.toolCalls, id)
		}
	}
}

func (s *Store) FindToolCall(responseID, callID string) (ToolCallRecord, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	byCall := s.toolCalls[responseID]
	if byCall == nil {
		return ToolCallRecord{}, false
	}
	record, ok := byCall[callID]
	return record, ok
}

func (s *Store) FindToolCallByCallID(callID string) (ToolCallRecord, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var found ToolCallRecord
	matches := 0
	for _, byCall := range s.toolCalls {
		record, ok := byCall[callID]
		if !ok {
			continue
		}
		matches++
		if matches == 1 {
			found = record
		}
	}
	return found, matches
}

func (s *Store) MarkToolCallResolved(responseID, callID string, resolvedAt int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	byCall := s.toolCalls[responseID]
	if byCall == nil {
		return false
	}
	record, ok := byCall[callID]
	if !ok || record.ResolvedAt != 0 {
		return false
	}
	record.ResolvedAt = resolvedAt
	byCall[callID] = record
	return true
}

func (s *Store) indexToolCallsLocked(record ResponseRecord) {
	existing := s.toolCalls[record.ID]
	next := map[string]ToolCallRecord{}
	toolUses := toolUsesByIndex(record.Transcript)
	toolUseIndex := 0
	for i, item := range record.Response.Output {
		if item.Type != "function_call" || item.CallID == "" {
			continue
		}
		toolUseID := item.CallID
		if toolUseIndex < len(toolUses) && toolUses[toolUseIndex].ID != "" {
			toolUseID = toolUses[toolUseIndex].ID
		}
		toolUseIndex++
		resolvedAt := int64(0)
		if old, ok := existing[item.CallID]; ok {
			resolvedAt = old.ResolvedAt
		}
		next[item.CallID] = ToolCallRecord{
			OpenAICallID:       item.CallID,
			AnthropicToolUseID: toolUseID,
			ResponseID:         record.ID,
			Name:               item.Name,
			Arguments:          item.Arguments,
			OutputIndex:        i,
			CreatedAt:          record.CreatedAt,
			ResolvedAt:         resolvedAt,
		}
	}
	if len(next) == 0 {
		delete(s.toolCalls, record.ID)
		return
	}
	s.toolCalls[record.ID] = next
}

func toolUsesByIndex(transcript []anthropic.MessageParam) []anthropic.ContentBlock {
	var out []anthropic.ContentBlock
	for _, msg := range transcript {
		if msg.Role != "assistant" {
			continue
		}
		for _, block := range msg.Content {
			if block.Type == "tool_use" {
				out = append(out, block)
			}
		}
	}
	return out
}

func cloneMessages(in []anthropic.MessageParam) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, len(in))
	for i, msg := range in {
		blocks := make([]anthropic.ContentBlock, len(msg.Content))
		copy(blocks, msg.Content)
		out[i] = anthropic.MessageParam{Role: msg.Role, Content: blocks}
	}
	return out
}

func cloneResponse(in openai.Response) openai.Response {
	out := in
	out.Output = make([]openai.OutputItem, len(in.Output))
	for i, item := range in.Output {
		out.Output[i] = item
		out.Output[i].Content = make([]openai.ContentItem, len(item.Content))
		copy(out.Output[i].Content, item.Content)
	}
	if in.Usage != nil {
		usage := *in.Usage
		out.Usage = &usage
	}
	if in.Error != nil {
		errObj := *in.Error
		out.Error = &errObj
	}
	return out
}
