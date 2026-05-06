package state

import (
	"sync"
	"time"

	"uni-api/internal/anthropic"
)

type ResponseRecord struct {
	ID         string
	Transcript []anthropic.MessageParam
	Status     string
	CreatedAt  int64
}

type Store struct {
	mu      sync.RWMutex
	ttl     time.Duration
	records map[string]ResponseRecord
}

func NewStore(ttl time.Duration) *Store {
	return &Store{ttl: ttl, records: map[string]ResponseRecord{}}
}

func (s *Store) Save(record ResponseRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()
	record.Transcript = cloneMessages(record.Transcript)
	s.records[record.ID] = record
}

func (s *Store) Get(id string) (ResponseRecord, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	record, ok := s.records[id]
	if !ok {
		return ResponseRecord{}, false
	}
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
		}
	}
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
