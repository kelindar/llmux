package identity

import (
	"fmt"
	"sync/atomic"
)

var next atomic.Uint64

// New returns a process-local identifier with the requested readable prefix.
// IDs are unique within the process and do not carry security meaning.
func New(prefix string) string {
	return fmt.Sprintf("%s%x", prefix, next.Add(1))
}
