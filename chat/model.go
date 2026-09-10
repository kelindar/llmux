package chat

// Model is one optional catalog entry returned by GET /models.
type Model struct {
	ID      string `json:"id"`       // Model identifier returned to clients.
	Object  string `json:"object"`   // Object type; defaults to "model" when empty.
	Created int64  `json:"created"`  // Unix creation time in seconds.
	OwnedBy string `json:"owned_by"` // Owner label shown in the models list.
}
