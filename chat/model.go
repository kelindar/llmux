package chat

// Model is one optional catalog entry returned by GET /models.
// The wire object discriminator is owned by the catalog handler.
type Model struct {
	ID      string // Model identifier returned to clients.
	Created int64  // Unix creation time in seconds.
	OwnedBy string // Owner label shown in the models list.
}
