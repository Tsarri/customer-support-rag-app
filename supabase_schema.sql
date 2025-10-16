-- Tickets table
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_name TEXT,
    customer_email TEXT,
    customer_location TEXT,
    customer_user_type TEXT,
    inquiry_text TEXT NOT NULL,
    inquiry_type TEXT,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'new',
    ai_confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticket_id UUID REFERENCES tickets(ticket_id) ON DELETE CASCADE,
    sender_type TEXT NOT NULL, -- 'customer' or 'ai'
    message_text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_ticket_id ON messages(ticket_id);
