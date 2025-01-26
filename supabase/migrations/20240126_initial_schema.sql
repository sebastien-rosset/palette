-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Generic Items Table for Flexible Inventory Management
CREATE TABLE items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    
    description TEXT,
    
    -- Dimensional attributes
    width_cm NUMERIC(10,2),
    height_cm NUMERIC(10,2),
    depth_cm NUMERIC(10,2),
    
    -- Valuation and ownership
    created_by VARCHAR(255),
    owner VARCHAR(255),
    estimated_value NUMERIC(10,2),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Flexible JSON storage for additional attributes
    additional_attributes JSONB,
    
    -- Digital assets
    primary_image_url TEXT,
    additional_image_urls TEXT[],
    
    -- Status tracking
    status VARCHAR(50) CHECK (status IN ('active', 'archived', 'sold', 'in_progress')),
    tags TEXT[]
);

-- Trigger to auto-update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_items_modtime
BEFORE UPDATE ON items
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();