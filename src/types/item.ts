export type Item = {
    id?: string;
    name: string;
    category?: string;
    description?: string;
    width_cm?: number;
    height_cm?: number;
    depth_cm?: number;
    created_by?: string;
    owner?: string;
    estimated_value?: number;
    created_at?: Date;
    updated_at?: Date;
    additional_attributes?: Record<string, any>;
    primary_image_url?: string;
    additional_image_urls?: string[];
    status?: 'active' | 'archived' | 'sold' | 'in_progress';
    tags?: string[];
};