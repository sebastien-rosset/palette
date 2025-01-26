import type { NextApiRequest, NextApiResponse } from 'next';
import { 
  createItem, 
  getItems, 
  updateItem, 
  deleteItem 
} from '../../lib/supabase-service';
import { Item } from '../../types/item';

export default async function handler(
  req: NextApiRequest, 
  res: NextApiResponse
) {
  try {
    switch (req.method) {
      case 'GET':
        const items = await getItems(req.query as Partial<Item>);
        return res.status(200).json(items);
      
      case 'POST':
        const newItem = await createItem(req.body);
        return res.status(201).json(newItem);
      
      case 'PUT':
        const { id, ...updates } = req.body;
        const updatedItem = await updateItem(id, updates);
        return res.status(200).json(updatedItem);
      
      case 'DELETE':
        await deleteItem(req.body.id);
        return res.status(204).end();
      
      default:
        res.setHeader('Allow', ['GET', 'POST', 'PUT', 'DELETE']);
        return res.status(405).end(`Method ${req.method} Not Allowed`);
    }
  } catch (error) {
    console.error('API error:', error);
    return res.status(500).json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
}