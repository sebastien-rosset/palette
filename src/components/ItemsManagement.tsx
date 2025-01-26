import React, { useState, useEffect } from 'react';
import { Item } from '../types/item';

export default function ItemsManagement() {
  const [items, setItems] = useState<Item[]>([]);
  const [newItem, setNewItem] = useState<Partial<Item>>({});

  useEffect(() => {
    fetchItems();
  }, []);

  const fetchItems = async () => {
    try {
      const response = await fetch('/api/items');
      const data = await response.json();
      setItems(data);
    } catch (error) {
      console.error('Failed to fetch items:', error);
    }
  };

  const handleCreateItem = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newItem)
      });
      const createdItem = await response.json();
      setItems([...items, createdItem]);
      setNewItem({});
    } catch (error) {
      console.error('Failed to create item:', error);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Items Management</h1>
      
      <form onSubmit={handleCreateItem} className="mb-4">
        <input
          type="text"
          placeholder="Name"
          value={newItem.name || ''}
          onChange={(e) => setNewItem({ ...newItem, name: e.target.value })}
          className="border p-2 mr-2"
        />
        <input
          type="text"
          placeholder="Category"
          value={newItem.category || ''}
          onChange={(e) => setNewItem({ ...newItem, category: e.target.value })}
          className="border p-2 mr-2"
        />
        <button 
          type="submit" 
          className="bg-blue-500 text-white p-2 rounded"
        >
          Add Item
        </button>
      </form>

      <div>
        {items.map((item) => (
          <div key={item.id} className="border p-2 mb-2">
            <h2>{item.name}</h2>
            <p>Category: {item.category}</p>
          </div>
        ))}
      </div>
    </div>
  );
}