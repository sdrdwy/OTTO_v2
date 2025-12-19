import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class LongTermMemory:
    def __init__(self, memory_file_path: str):
        self.memory_file_path = memory_file_path
        self.memos = []
        self._load_memories()
    
    def _load_memories(self):
        """Load memories from the JSONL file"""
        if os.path.exists(self.memory_file_path):
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            memory = json.loads(line)
                            self.memos.append(memory)
                        except json.JSONDecodeError:
                            continue
        else:
            # Create the file if it doesn't exist
            os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                pass
    
    def save_memories(self):
        """Save all memories to the JSONL file"""
        with open(self.memory_file_path, 'w', encoding='utf-8') as f:
            for memory in self.memos:
                f.write(json.dumps(memory, ensure_ascii=False) + '\n')
    
    def add_memory(self, memory: Dict[str, Any]):
        """Add a new memory to the collection"""
        # Check if memory already exists by ID
        if 'id' in memory:
            for i, existing_memory in enumerate(self.memos):
                if existing_memory.get('id') == memory['id']:
                    # Update existing memory
                    self.memos[i] = memory
                    self.save_memories()
                    return
        # Add new memory
        self.memos.append(memory)
        self.save_memories()
    
    def search_memories(self, query: str = "", limit: int = 10, memory_type: str = None) -> List[Dict[str, Any]]:
        """Search memories based on query, type and limit, adjusting weights"""
        results = []
        
        for memory in self.memos:
            # Apply type filter if specified
            if memory_type and memory.get('type') != memory_type:
                continue
            
            # Calculate relevance score based on query
            relevance_score = 0
            if query:
                content = memory.get('content', '').lower()
                if query.lower() in content:
                    relevance_score += 1
                # Check other fields too
                for key, value in memory.items():
                    if isinstance(value, str) and query.lower() in value.lower():
                        relevance_score += 0.5
            
            if relevance_score > 0 or not query:
                # Adjust weight based on access
                weight = memory.get('weight', 1.0)
                adjusted_weight = weight  # In a real implementation, we'd track access count
                
                results.append({
                    'memory': memory,
                    'relevance_score': relevance_score,
                    'weight': adjusted_weight
                })
        
        # Sort by relevance and weight
        results.sort(key=lambda x: (x['relevance_score'], x['weight']), reverse=True)
        
        # Return top results
        return [item['memory'] for item in results[:limit]]
    
    def update_memory_weight(self, memory_id: str, new_weight: float):
        """Update the weight of a specific memory"""
        for i, memory in enumerate(self.memos):
            if memory.get('id') == memory_id:
                self.memos[i]['weight'] = new_weight
                self.save_memories()
                break
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories"""
        return self.memos