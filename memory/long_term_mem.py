import json
import os
import re
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
            
            # Calculate relevance score based on query using semantic matching
            relevance_score = 0
            if query:
                # Exact keyword match gets higher score
                content = memory.get('content', '')
                if query.lower() in content.lower():
                    relevance_score += 2
                # Partial word match
                query_words = query.lower().split()
                for word in query_words:
                    if word in content.lower():
                        relevance_score += 0.5
                
                # Check other fields too
                for key, value in memory.items():
                    if isinstance(value, str):
                        if query.lower() in value.lower():
                            relevance_score += 1
                        # Check for partial matches in other fields
                        for word in query_words:
                            if word in value.lower():
                                relevance_score += 0.3
            
            if relevance_score > 0 or not query:
                # Adjust weight based on access and recency
                weight = memory.get('weight', 1.0)
                
                # Boost recency for more recent memories
                try:
                    timestamp = memory.get('timestamp', '')
                    if timestamp:
                        from datetime import datetime
                        memory_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')) if 'Z' in timestamp else datetime.fromisoformat(timestamp)
                        now = datetime.now()
                        age_in_hours = (now - memory_time).total_seconds() / 3600
                        # Recent memories get higher weight (less than 24 hours)
                        if age_in_hours < 24:
                            weight *= 1.2
                        elif age_in_hours < 168:  # Less than a week
                            weight *= 1.1
                except:
                    pass  # If timestamp parsing fails, just continue
                
                results.append({
                    'memory': memory,
                    'relevance_score': relevance_score,
                    'weight': weight
                })
        
        # Sort by relevance and weight
        results.sort(key=lambda x: (x['relevance_score'], x['weight']), reverse=True)
        
        # Return top results
        return [item['memory'] for item in results[:limit]]
    
    def search_by_content_fields(self, query: str = "", fields: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by specific fields for more targeted retrieval"""
        results = []
        
        for memory in self.memos:
            relevance_score = 0
            
            if query:
                # Search in specified fields if provided
                if fields:
                    for field in fields:
                        if field in memory:
                            field_value = str(memory[field])
                            if query.lower() in field_value.lower():
                                relevance_score += 2  # High score for exact field match
                            # Partial match in field
                            for word in query.lower().split():
                                if word in field_value.lower():
                                    relevance_score += 0.5
                else:
                    # Search in all fields
                    for key, value in memory.items():
                        if isinstance(value, str):
                            if query.lower() in value.lower():
                                relevance_score += 1
                            # Partial match
                            for word in query.lower().split():
                                if word in value.lower():
                                    relevance_score += 0.3
            
            if relevance_score > 0:
                weight = memory.get('weight', 1.0)
                results.append({
                    'memory': memory,
                    'relevance_score': relevance_score,
                    'weight': weight
                })
        
        # Sort by relevance and weight
        results.sort(key=lambda x: (x['relevance_score'], x['weight']), reverse=True)
        
        # Return top results
        return [item['memory'] for item in results[:limit]]
    
    def get_recent_memories(self, limit: int = 10, memory_type: str = None) -> List[Dict[str, Any]]:
        """Get most recent memories, optionally filtered by type"""
        filtered_memories = []
        
        for memory in self.memos:
            if memory_type and memory.get('type') != memory_type:
                continue
            filtered_memories.append(memory)
        
        # Sort by timestamp (most recent first)
        try:
            filtered_memories.sort(
                key=lambda m: m.get('timestamp', ''), 
                reverse=True
            )
        except:
            # If sorting by timestamp fails, keep original order
            pass
        
        return filtered_memories[:limit]
    
    def get_memories_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get memories related to a specific topic"""
        results = []
        
        for memory in self.memos:
            # Check if topic appears in content or details
            content = memory.get('content', '')
            details = memory.get('details', {})
            
            topic_relevance = 0
            
            if topic.lower() in content.lower():
                topic_relevance += 2
            
            # Check in details dict
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, str) and topic.lower() in value.lower():
                        topic_relevance += 1
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and topic.lower() in item.lower():
                                topic_relevance += 0.5
            
            if topic_relevance > 0:
                results.append({
                    'memory': memory,
                    'relevance': topic_relevance
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
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