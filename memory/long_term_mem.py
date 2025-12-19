import json
import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from memory.vector_memory import VectorMemoryManager


class LongTermMemory:
    def __init__(self, memory_file_path: str, persist_directory: str = "./chroma_db"):
        # 使用向量数据库管理器替代文件存储
        agent_name = os.path.basename(memory_file_path).replace('_long_term.jsonl', '')
        self.vector_memory = VectorMemoryManager(collection_name=f"memory_{agent_name}", persist_directory=persist_directory)
        self.memory_file_path = memory_file_path
        # 迁移旧的JSONL文件到向量数据库（如果存在）
        self._migrate_from_jsonl()
    
    def _migrate_from_jsonl(self):
        """从JSONL文件迁移到向量数据库"""
        if os.path.exists(self.memory_file_path):
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            memory = json.loads(line)
                            self.vector_memory.add_memory(memory)
                        except json.JSONDecodeError:
                            continue
    
    def save_memories(self):
        """保存记忆到向量数据库（实际上不需要显式保存，因为已实时更新）"""
        pass  # 向量数据库实时更新，不需要显式保存
    
    def add_memory(self, memory: Dict[str, Any]):
        """添加新记忆到向量数据库"""
        # 向量数据库管理器会处理ID重复问题
        self.vector_memory.add_memory(memory)
    
    def search_memories(self, query: str = "", limit: int = 10, memory_type: str = None) -> List[Dict[str, Any]]:
        """搜索记忆"""
        # 使用向量数据库进行语义搜索
        if query:
            return self.vector_memory.search_memories(query, limit=limit, memory_type=memory_type)
        else:
            # 如果没有查询，返回最近的记忆
            return self.vector_memory.get_recent_memories(limit=limit, memory_type=memory_type)
    
    def search_by_content_fields(self, query: str = "", fields: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """根据特定字段搜索记忆"""
        # 使用向量数据库进行语义搜索，然后根据字段进行过滤
        if query:
            memories = self.vector_memory.search_memories(query, limit=limit*2)  # 获取更多结果以供筛选
        else:
            memories = self.vector_memory.get_recent_memories(limit=limit*2)
        
        if fields:
            # 如果指定了字段，筛选包含这些字段的记忆
            filtered_memories = []
            for memory in memories:
                for field in fields:
                    if field in memory:
                        field_value = str(memory[field])
                        if not query or query.lower() in field_value.lower():
                            filtered_memories.append(memory)
                            break
            return filtered_memories[:limit]
        else:
            # 如果没有指定字段，返回所有搜索结果
            return memories[:limit]
    
    def get_recent_memories(self, limit: int = 10, memory_type: str = None) -> List[Dict[str, Any]]:
        """获取最近的记忆，可选择按类型过滤"""
        return self.vector_memory.get_recent_memories(limit=limit, memory_type=memory_type)
    
    def get_memories_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取与特定主题相关的记忆"""
        return self.vector_memory.get_memories_by_topic(topic, limit=limit)
    
    def update_memory_weight(self, memory_id: str, new_weight: float):
        """更新特定记忆的权重"""
        self.vector_memory.update_memory_weight(memory_id, new_weight)
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """获取所有记忆"""
        return self.vector_memory.get_all_memories()