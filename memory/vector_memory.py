import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
import uuid


class VectorMemoryManager:
    """
    向量数据库管理器，使用Chroma进行向量存储和检索
    """
    def __init__(self, collection_name: str = "long_term_memory"):
        # 初始化嵌入模型和向量数据库
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        self.collection_name = collection_name
        
        # 创建或连接到向量数据库
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # 用于存储额外的元数据
        self.metadata_store = {}
    
    def add_memory(self, memory: Dict[str, Any]):
        """添加记忆到向量数据库"""
        # 生成唯一ID
        memory_id = memory.get("id", f"memory_{uuid.uuid4()}")
        
        # 准备文档内容
        content = memory.get("content", "")
        if not content:
            # 如果没有内容，组合其他字段
            content = f"{memory.get('type', '')} {memory.get('timestamp', '')}"
        
        # 创建文档
        doc = Document(
            page_content=content,
            metadata={
                "id": memory_id,
                "type": memory.get("type", ""),
                "timestamp": memory.get("timestamp", str(datetime.now())),
                "details": json.dumps(memory.get("details", {})),
                "weight": memory.get("weight", 1.0),
                **memory.get("metadata", {})
            }
        )
        
        # 添加到向量数据库
        self.vector_store.add_documents([doc])
        
        # 存储完整的记忆数据
        self.metadata_store[memory_id] = memory
    
    def search_memories(self, query: str, limit: int = 10, memory_type: str = None) -> List[Dict[str, Any]]:
        """搜索记忆"""
        # 使用向量相似性搜索
        results = self.vector_store.similarity_search_with_score(
            query, 
            k=limit,
            filter={"type": memory_type} if memory_type else None
        )
        
        # 提取记忆数据
        memories = []
        for doc, score in results:
            memory_id = doc.metadata.get("id")
            if memory_id and memory_id in self.metadata_store:
                memory = self.metadata_store[memory_id]
            else:
                # 如果没有找到完整记忆，构建一个基础版本
                memory = {
                    "id": memory_id,
                    "content": doc.page_content,
                    "type": doc.metadata.get("type"),
                    "timestamp": doc.metadata.get("timestamp"),
                    "details": json.loads(doc.metadata.get("details", "{}")),
                    "weight": doc.metadata.get("weight", 1.0),
                    "similarity_score": 1.0 - score  # 转换为相似度分数
                }
            
            memories.append(memory)
        
        # 根据权重和相似度排序
        memories.sort(key=lambda x: (x.get("weight", 1.0) * x.get("similarity_score", 1.0)), reverse=True)
        return memories[:limit]
    
    def get_memories_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据主题获取记忆"""
        return self.search_memories(query=topic, limit=limit, memory_type="learning_from_teacher")
    
    def get_recent_memories(self, limit: int = 10, memory_type: str = None) -> List[Dict[str, Any]]:
        """获取最近的记忆"""
        # 先获取所有相关的记忆
        all_memories = list(self.metadata_store.values())
        
        if memory_type:
            all_memories = [m for m in all_memories if m.get("type") == memory_type]
        
        # 按时间戳排序
        all_memories.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
        return all_memories[:limit]
    
    def update_memory_weight(self, memory_id: str, new_weight: float):
        """更新记忆权重"""
        if memory_id in self.metadata_store:
            self.metadata_store[memory_id]["weight"] = new_weight
            # 重新添加文档以更新
            memory = self.metadata_store[memory_id]
            self.delete_memory(memory_id)
            self.add_memory(memory)
    
    def delete_memory(self, memory_id: str):
        """删除记忆"""
        if memory_id in self.metadata_store:
            del self.metadata_store[memory_id]
        # 从向量数据库中删除（如果支持）
        try:
            self.vector_store._collection.delete(ids=[memory_id])
        except:
            pass  # 如果删除失败，忽略
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """获取所有记忆"""
        return list(self.metadata_store.values())


class KnowledgeBaseManager:
    """
    知识库管理器，专门用于处理知识库的向量化存储
    """
    def __init__(self, collection_name: str = "knowledge_base"):
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        self.collection_name = collection_name
        
        # 创建或连接到向量数据库
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # 用于存储知识项
        self.knowledge_store = {}
    
    def load_knowledge_from_jsonl(self, jsonl_path: str):
        """从JSONL文件加载知识到向量数据库（安全版本）"""
        if not os.path.exists(jsonl_path):
            print(f"知识库文件不存在: {jsonl_path}")
            return
        
        MAX_CONTENT_LENGTH = 6000  # DashScope 安全上限（中文字符）

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    knowledge_item = json.loads(line)
                    if not isinstance(knowledge_item, dict):
                        print(f"第 {line_num} 行不是有效 JSON 对象，跳过")
                        continue

                    # === 关键修复：动态拼接所有非空字段 ===
                    content_parts = []
                    for key, value in knowledge_item.items():
                        if value is None:
                            continue
                        value_str = str(value).strip()
                        if value_str and value_str.lower() not in ("null", "none", ""):
                            # 保留字段名作为上下文（增强语义）
                            content_parts.append(f"[{key}]{value_str}")

                    if not content_parts:
                        print(f"第 {line_num} 行无有效内容字段，跳过")
                        continue

                    content = "\n".join(content_parts)

                    # === 防超长截断 ===
                    if len(content) > MAX_CONTENT_LENGTH:
                        content = content[:MAX_CONTENT_LENGTH] + " [内容已截断]"

                    # 生成唯一ID
                    knowledge_id = knowledge_item.get("id") or f"kb_{line_num}_{uuid.uuid4()}"

                    # 推荐 topic：优先用 name / topic / 第一个字段
                    topic = (
                        knowledge_item.get("topic") or
                        knowledge_item.get("name") or
                        knowledge_item.get("title") or
                        next(iter(knowledge_item.keys()), "通用")
                    )
                    if topic and isinstance(topic, str):
                        topic = topic.strip().split('\n')[0]  # 取第一行避免换行
                    else:
                        topic = "通用"

                    # 创建 Document
                    doc = Document(
                        page_content=content,
                        metadata={
                            "id": knowledge_id,
                            "topic": topic,
                            "source": jsonl_path,
                            "original_data": json.dumps(knowledge_item, ensure_ascii=False)
                        }
                    )

                    # 安全添加：再次确认非空
                    if not doc.page_content.strip():
                        print(f"第 {line_num} 行生成内容为空，跳过")
                        continue

                    self.vector_store.add_documents([doc])
                    self.knowledge_store[knowledge_id] = knowledge_item

                except json.JSONDecodeError as e:
                    print(f"解析知识库第 {line_num} 行 JSON 失败: {e}")
                except Exception as e:
                    print(f"处理知识库第 {line_num} 行时发生错误: {e}")
                    continue
    
    def search_knowledge(self, query: str, topic: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识库"""
        # 构建过滤条件
        filters = None
        if topic:
            filters = {"topic": topic.strip()}
        
        # 使用向量相似性搜索
        results = self.vector_store.similarity_search_with_score(
            query,
            k=limit,
            filter=filters
        )
        
        # 提取知识数据
        knowledge_items = []
        for doc, score in results:
            knowledge_id = doc.metadata.get("id")
            if knowledge_id and knowledge_id in self.knowledge_store:
                knowledge_item = self.knowledge_store[knowledge_id]
            else:
                # 如果没有找到完整数据，从元数据重建
                original_data = doc.metadata.get("original_data")
                if original_data:
                    knowledge_item = json.loads(original_data)
                else:
                    knowledge_item = {
                        "topic": doc.metadata.get("topic", "通用"),
                        "content": doc.page_content
                    }
            
            knowledge_item["similarity_score"] = 1.0 - score
            knowledge_items.append(knowledge_item)
        
        return knowledge_items
    
    def get_knowledge_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据主题获取知识"""
        return self.search_knowledge(query="", topic=topic, limit=limit)
    
    def get_all_topics(self) -> List[str]:
        """获取所有主题"""
        all_knowledge = list(self.knowledge_store.values())
        topics = list(set([item.get("topic", "通用") for item in all_knowledge]))
        return topics