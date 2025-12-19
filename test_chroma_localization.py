#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Chroma本地化存储和JSONL备份功能
"""
import os
import json
from unittest.mock import Mock

# 设置API密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-c763fc92bf8c46c7ae31639b05d89c96"

from memory.vector_memory import VectorMemoryManager, KnowledgeBaseManager
from memory.long_term_mem import LongTermMemory

def test_vector_memory_localization():
    """测试向量记忆的本地化存储"""
    print("测试向量记忆本地化存储...")
    
    # 创建向量记忆管理器
    vm = VectorMemoryManager(collection_name="test_memory", persist_directory="./chroma_db")
    
    # 添加一些测试记忆
    test_memory = {
        "type": "test_memory",
        "timestamp": "2025-01-01T00:00:00",
        "content": "这是一个测试记忆",
        "details": {"test": "value", "number": 42}
    }
    
    vm.add_memory(test_memory)
    print("✓ 添加记忆成功")
    
    # 搜索记忆
    results = vm.search_memories("测试")
    print(f"✓ 搜索到 {len(results)} 条记忆")
    
    # 检查备份文件
    backup_file = vm.backup_file_path
    if os.path.exists(backup_file):
        with open(backup_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✓ 备份文件存在，包含 {len(lines)} 条记录")
    else:
        print("✗ 备份文件不存在")
    
    # 验证备份文件内容
    if len(lines) > 0:
        backup_memory = json.loads(lines[0])
        if backup_memory["content"] == test_memory["content"]:
            print("✓ 备份文件内容正确")
        else:
            print("✗ 备份文件内容不正确")
    
    print()

def test_knowledge_base_localization():
    """测试知识库本地化存储"""
    print("测试知识库本地化存储...")
    
    # 创建知识库管理器
    kb = KnowledgeBaseManager(collection_name="test_knowledge", persist_directory="./chroma_db")
    
    # 创建测试知识库文件
    test_kb_file = "./data/test_knowledge.jsonl"
    os.makedirs("./data", exist_ok=True)
    
    test_knowledge = {
        "topic": "test_topic",
        "content": "这是一个测试知识项",
        "details": {"info": "test", "value": 123}
    }
    
    with open(test_kb_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_knowledge, ensure_ascii=False) + '\n')
    
    # 加载知识库
    kb.load_knowledge_from_jsonl(test_kb_file)
    print("✓ 知识库加载成功")
    
    # 搜索知识
    results = kb.search_knowledge("测试")
    print(f"✓ 搜索到 {len(results)} 条知识")
    
    # 检查备份文件
    backup_file = kb.backup_file_path
    if os.path.exists(backup_file):
        with open(backup_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✓ 知识库备份文件存在，包含 {len(lines)} 条记录")
    else:
        print("✗ 知识库备份文件不存在")
    
    # 清理测试文件
    if os.path.exists(test_kb_file):
        os.remove(test_kb_file)
    
    print()

def test_long_term_memory():
    """测试长期记忆"""
    print("测试长期记忆...")
    
    # 创建长期记忆
    ltm = LongTermMemory("./memory/test_agent_long_term.jsonl", persist_directory="./chroma_db")
    
    # 添加测试记忆
    test_memory = {
        "type": "test_long_term",
        "timestamp": "2025-01-01T00:00:00",
        "content": "长期记忆测试内容",
        "details": {"test": "long_term", "value": 999}
    }
    
    ltm.add_memory(test_memory)
    print("✓ 长期记忆添加成功")
    
    # 搜索记忆
    results = ltm.search_memories("长期")
    print(f"✓ 搜索到 {len(results)} 条长期记忆")
    
    print()

def main():
    print("开始测试Chroma本地化存储和JSONL备份功能")
    print("="*50)
    
    # 确保目录存在
    os.makedirs("./chroma_db", exist_ok=True)
    os.makedirs("./memory_backup", exist_ok=True)
    os.makedirs("./knowledge_backup", exist_ok=True)
    
    test_vector_memory_localization()
    test_knowledge_base_localization()
    test_long_term_memory()
    
    print("="*50)
    print("所有测试完成！")

if __name__ == "__main__":
    main()