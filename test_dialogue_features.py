#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新添加的对话功能特性：
1. 实时输出对话内容
2. 创建JSON日志文件
3. 完善事件记忆功能
"""

import json
import os
from datetime import datetime
from agents.expert_agent import ExpertAgent
from agents.student_agent import StudentAgent
from world.world_simluator import WorldSimulator

def test_dialogue_features():
    print("开始测试对话功能特性...")
    
    # 设置API密钥
    os.environ["DASHSCOPE_API_KEY"] = "sk-c763fc92bf8c46c7ae31639b05d89c96"
    
    # 创建专家代理（教师）
    expert_agent = ExpertAgent("./config/agents/expert.json")
    
    # 创建学生代理
    student_agents = [
        StudentAgent("./config/agents/Arisu.json"),
        StudentAgent("./config/agents/Midori.json")
    ]
    
    # 创建世界模拟器
    world_simulator = WorldSimulator()
    
    # 注册所有代理
    all_agents = [expert_agent] + student_agents
    for agent in all_agents:
        world_simulator.register_agent(agent)
    
    # 将所有代理移动到同一位置以触发对话
    for agent in all_agents:
        world_simulator.move_agent(agent.name, "library")
    
    print(f"所有代理已移动到: library")
    print(f"在library的代理: {world_simulator.get_agents_at_location('library')}")
    
    # 手动触发一次交互
    print("\n触发交互...")
    world_simulator.handle_interactions("test_time_slot")
    
    # 检查是否生成了日志文件
    log_files = os.listdir("./log/")
    print(f"\n日志目录中的文件: {log_files}")
    
    if log_files:
        print("\n检查日志文件内容...")
        for log_file in log_files:
            if log_file.endswith('.json'):
                with open(f"./log/{log_file}", 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    print(f"日志文件 {log_file} 的内容:")
                    print(f"  位置: {log_data.get('location')}")
                    print(f"  话题: {log_data.get('topic')}")
                    print(f"  参与者: {log_data.get('participants')}")
                    print(f"  对话轮数: {len(log_data.get('dialogue_history', []))}")
                    if log_data.get('dialogue_history'):
                        print("  对话内容:")
                        for i, turn in enumerate(log_data['dialogue_history'][:3]):  # 只显示前3条
                            print(f"    {i+1}. {turn['speaker']}: {turn['message'][:100]}...")
    
    # 检查代理的记忆
    print("\n检查代理的记忆...")
    for agent in all_agents:
        memories = agent.long_term_memory.search_memories(limit=3)
        print(f"{agent.name} 的最近记忆:")
        for i, mem in enumerate(memories[:2]):  # 只显示前2条
            print(f"  {i+1}. {mem.get('content', '')[:100]}...")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_dialogue_features()