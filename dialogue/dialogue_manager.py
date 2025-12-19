import json
import os
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import SystemMessage
from agents.expert_agent import ExpertAgent


class DialogueManager:
    def __init__(self, world_simulator):
        self.world_simulator = world_simulator
        self.dialogue_templates = self._load_dialogue_templates()
    
    def _load_dialogue_templates(self):
        """Load dialogue templates from the prompts directory"""
        templates = {}
        template_path = os.path.join(os.path.dirname(__file__), "../prompts/dialogue_template.txt")
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                templates['structured_dialogue'] = f.read()
        else:
            # Default template if file not found
            templates['structured_dialogue'] = """
你正在参与一场围绕【{topic}】的多轮对话。
背景信息：
- 你的身份：{agent_persona}
- 今日目标：{daily_goal}
- 相关知识：{kb_snippet}
- 先前记忆：{relevant_memory}
- 对话目标：{dialogue_goal}

请基于以上内容，提出**具体问题**、分享**具体见解**，或**请求协作**。避免泛泛而谈。
当前对话历史：
{history}

现在轮到你发言，请保持专业、具体、有推进性。
"""
        return templates
    
    def run_structured_dialogue(self, initiating_agent, participants: List[str], topic: str, max_rounds: int = 5):
        """Run a structured dialogue with specific context grounding"""
        # Get all participant agents
        participant_agents = [self.world_simulator.agents[name] for name in participants]
        all_agents = [initiating_agent] + participant_agents
        
        # Prepare initial context for the dialogue
        dialogue_context = self._prepare_dialogue_context(topic, all_agents)
        
        # Initialize dialogue history
        dialogue_history = []
        
        # Create the initial message from the initiating agent
        initial_message = self._generate_agent_response(
            agent=initiating_agent,
            topic=topic,
            context=dialogue_context,
            history=[],
            all_agents=all_agents
        )
        
        if initial_message:
            dialogue_history.append({
                "speaker": initiating_agent.name,
                "message": initial_message,
                "timestamp": str(datetime.now()),
                "topic": topic
            })
        
        # Run the dialogue for max_rounds or until terminated early
        for round_num in range(1, max_rounds):
            # Each agent takes a turn (in order)
            for agent in all_agents:
                # Check if dialogue should terminate
                if self._should_terminate_dialogue(dialogue_history, topic):
                    print(f"    对话因终止条件结束，共 {len(dialogue_history)} 轮")
                    return dialogue_history
                
                # Prepare context for this agent's turn
                agent_context = self._prepare_agent_context(agent, topic, all_agents)
                
                # Generate agent's response
                response = self._generate_agent_response(
                    agent=agent,
                    topic=topic,
                    context=agent_context,
                    history=dialogue_history,
                    all_agents=all_agents
                )
                
                if response and agent.name != initiating_agent.name:  # Don't duplicate initiating agent's message
                    dialogue_history.append({
                        "speaker": agent.name,
                        "message": response,
                        "timestamp": str(datetime.now()),
                        "topic": topic
                    })
                    
                    # Limit dialogue length to prevent infinite loops
                    if len(dialogue_history) >= max_rounds * len(all_agents):
                        break
            
            # Check again after each full round
            if self._should_terminate_dialogue(dialogue_history, topic):
                print(f"    对话因终止条件结束，共 {len(dialogue_history)} 轮")
                return dialogue_history
        
        print(f"    对话达到最大轮数限制，共 {len(dialogue_history)} 轮")
        return dialogue_history
    
    def _prepare_dialogue_context(self, topic: str, all_agents: List[Any]):
        """Prepare overall dialogue context"""
        # Find relevant knowledge base content
        kb_content = ""
        expert_agent = None
        for agent in all_agents:
            if hasattr(agent, 'is_expert') and agent.is_expert:
                expert_agent = agent
                break
        
        if expert_agent:
            topic_knowledge = expert_agent.get_kb_content_by_topic(topic)
            if topic_knowledge:
                # Format all fields of the knowledge base entry
                kb_content = "\n".join(f"【{k}】{v}" for k, v in topic_knowledge.items() if v and k != "id")
        
        # Gather relevant memories from all agents
        all_relevant_memories = []
        for agent in all_agents:
            agent_memories = agent.long_term_memory.get_memories_by_topic(topic, limit=2)
            all_relevant_memories.extend(agent_memories)
        
        # Get daily goals for all agents
        daily_goals = [f"{agent.name}: {getattr(agent, 'daily_goal', '无特定目标')}" for agent in all_agents]
        
        return {
            "topic": topic,
            "kb_content": kb_content,
            "all_relevant_memories": all_relevant_memories,
            "daily_goals": daily_goals,
            "participants": [agent.name for agent in all_agents]
        }
    
    def _prepare_agent_context(self, agent, topic: str, all_agents: List[Any]):
        """Prepare context specifically for one agent"""
        # Get agent-specific relevant memories
        agent_memories = agent.long_term_memory.get_memories_by_topic(topic, limit=3)
        memory_content = [mem.get('content', '') for mem in agent_memories]
        
        # Get relevant knowledge base content
        kb_content = ""
        expert_agent = None
        for ag in all_agents:
            if hasattr(ag, 'is_expert') and ag.is_expert:
                expert_agent = ag
                break
        
        if expert_agent:
            topic_knowledge = expert_agent.get_kb_content_by_topic(topic)
            if topic_knowledge:
                # Format all fields of the knowledge base entry
                kb_content = "\n".join(f"【{k}】{v}" for k, v in topic_knowledge.items() if v and k != "id")
        
        # Get other agents' recent contributions to this topic
        other_agents_memories = []
        for ag in all_agents:
            if ag != agent:
                ag_memories = ag.long_term_memory.get_memories_by_topic(topic, limit=2)
                other_agents_memories.extend([mem.get('content', '') for mem in ag_memories])
        
        return {
            "agent_persona": agent.persona,
            "daily_goal": getattr(agent, 'daily_goal', '无特定目标'),
            "kb_snippet": kb_content,
            "relevant_memory": memory_content,
            "other_agents_contributions": other_agents_memories,
            "topic": topic,
            "dialogue_goal": f"深入探讨{topic}的相关问题"
        }
    
    def _generate_agent_response(self, agent, topic: str, context: Dict, history: List[Dict], all_agents: List[Any]):
        """Generate a response from an agent using structured context"""
        # Format the dialogue history for the prompt
        history_text = ""
        for turn in history[-3:]:  # Use last 3 turns to keep prompt manageable
            history_text += f"{turn['speaker']}: {turn['message']}\n"
        
        # Fill in the dialogue template
        template = self.dialogue_templates['structured_dialogue']
        prompt = template.format(
            topic=context.get('topic', topic),
            agent_persona=context.get('agent_persona', agent.name),
            daily_goal=context.get('daily_goal', '无特定目标'),
            kb_snippet=context.get('kb_snippet', ''),
            relevant_memory=context.get('relevant_memory', []),
            dialogue_goal=context.get('dialogue_goal', f'讨论{topic}'),
            history=history_text
        )
        
        try:
            response = agent.llm.invoke([SystemMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"生成{agent.name}的对话回应时出错: {e}")
            # Fallback: simple response
            return f"关于{topic}，我认为我们需要进一步讨论。"
    
    def _should_terminate_dialogue(self, dialogue_history: List[Dict], topic: str) -> bool:
        """Check if dialogue should be terminated based on termination conditions"""
        if not dialogue_history:
            return False
        
        # Check if dialogue is too long
        if len(dialogue_history) >= 20:  # Maximum 20 turns to prevent infinite loops
            return True
        
        # Check if last few messages indicate agreement or conclusion
        recent_messages = dialogue_history[-3:] if len(dialogue_history) >= 3 else dialogue_history
        for turn in recent_messages:
            message = turn.get('message', '').lower()
            if '结束' in message or '完成' in message or '同意' in message or '明白了' in message:
                return True
        
        return False


def run_dialogue_with_context(world_simulator, initiating_agent, participants: List[str], topic: str, max_rounds: int = 5):
    """Convenience function to run a contextual dialogue"""
    dialogue_manager = DialogueManager(world_simulator)
    return dialogue_manager.run_structured_dialogue(
        initiating_agent=initiating_agent,
        participants=participants,
        topic=topic,
        max_rounds=max_rounds
    )