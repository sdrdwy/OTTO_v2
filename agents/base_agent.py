import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
try:
    from langchain_community.chat_models import ChatTongyi
except ImportError:
    try:
        from dashscope import Generation
        ChatTongyi = None
    except ImportError:
        raise ImportError("Either langchain-community or dashscope must be installed")
from langchain_core.messages import SystemMessage
from memory.conversation_mem import ConversationMemory
from memory.long_term_mem import LongTermMemory
import uuid


class BaseAgent:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.name = self.config["name"]
        self.persona = self.config["persona"]
        self.is_expert = self.config["is_expert"]
        self.dialogue_style = self.config["dialogue_style"]
        self.daily_habits = self.config["daily_habits"]
        self.max_dialogue_rounds = self.config["max_dialogue_rounds"]
        
        # Initialize LLM
        if ChatTongyi is not None:
            # Use langchain-community
            self.llm = ChatTongyi(
                model_name="qwen-max",
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", "sk-c763fc92bf8c46c7ae31639b05d89c96")
            )
        else:
            # Use dashscope directly
            from dashscope import Generation
            self.dashscope_generation = Generation
            self.llm = self  # Use self as a way to call our own method
        
        # Initialize memories
        self.conversation_memory = ConversationMemory()
        self.long_term_memory = LongTermMemory(f"./memory/{self.name}_long_term.jsonl")
        
        # Agent state
        self.current_location = None
        self.current_schedule = {}
        self.personal_calendar = {}
        
    def set_location(self, location: str):
        """Set agent's current location"""
        self.current_location = location
    
    def get_current_location(self):
        """Get agent's current location"""
        return self.current_location
    
    def create_daily_schedule(self, date: str, world_map: Dict, global_schedule: Dict, personal_memories: List[Dict]):
        """Create daily schedule based on persona, global schedule, map info, and personal memories"""
        system_prompt = f"""
        你是{self.name}，人设：{self.persona}。
        你的对话风格：{self.dialogue_style}。
        你的日常习惯：{self.daily_habits}。
        
        请根据以下信息制定今天的日程安排：
        - 全局日程：{global_schedule}
        - 当前地图信息：{world_map}
        - 个人记忆：{personal_memories}
        
        请返回一个包含以下时间段的日程安排的JSON格式：
        {{
            "morning_1": {{"activity": "活动名称", "location": "地点", "reason": "原因"}},
            "morning_2": {{"activity": "活动名称", "location": "地点", "reason": "原因"}},
            "afternoon_1": {{"activity": "活动名称", "location": "地点", "reason": "原因"}},
            "afternoon_2": {{"activity": "活动名称", "location": "地点", "reason": "原因"}},
            "evening": {{"activity": "活动名称", "location": "地点", "reason": "原因"}}
        }}
        """
        
        try:
            response = self.invoke_llm([SystemMessage(content=system_prompt)])
            schedule_text = response.content
            
            # Extract JSON from response
            start_idx = schedule_text.find('{')
            end_idx = schedule_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = schedule_text[start_idx:end_idx]
                self.current_schedule = json.loads(json_str)
            else:
                # Fallback to default schedule if parsing fails
                self.current_schedule = {
                    "morning_1": {"activity": "自习", "location": "图书馆", "reason": "默认安排"},
                    "morning_2": {"activity": "课程", "location": "教室", "reason": "默认安排"},
                    "afternoon_1": {"activity": "自由活动", "location": "公园", "reason": "默认安排"},
                    "afternoon_2": {"activity": "自由活动", "location": "咖啡厅", "reason": "默认安排"},
                    "evening": {"activity": "自由活动", "location": "公园", "reason": "默认安排"}
                }
        except Exception as e:
            print(f"Error creating daily schedule for {self.name}: {e}")
            self.current_schedule = {
                "morning_1": {"activity": "自习", "location": "图书馆", "reason": "默认安排"},
                "morning_2": {"activity": "课程", "location": "教室", "reason": "默认安排"},
                "afternoon_1": {"activity": "自由活动", "location": "公园", "reason": "默认安排"},
                "afternoon_2": {"activity": "自由活动", "location": "咖啡厅", "reason": "默认安排"},
                "evening": {"activity": "自由活动", "location": "公园", "reason": "默认安排"}
            }
        
        return self.current_schedule
    
    def move_to_location(self, world_simulator, location: str):
        """Request to move to a specific location in the world"""
        return world_simulator.move_agent(self.name, location)
    
    def initiate_dialogue(self, participants: List[str], topic: str, max_rounds: int, world_simulator):
        """Initiate a multi-round dialogue with other agents"""
        dialogue_history = []
        
        # Include self in the participants list for a complete dialogue
        all_participants = [self.name] + participants
        available_participants = all_participants
        print(available_participants)
        for round_num in range(max_rounds):
            temp_participants=[]
            # Determine which participants are available and willing to join this round
            for participant_name in available_participants:
                # Check if agent is willing to continue (using the enhanced context-based method)
                participant_agent = world_simulator.get_agent_by_name(participant_name)
                if participant_agent:
                    decision = participant_agent._should_continue_dialogue(
                        other_agent=available_participants,
                        con_memory = dialogue_history
                    )
                    
                    if decision["should_join"]:
                        temp_participants.append(participant_name)
            available_participants = temp_participants.copy()
            # Only continue if we have at least 2 participants
            if len(available_participants) < 2:
                print(f"    对话在第 {round_num+1} 轮结束 - 只有 {len(available_participants)} 人参与")
                break
            
            # Generate dialogue turns for each available participant
            for participant_name in available_participants:
                participant_agent = world_simulator.get_agent_by_name(participant_name)
                
                # Skip if this is the initiating agent's turn (we'll handle it separately)
                if participant_name == self.name:
                    continue
                
                # Generate the participant's turn
                participant_turn = participant_agent._generate_dialogue_turn(
                    topic=topic, 
                    history=dialogue_history, 
                    participants=available_participants
                )
                
                if participant_turn:
                    dialogue_history.append(participant_turn)
                    
                    # Add to conversation memory
                    participant_agent.conversation_memory.add_dialogue_turn(
                        participant_agent.name, 
                        topic, 
                        participant_turn
                    )
            
            # Generate turn for the initiating agent
            initiating_turn = self._generate_dialogue_turn(
                topic=topic, 
                history=dialogue_history, 
                participants=available_participants
            )
            
            if initiating_turn:
                dialogue_history.append(initiating_turn)
                
                # Add to conversation memory
                self.conversation_memory.add_dialogue_turn(
                    self.name, 
                    topic, 
                    initiating_turn
                )
        
        # Generate a memory of the dialogue event if it has content
        if dialogue_history:
            # Generate memory based on the dialogue content and existing memories
            self._generate_dialogue_memory(topic, dialogue_history, all_participants)
            
            # Have other participants also generate memories
            for participant_name in participants:
                participant_agent = world_simulator.get_agent_by_name(participant_name)
                if participant_agent:
                    participant_agent._generate_dialogue_memory(topic, dialogue_history, all_participants)
        
        return dialogue_history
    
    def _generate_dialogue_memory(self, topic: str, dialogue_history: List[Dict], participants: List[str]):
        """Generate a memory of the dialogue based on the content and existing memories"""
        # Create a summary of the dialogue
        dialogue_summary = f"参与了关于'{topic}'的对话，参与者: {', '.join(participants)}"
        
        # Extract key points from the dialogue
        key_points = []
        for turn in dialogue_history:
            message = turn.get('message', '')
            if len(message) > 0:
                # Extract key information from the message
                key_points.append(f"{turn['speaker']}: {message[:100]}...")  # Take first 100 chars
        
        # Get relevant memories to contextualize this dialogue
        relevant_memories = self.long_term_memory.search_memories(query=topic, limit=3)
        
        # Create a more meaningful memory based on the dialogue content
        if key_points:
            # Use LLM to generate a more meaningful memory if we have content
            system_prompt = f"""
            你是{self.name}，人设：{self.persona}。
            
            你刚刚参与了一个关于"{topic}"的对话，参与者包括: {participants}。
            对话内容摘要: {key_points}
            
            请根据这个对话内容和你的人设，生成一个简洁但有意义的记忆记录，
            描述这次对话的主要内容和你的收获或感受。
            记忆应该包含: 
            - 对话主题
            - 重要的讨论点
            - 你的感受或收获
            - 与你已有知识或经历的联系
            
            请返回一个简短但内容丰富的记忆描述。
            """
            
            try:
                response = self.invoke_llm([SystemMessage(content=system_prompt)])
                memory_content = response.content
            except Exception as e:
                # Fallback to a simple summary if LLM call fails
                memory_content = f"参与了关于'{topic}'的对话，与{', '.join(participants)}讨论了相关内容"
        else:
            # If no key points, create a basic memory
            memory_content = f"参与了关于'{topic}'的对话，参与者: {', '.join(participants)}，但对话内容未记录"
        
        # Create the memory object
        memory = {
            "id": f"dialogue_{self.name}_{topic}_{str(uuid.uuid4())[:8]}_{str(datetime.now())}",
            "type": "dialogue",
            "timestamp": str(datetime.now()),
            "content": memory_content,
            "details": {
                "topic": topic,
                "participants": participants,
                "dialogue_summary": key_points,
                "location": self.current_location
            },
            "weight": 1.2
        }
        
        # Add the memory to long-term memory
        self.long_term_memory.add_memory(memory)
        return memory

    def _should_continue_dialogue(self,other_agent,con_memory):
        current_topic = "general"  # This would be passed from the calling function
        if hasattr(self, '_current_dialogue_topic'):
            current_topic = self._current_dialogue_topic
        # print(current_topic)
        system_prompt = f"""
        你是{self.name}，人设：{self.persona}。
        你的对话风格：{self.dialogue_style}。
        你的日常习惯：{self.daily_habits}。
        
        当前话题是：{current_topic}
        当前在场的参与者：{[agent for agent in [other_agent]] if other_agent else []}
        
        你的对话当前对话记忆：{con_memory}
        
        请根据你的人设、记忆和当前情况，判断你是否应该继续这个对话。
        返回一个JSON格式的决策：
        {{
            "should_join": true/false,
            "reason": "简短的解释原因",
            "confidence": 0.0-1.0之间的置信度
        }}
        """

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            response_text = response.content
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                decision = json.loads(json_str)
                # print(decision)
                return decision # .get('should_join', False)
            else:
                # If parsing fails, use a simple heuristic
                return len(con_memory) > 0  # Join if agent has recent memories
        except Exception as e:
            print(f"Error determining if {self.name} should join dialogue: {e}")
            return False  # Default to not joining if there's an error
    
    def _should_join_dialogue(self, other_agent):
        """Determine if agent should join a dialogue based on persona, long-term memory and current situation"""
        # Get recent memories to understand context
        recent_memories = self.long_term_memory.search_memories(limit=5)
        
        # Get current topic if available
        current_topic = "general"  # This would be passed from the calling function
        if hasattr(self, '_current_dialogue_topic'):
            current_topic = self._current_dialogue_topic
        
        # Create a prompt to decide whether to join the dialogue
        system_prompt = f"""
        你是{self.name}，人设：{self.persona}。
        你的对话风格：{self.dialogue_style}。
        你的日常习惯：{self.daily_habits}。
        
        当前话题是：{current_topic}
        当前在场的参与者：{[agent.name for agent in [other_agent]] if other_agent else []}
        
        你的近期记忆：{recent_memories}
        
        请根据你的人设、记忆和当前情况，判断你是否应该参与这个对话。
        返回一个JSON格式的决策：
        {{
            "should_join": true/false,
            "reason": "简短的解释原因",
            "confidence": 0.0-1.0之间的置信度
        }}
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            response_text = response.content
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                decision = json.loads(json_str)
                return decision.get('should_join', False)
            else:
                # If parsing fails, use a simple heuristic
                return len(recent_memories) > 0  # Join if agent has recent memories
        except Exception as e:
            print(f"Error determining if {self.name} should join dialogue: {e}")
            return False  # Default to not joining if there's an error
    
    def should_join_dialogue_based_on_context(self, topic: str, participants: List[str], world_simulator, location: str):
        """Enhanced method to determine if agent should join dialogue based on full context"""
        self._current_dialogue_topic = topic
        
        # Get recent memories relevant to the topic
        topic_memories = self.long_term_memory.search_memories(query=topic, limit=3)
        all_memories = self.long_term_memory.search_memories(limit=5)
        
        # Get information about other participants
        other_agents_info = []
        for participant_name in participants:
            if participant_name != self.name:
                other_agent = world_simulator.get_agent_by_name(participant_name)
                if other_agent:
                    other_agents_info.append({
                        "name": other_agent.name,
                        "persona": other_agent.persona,
                        "relationship": self._assess_relationship(other_agent)
                    })
        
        system_prompt = f"""
        你是{self.name}，人设：{self.persona}。
        你的对话风格：{self.dialogue_style}。
        你的日常习惯：{self.daily_habits}。
        
        当前情况：
        - 地点：{location}
        - 话题：{topic}
        - 参与者：{participants}
        - 其他参与者信息：{other_agents_info}
        
        相关记忆：
        - 话题相关记忆：{topic_memories}
        - 近期记忆：{all_memories}
        
        请根据你的人设、记忆、话题相关性、其他参与者和当前环境，判断你是否应该参与这个对话。
        返回一个JSON格式的决策：
        {{
            "should_join": true/false,
            "reason": "简短的解释原因",
            "confidence": 0.0-1.0之间的置信度
        }}
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            response_text = response.content
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                decision = json.loads(json_str)
                return decision
            else:
                # If parsing fails, use a simple heuristic
                return {
                    "should_join": len(topic_memories) > 0,  # Join if there are topic-related memories
                    "reason": "基于话题相关记忆的默认决策",
                    "confidence": 0.5
                }
        except Exception as e:
            print(f"Error determining if {self.name} should join dialogue with context: {e}")
            return {
                "should_join": False,
                "reason": f"处理决策时出错: {e}",
                "confidence": 0.0
            }
    
    def _assess_relationship(self, other_agent):
        """Assess the relationship with another agent based on memories"""
        # Search for memories involving the other agent
        relationship_memories = self.long_term_memory.search_memories(
            query=other_agent.name, 
            limit=3
        )
        
        if not relationship_memories:
            return "unknown"  # No prior interaction
        
        # Determine relationship based on memory types and content
        positive_interactions = [m for m in relationship_memories if any(pos in str(m.get('content', '')) for pos in ['合作', '帮助', '友好', 'positive', 'good'])]
        negative_interactions = [m for m in relationship_memories if any(neg in str(m.get('content', '')) for neg in ['冲突', '矛盾', 'negative', 'bad', 'disagreement'])]
        
        if len(positive_interactions) > len(negative_interactions):
            return "positive"
        elif len(negative_interactions) > len(positive_interactions):
            return "negative"
        else:
            return "neutral"
    
    def _generate_dialogue_turn(self, topic: str, history: List[Dict], participants: List[str]):
        """Generate a dialogue turn"""
        system_prompt = f"""
        你是{self.name}，人设：{self.persona}。
        你的对话风格：{self.dialogue_style}。
        当前话题：{topic}
        对话历史：{history}
        参与者：{participants}
        
        请生成你的一句话回应，保持符合你的角色设定。
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            print(f"{self.name}: {response.content}")
            return {
                "speaker": self.name,
                "message": response.content,
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            return {
                "speaker": self.name,
                "message": f"无法生成回应: {e}",
                "timestamp": str(datetime.now())
            }
    
    def participate_in_dialogue(self, topic: str, history: List[Dict], participants: List[str], max_rounds: int):
        """Participate in an ongoing dialogue"""
        if not self._should_join_dialogue(self):
            return None
            
        return self._generate_dialogue_turn(topic, history, participants)
    
    def start_battle(self, opponent: str, world_simulator):
        """Initiate a battle with another agent"""
        # Simulate battle
        battle_result = self._simulate_battle(opponent)
        
        # Create long-term memory of the battle
        battle_memory = {
            "id": str(uuid.uuid4()),
            "type": "battle",
            "timestamp": str(datetime.now()),
            "participants": [self.name, opponent],
            "result": battle_result,
            "summary": f"与{opponent}的战斗结果：{battle_result}"
        }
        
        self.long_term_memory.add_memory(battle_memory)
        
        return battle_result
    
    def _simulate_battle(self, opponent: str):
        """Simulate a battle and return result"""
        # Simple simulation - can be enhanced
        import random
        outcomes = ["胜利", "失败", "平局"]
        return random.choice(outcomes)
    
    def generate_memory(self, event: Dict):
        """Generate a memory from an event, using LLM to create meaningful content based on existing memories"""
        # Get recent memories to provide context
        recent_memories = self.long_term_memory.search_memories(limit=5)
        
        # Create a more meaningful memory using LLM if possible
        event_location = event.get('location', '未知地点')
        event_activity = event.get('activity', '未知活动')
        event_result = event.get('result', '未记录')
        
        # Use LLM to generate a more meaningful memory if we have context
        system_prompt = f"""
        你是{self.name}，人设：{self.persona}。
        
        你刚刚经历了一个事件:
        - 地点: {event_location}
        - 活动: {event_activity}
        - 结果: {event_result}
        
        你的近期记忆: {recent_memories}
        
        请根据这个事件和你的人设，生成一个有意义的记忆描述。
        这个记忆应该:
        1. 与你的个性和经历相符
        2. 反映事件的重要性和意义
        3. 包含你对事件的感受或思考
        4. 与你之前的记忆有所关联
        
        请返回一个简洁但内容丰富的记忆描述。
        """
        
        try:
            response = self.invoke_llm([SystemMessage(content=system_prompt)])
            memory_content = response.content
        except Exception as e:
            # Fallback to simple concatenation if LLM call fails
            memory_content = f"{self.name}在{event_location}进行了{event_activity}，结果是{event_result}"
        
        memory = {
            "id": str(uuid.uuid4()),
            "type": event.get("type", "general"),
            "timestamp": str(datetime.now()),
            "content": memory_content,
            "event": event,
            "details": {
                "location": event_location,
                "activity": event_activity,
                "result": event_result,
                "context_memories": [mem.get('content', '')[:100] for mem in recent_memories[:3]]  # Include context from recent memories
            },
            "weight": 1.0
        }
        
        self.long_term_memory.add_memory(memory)
        return memory
    
    def get_action_for_time_slot(self, time_slot: str):
        """Get the action for a specific time slot from current schedule"""
        return self.current_schedule.get(time_slot, {
            "activity": "自由活动", 
            "location": self.current_location or "未知地点", 
            "reason": "未安排"
        })
    
    def invoke_llm(self, messages):
        """Unified method to invoke LLM using either langchain-community or dashscope"""
        if ChatTongyi is not None:
            # Using langchain-community
            return self.llm.invoke(messages)
        else:
            # Using dashscope directly
            from dashscope import Generation
            import os
            api_key = os.getenv("DASHSCOPE_API_KEY", "sk-c763fc92bf8c46c7ae31639b05d89c96")
            
            # Extract the system message content
            system_prompt = messages[0].content if messages else ""
            
            try:
                response = Generation.call(
                    model="qwen-max",
                    api_key=api_key,
                    prompt=system_prompt,
                    result_format='message'
                )
                
                # Create a mock response object with content attribute
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                
                if response.status_code == 200:
                    content = response.output.get('choices', [{}])[0].get('message', {}).get('content', str(response))
                    return MockResponse(content)
                else:
                    return MockResponse(f"Error calling LLM: {response.code} - {response.message}")
            except Exception as e:
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                return MockResponse(f"Error calling LLM: {str(e)}")