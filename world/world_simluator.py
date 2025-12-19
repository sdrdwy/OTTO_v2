import json
from typing import Dict, List, Any
from datetime import datetime
from world.calendar import Calendar
import os
from langchain_core.messages import SystemMessage

class WorldSimulator:
    def __init__(self, map_config_path: str = "./config/map.json", system_config_path: str = "./config/system.json"):
        with open(system_config_path, 'r') as f:
            self.system_config = json.load(f)
        
        with open(map_config_path, 'r') as f:
            self.map = json.load(f)
        
        self.agents = {}
        self.calendar = Calendar()
        self.current_day = 0
        self.current_time_slot_idx = 0
        self.time_slots = self.system_config['simulation']['time_slots']
        self.exam_questions = []
        self.exam_scores = {}
        self.total_days = self.system_config['simulation']['total_days']
        
        # Initialize agent positions in the map
        for location in self.map['locations'].values():
            location['agents'] = []
        
        # Initialize dialogue logging
        self.dialogue_log_path = "./log"
        os.makedirs(self.dialogue_log_path, exist_ok=True)

    def register_agent(self, agent):
        """Register an agent with the world"""
        self.agents[agent.name] = agent
        # Place agent at a default location initially
        if self.map['locations']:
            default_location = list(self.map['locations'].keys())[0]
            agent.set_location(default_location)
            self.map['locations'][default_location]['agents'].append(agent.name)

    def move_agent(self, agent_name: str, location: str) -> bool:
        """Move an agent to a specific location"""
        if agent_name not in self.agents:
            return False
        
        agent = self.agents[agent_name]
        current_location = agent.get_current_location()
        
        # Remove agent from current location
        if current_location and current_location in self.map['locations']:
            if agent_name in self.map['locations'][current_location]['agents']:
                self.map['locations'][current_location]['agents'].remove(agent_name)
        
        # Add agent to new location
        if location in self.map['locations']:
            self.map['locations'][location]['agents'].append(agent_name)
            agent.set_location(location)
            return True
        
        return False

    def get_agents_at_location(self, location: str) -> List[str]:
        """Get list of agent names at a specific location"""
        if location in self.map['locations']:
            return self.map['locations'][location]['agents'][:]
        return []

    def get_agent_by_name(self, name: str):
        """Get agent object by name"""
        return self.agents.get(name)

    def get_location_info(self, location: str) -> Dict[str, Any]:
        """Get information about a specific location"""
        return self.map['locations'].get(location, {})

    def display_world_state(self):
        """Display current world state including agent positions and locations"""
        print(f"\n=== 日期: {self.calendar.get_current_date_str()} ===")
        print("地图状态:")
        for loc_name, loc_info in self.map['locations'].items():
            agents_here = loc_info['agents']
            print(f"  {loc_name}: {loc_info['description']} | 人员: {agents_here}")
        print()

    def start_simulation(self, agents_list,is_exam=True):
        """Start the world simulation"""
        print("开始模拟世界运行...")
        
        # Register all agents
        for agent in agents_list:
            self.register_agent(agent)
        
        # Generate exam before simulation starts
        expert_agent = None
        for agent in agents_list:
            if hasattr(agent, 'is_expert') and agent.is_expert:
                expert_agent = agent
                break
        if is_exam:

            if expert_agent:
                num_questions = self.system_config['simulation']['exam_question_count']
                self.exam_questions = expert_agent.create_exam(num_questions)
                print(f"已生成考试题目 ({len(self.exam_questions)}题)")
                
                # Pre-simulation exam
                print("\n=== 模拟开始前考试 ===")
                for agent_name, agent in self.agents.items():
                    if not agent.is_expert:  # Only students take the exam
                        answers = agent.take_exam(self.exam_questions)
                        scores = expert_agent.grade_exam(agent_name, answers, self.exam_questions)
                        self.exam_scores[f"{agent_name}_pre"] = scores['total_score']
                        print(f"{agent_name} 考试成绩: {scores['total_score']:.1f}")
        
        # Run simulation for specified number of days
        for day in range(self.total_days):
            print(f"\n{'='*50}")
            print(f"第 {day+1} 天开始")
            print(f"{'='*50}")
            
            # Advance calendar to next day
            self.calendar.advance_day()
            
            # Display world state at the beginning of each day
            self.display_world_state()
            
            # Each day has multiple time slots
            for time_slot_idx, time_slot in enumerate(self.time_slots):
                print(f"\n--- 时间段: {time_slot} ---")
                
                # Get the global schedule for this time slot
                day_schedule = self.calendar.get_schedule_for_day()
                time_slot_schedule = day_schedule.get(time_slot, {})
                
                # Each agent creates their daily schedule and acts accordingly
                for agent_name, agent in self.agents.items():
                    # Create daily schedule based on global schedule, map info, and personal memories
                    personal_memories = agent.long_term_memory.search_memories(limit=5)
                    agent.create_daily_schedule(
                        date=self.calendar.get_current_date_str(),
                        world_map=self.map,
                        global_schedule=time_slot_schedule,
                        personal_memories=personal_memories
                    )
                    
                    # Get the action for this time slot
                    action = agent.get_action_for_time_slot(time_slot)
                    
                    # Move agent to designated location if different from current
                    if action['location'] != agent.get_current_location():
                        self.move_agent(agent_name, action['location'])
                    
                    print(f"{agent_name} 在 {action['location']} 进行 {action['activity']} (原因: {action['reason']})")
                    
                    # Generate memory of the activity
                    activity_memory = {
                        "type": "daily_activity",
                        "timestamp": str(datetime.now()),
                        "location": action['location'],
                        "activity": action['activity'],
                        "reason": action['reason'],
                        "summary": f"{agent_name}在{action['location']}进行了{action['activity']}"
                    }
                    agent.generate_memory(activity_memory)
                
                # Handle interactions between agents at the same location
                self.handle_interactions(time_slot)
                
                # Process any requests from agents
                self.process_agent_requests()
        
        if is_exam:

            # Post-simulation exam
            print(f"\n{'='*50}")
            print("模拟结束后考试")
            print(f"{'='*50}")
            
            if expert_agent:
                print("\n=== 模拟结束后考试 ===")
                for agent_name, agent in self.agents.items():
                    if not agent.is_expert:  # Only students take the exam
                        answers = agent.take_exam(self.exam_questions)
                        scores = expert_agent.grade_exam(agent_name, answers, self.exam_questions)
                        self.exam_scores[f"{agent_name}_post"] = scores['total_score']
                        print(f"{agent_name} 考试成绩: {scores['total_score']:.1f}")
            
            # Print final results
            print(f"\n{'='*50}")
            print("最终成绩对比")
            print(f"{'='*50}")
            for agent_name in [name for name in self.agents.keys() if not self.agents[name].is_expert]:
                pre_score = self.exam_scores.get(f"{agent_name}_pre", 0)
                post_score = self.exam_scores.get(f"{agent_name}_post", 0)
                improvement = post_score - pre_score
                print(f"{agent_name}: {pre_score:.1f} -> {post_score:.1f} (提升: {improvement:+.1f})")

    def handle_interactions(self, time_slot: str):
        """Handle interactions between agents at the same location"""
        # Group agents by location
        agents_by_location = {}
        for agent_name, agent in self.agents.items():
            location = agent.get_current_location()
            if location not in agents_by_location:
                agents_by_location[location] = []
            agents_by_location[location].append(agent_name)
        
        # Process interactions at each location
        for location, agent_names in agents_by_location.items():
            if len(agent_names) > 1:
                # Multiple agents at the same location - potentially start dialogues or battles
                print(f"  在 {location} 有 {len(agent_names)} 个代理: {agent_names}")
                
                # Determine if they should interact based on more sophisticated logic
                import random
                # Use a more sophisticated approach instead of pure randomness
                should_interact = self._should_agents_interact(agent_names, location)
                
                if should_interact:
                    # Start a multi-agent dialogue
                    max_rounds = self.system_config['simulation']['max_dialogue_rounds']
                    
                    # Select a topic for the dialogue based on agents' interests or recent activities
                    topic = self._select_dialogue_topic(agent_names, location)
                    
                    print(f"    开始关于 '{topic}' 的多轮对话...")
                    
                    # Have agents decide individually whether to join the dialogue based on their long-term memory and context
                    participating_agents = []
                    for agent_name in agent_names:
                        agent = self.agents[agent_name]
                        decision = agent.should_join_dialogue_based_on_context(
                            topic=topic,
                            participants=agent_names,
                            world_simulator=self,
                            location=location
                        )
                        
                        if decision["should_join"]:
                            participating_agents.append(agent_name)
                            print(f"      {agent_name} 决定加入对话 ({decision['reason']}, 置信度: {decision['confidence']:.2f})")
                        else:
                            print(f"      {agent_name} 决定不加入对话 ({decision['reason']})")
                    
                    # Only start dialogue if at least 2 agents are participating
                    if len(participating_agents) >= 2:
                        # Have the first participating agent initiate the dialogue
                        initiating_agent = self.agents[participating_agents[0]]
                        dialogue_history = initiating_agent.initiate_dialogue(
                            participants=participating_agents[1:],  # Others participate
                            topic=topic,
                            max_rounds=max_rounds,
                            world_simulator=self
                        )
                        
                        if dialogue_history:
                            print(f"    对话结束，共 {len(dialogue_history)} 轮")
                            
                            
                            # Create JSON log file for the dialogue
                            dialogue_log = {
                                "location": location,
                                "topic": topic,
                                "participants": participating_agents,
                                "time_slot": time_slot,
                                "date": self.calendar.get_current_date_str(),
                                "dialogue_history": dialogue_history,
                                "summary": f"关于'{topic}'的{len(dialogue_history)}轮对话"
                            }
                            
                            # Generate filename with timestamp
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            log_filename = f"dialogue_log_{location.replace(' ', '_')}_{timestamp_str}.json"
                            log_filepath = os.path.join(self.dialogue_log_path, log_filename)
                            
                            # Save dialogue to JSON file
                            with open(log_filepath, 'w', encoding='utf-8') as f:
                                json.dump(dialogue_log, f, ensure_ascii=False, indent=2)
                            
                            print(f"    对话日志已保存至: {log_filepath}")
                            
                            # Generate memory of the dialogue for each participating agent
                            for agent_name in participating_agents:
                                agent = self.agents[agent_name]
                                dialogue_memory = {
                                    "id": f"dialogue_{agent_name}_{topic}_{timestamp_str}",
                                    "type": "dialogue",
                                    "timestamp": str(datetime.now()),
                                    "content": f"参与了关于'{topic}'的对话，共{len(dialogue_history)}轮",
                                    "details": {
                                        "topic": topic,
                                        "participants": participating_agents,
                                        "location": location,
                                        "dialogue_log_file": log_filename,
                                        "dialogue_summary": [f"{turn['speaker']}: {turn['message'][:50]}..." for turn in dialogue_history]
                                    },
                                    "weight": 1.2
                                }
                                agent.long_term_memory.add_memory(dialogue_memory)
                            
                            # After dialogue, check if expert agent should teach based on the dialogue content
                            expert_agent = None
                            student_agents = []
                            for agent_name in participating_agents:
                                agent = self.agents[agent_name]
                                if agent.is_expert:
                                    expert_agent = agent
                                else:
                                    student_agents.append(agent)
                            
                            if expert_agent and student_agents:
                                # Expert may decide to teach based on the dialogue content
                                dialogue_content = " ".join([turn.get('message', '') for turn in dialogue_history if turn])
                                should_teach_decision = self._should_expert_teach(expert_agent, student_agents, dialogue_content, topic)
                                
                                if should_teach_decision["should_teach"]:
                                    print(f"    {expert_agent.name} 决定对学生进行教学: {should_teach_decision['reason']}")
                                    for student_agent in student_agents:
                                        teaching_content = expert_agent.teach(student_agent.name, topic)
                                        print(f"      教学内容: {teaching_content[:100]}...")
                        else:
                            print(f"    对话结束，但没有产生对话记录")
                    else:
                        print(f"    多轮对话未成功 - 只有 {len(participating_agents)} 人参与对话，至少需要2人")
                        # Record that a dialogue attempt failed
                        self._record_failed_dialogue(topic, agent_names, participating_agents, location)
                else:
                    print(f"    代理们决定不进行交互")
                    
                    # Even if they don't interact directly, expert may still teach or students may ask questions
                    expert_agent = None
                    student_agents = []
                    for agent_name in agent_names:
                        agent = self.agents[agent_name]
                        if agent.is_expert:
                            expert_agent = agent
                        else:
                            student_agents.append(agent)
                    
                    # If there's an expert with students, expert might initiate teaching
                    if expert_agent and student_agents:
                        should_teach_decision = self._should_expert_teach_randomly(expert_agent, student_agents, location)
                        if should_teach_decision["should_teach"]:
                            import random
                            topic = random.choice(["学习交流", "学术讨论", "知识讲解"])
                            print(f"    {expert_agent.name} 主动开始教学: {should_teach_decision['reason']}")
                            for student_agent in student_agents:
                                teaching_content = expert_agent.teach(student_agent.name, topic)
                                print(f"      教学内容: {teaching_content[:100]}...")
                                student_name = student_agent.name
                                dial_history = [{
                                    "message": teaching_content,
                                    "speaker": expert_agent.name,
                                }]
                                participant = [expert_agent.name,student_agent.name]
                                student_agent._generate_dialogue_memory(topic,dial_history,participant)
            elif len(agent_names) == 1:
                # Single agent at location - may still have individual activities
                agent_name = agent_names[0]
                agent = self.agents[agent_name]
                print(f"  在 {location} 只有 1 个代理: {agent_name}")
                
                # If it's the expert agent, they might review students' progress or prepare materials
                if agent.is_expert:
                    self._expert_single_activity(agent, location)
                else:
                    # If it's a student, they might study or ask questions based on their memories
                    self._student_single_activity(agent, location)
    
    def _should_agents_interact(self, agent_names: List[str], location: str) -> bool:
        """Determine if agents should interact based on location and agent types"""
        # For now, use a probability based on location type and agent personalities
        # This could be enhanced with more sophisticated logic
        import random
        
        # Check if location is a social location (higher chance of interaction)
        social_locations = ["咖啡厅", "公园", "休息室", "图书馆", "教室"]  # Common social locations
        is_social_location = any(social_loc in location for social_loc in social_locations)
        
        # Calculate base probability
        base_prob = 0.8 if is_social_location else 0.5  # Increased probabilities
        
        # Consider agent types (students more likely to interact with each other)
        expert_count = sum(1 for name in agent_names if self.agents[name].is_expert)
        student_count = len(agent_names) - expert_count
        
        # If there are students together, increase probability
        if student_count > 1:
            base_prob += 0.3  # Increased from 0.2 to 0.3
        
        # If there's an expert with students, interaction probability may change
        if expert_count > 0 and student_count > 0:
            base_prob += 0.2  # Increased from 0.1 to 0.2 - Teaching/learning opportunity
        
        return random.random() < base_prob
    
    def _select_dialogue_topic(self, agent_names: List[str], location: str) -> str:
        """Select a dialogue topic based on agents' interests and location"""
        import random
        
        # Get topics that might be relevant to the agents
        possible_topics = ["学习交流", "日常聊天", "学术讨论", "兴趣分享"]
        
        # Check recent memories of agents to find relevant topics
        all_memories = []
        for agent_name in agent_names:
            agent = self.agents[agent_name]
            recent_memories = agent.long_term_memory.search_memories(limit=3)
            all_memories.extend(recent_memories)
        
        # Look for topics in recent memories
        topic_keywords = []
        for memory in all_memories:
            content = memory.get('content', '')
            if '学习' in content:
                topic_keywords.append('学习交流')
            elif '课程' in content or '知识' in content:
                topic_keywords.append('学术讨论')
            elif '兴趣' in content:
                topic_keywords.append('兴趣分享')
        
        # If we found relevant topics in memories, use one of them
        if topic_keywords:
            return random.choice(topic_keywords + possible_topics)
        else:
            return random.choice(possible_topics)
    
    def _record_failed_dialogue(self, topic: str, all_agents: List[str], participating_agents: List[str], location: str):
        """Record when a dialogue fails to start due to insufficient participants"""
        print(f"    记录对话失败: 话题='{topic}', 所有在场代理={all_agents}, 参与代理={participating_agents}, 地点={location}")
        
        # Add this information to each agent's memory
        for agent_name in all_agents:
            agent = self.agents[agent_name]
            failed_dialogue_memory = {
                "id": f"failed_dialogue_{agent_name}_{topic}_{location}_{str(datetime.now())}",
                "type": "failed_dialogue",
                "timestamp": str(datetime.now()),
                "content": f"在{location}关于'{topic}'的对话尝试失败，只有{len(participating_agents)}/{len(all_agents)}人参与",
                "details": {
                    "topic": topic,
                    "location": location,
                    "all_agents": all_agents,
                    "participating_agents": participating_agents,
                    "reason": "参与人数不足"
                },
                "weight": 0.8
            }
            agent.long_term_memory.add_memory(failed_dialogue_memory)

    def _should_expert_teach(self, expert_agent, student_agents, dialogue_content, topic):
        """Determine if expert should teach based on dialogue content"""
        # Get recent memories of expert and students
        expert_memories = expert_agent.long_term_memory.search_memories(limit=3)
        student_memories = []
        for student in student_agents:
            student_memories.extend(student.long_term_memory.search_memories(limit=2))
        
        system_prompt = f"""
        你是{expert_agent.name}，一名专业教师，人设：{expert_agent.persona}。
        你的教学风格：{expert_agent.dialogue_style}。
        
        刚刚结束了一次关于"{topic}"的对话。
        对话内容摘要：{dialogue_content[:200]}
        
        参与对话的学生：{[s.name for s in student_agents]}
        
        你的近期记忆：{expert_memories}
        学生的近期记忆：{student_memories[:3]}
        
        基于对话内容和你的人设，判断你是否应该对学生进行教学。
        返回一个JSON格式的决策：
        {{
            "should_teach": true/false,
            "reason": "简短的解释原因"
        }}
        """
        
        try:
            response = expert_agent.llm.invoke([SystemMessage(content=system_prompt)])
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
                    "should_teach": len(dialogue_content) > 10,  # Teach if there was meaningful dialogue
                    "reason": "对话后教学的默认决策"
                }
        except Exception as e:
            print(f"Error determining if expert should teach: {e}")
            return {
                "should_teach": False,
                "reason": f"处理决策时出错: {e}"
            }

    def _should_expert_teach_randomly(self, expert_agent, student_agents, location):
        """Randomly determine if expert should initiate teaching"""
        import random
        # Higher chance to teach when with students
        should_teach = random.random() <= 1  # 70% chance when with students
        
        return {
            "should_teach": should_teach,
            "reason": f"在{location}与学生相遇，决定进行教学活动" if should_teach else "暂时不进行教学"
        }

    def _expert_single_activity(self, expert_agent, location):
        """Handle expert agent's activity when alone"""
        import random
        # Expert may review students' progress, prepare materials, or reflect on teaching
        activities = [
            "复习学生的学习进度",
            "准备教学材料", 
            "反思教学方法",
            "更新知识库"
        ]
        
        if random.random() < 0.6:  # 60% chance of activity
            activity = random.choice(activities)
            print(f"    {expert_agent.name} 正在{location}进行: {activity}")
            
            # Create memory of the activity
            activity_memory = {
                "id": f"expert_activity_{activity.replace(' ', '_')}_{str(datetime.now())}",
                "type": "expert_activity",
                "timestamp": str(datetime.now()),
                "content": f"在{location}进行了{activity}",
                "details": {
                    "activity": activity,
                    "location": location
                },
                "weight": 1.0
            }
            expert_agent.long_term_memory.add_memory(activity_memory)

    def _student_single_activity(self, student_agent, location):
        """Handle student agent's activity when alone"""
        import random
        # Student may study, review notes, or think about questions to ask
        activities = [
            "自主学习",
            "复习笔记", 
            "思考问题",
            "整理学习资料"
        ]
        
        if random.random() < 0.6:  # 60% chance of activity
            activity = random.choice(activities)
            print(f"    {student_agent.name} 正在{location}进行: {activity}")
            
            # Create memory of the activity
            activity_memory = {
                "id": f"student_activity_{activity.replace(' ', '_')}_{str(datetime.now())}",
                "type": "student_activity",
                "timestamp": str(datetime.now()),
                "content": f"在{location}进行了{activity}",
                "details": {
                    "activity": activity,
                    "location": location
                },
                "weight": 1.0
            }
            student_agent.long_term_memory.add_memory(activity_memory)
        
        # Check if there's an expert in the world and student might want to ask a question
        expert_agent = None
        for agent_name, agent in self.agents.items():
            if agent.is_expert:
                expert_agent = agent
                break
        
        # If there's an expert, student might want to ask for help on a topic
        if expert_agent and random.random() < 0.3:  # 30% chance to ask for help
            # Select a topic from student's recent memories or a general topic
            recent_memories = student_agent.long_term_memory.search_memories(limit=3)
            topic = "学习交流"  # Default topic
            if recent_memories:
                # Try to extract a topic from recent memories
                for memory in recent_memories:
                    content = memory.get('content', '')
                    if '学习' in content:
                        topic = "学习方法"
                        break
                    elif '知识' in content:
                        topic = "知识理解"
                        break
                    elif '考试' in content:
                        topic = "考试准备"
                        break
            
            print(f"    {student_agent.name} 决定向{expert_agent.name}寻求关于'{topic}'的帮助")
            help_result = student_agent.ask_teacher_for_help(expert_agent, topic)
            print(f"      问题: {help_result['question'][:100]}...")
            print(f"      回答: {help_result['teacher_response'][:100]}...")

    def process_agent_requests(self):
        """Process any requests from agents (like battles, etc.)"""
        # This method can be extended to handle various types of agent requests
        pass
