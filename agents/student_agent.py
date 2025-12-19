import json
from typing import Dict, List, Any
from datetime import datetime
from langchain_core.messages import SystemMessage
from agents.base_agent import BaseAgent


class StudentAgent(BaseAgent):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        # Students don't have a knowledge base
        self.knowledge_base = []
        self.curriculum = {}
    
    def ask_question(self, teacher_name: str, topic: str, question: str):
        """Ask a question to the teacher"""
        # Get relevant memories related to the topic
        relevant_memories = self.long_term_memory.get_memories_by_topic(topic, limit=3)
        
        memory_context = ""
        if relevant_memories:
            memory_context = "相关学习记忆：\n"
            for mem in relevant_memories:
                memory_context += f"- {mem.get('content', '')}\n"
        
        system_prompt = f"""
你是{self.name}，一个学生，人设：{self.persona}。
你想向{teacher_name}老师询问关于"{topic}"的问题："{question}"

{memory_context}

请以学生的身份提出问题，保持符合你的角色设定，并基于你已有的知识和记忆。
"""

        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            question_content = response.content
            
            # Generate memory of asking question
            question_memory = {
                "id": f"ask_question_{topic}_{datetime.now().isoformat()}",
                "type": "question_asked",
                "timestamp": str(datetime.now()),
                "content": f"向{teacher_name}询问了关于{topic}的问题",
                "details": {
                    "teacher": teacher_name,
                    "topic": topic,
                    "question": question_content,
                    "context_memories": [mem.get("id") for mem in relevant_memories]
                },
                "weight": 1.0
            }
            
            self.long_term_memory.add_memory(question_memory)
            
            return question_content
        except Exception as e:
            error_msg = f"提问过程中出现错误: {e}"
            print(error_msg)
            return error_msg
    
    def study_topic(self, topic: str, study_materials: List[str] = None):
        """Study a specific topic"""
        if study_materials is None:
            study_materials = []
        
        # Get relevant memories related to the topic
        relevant_memories = self.long_term_memory.get_memories_by_topic(topic, limit=3)
        
        memory_context = ""
        if relevant_memories:
            memory_context = "相关学习记忆：\n"
            for mem in relevant_memories:
                memory_context += f"- {mem.get('content', '')}\n"
        
        system_prompt = f"""
你是{self.name}，一个学生，人设：{self.persona}。
你正在学习"{topic}"这个主题。
学习材料：{study_materials}

{memory_context}

请描述你的学习过程和收获，保持符合你的角色设定，并将新知识与已有记忆联系起来。
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            study_result = response.content
            
            # Generate memory of studying
            study_memory = {
                "id": f"study_{topic}_{datetime.now().isoformat()}",
                "type": "studying",
                "timestamp": str(datetime.now()),
                "content": f"学习了{topic}的内容",
                "details": {
                    "topic": topic,
                    "result": study_result,
                    "materials": study_materials,
                    "context_memories": [mem.get("id") for mem in relevant_memories]
                },
                "weight": 1.0
            }
            
            self.long_term_memory.add_memory(study_memory)
            
            return study_result
        except Exception as e:
            error_msg = f"学习过程中出现错误: {e}"
            print(error_msg)
            return error_msg
    
    def take_exam(self, exam_questions: List[Dict]):
        """Take an exam and return answers using LLM"""
        answers = []
        
        for i, question in enumerate(exam_questions):
            question_text = question['question']
            topic = question.get('topic', '通用')
            
            # Get relevant memories related to the topic
            relevant_memories = self.long_term_memory.get_memories_by_topic(topic, limit=5)
            
            memory_context = ""
            if relevant_memories:
                memory_context = "相关学习记忆：\n"
                for mem in relevant_memories:
                    memory_context += f"- {mem.get('content', '')}\n"
            
            # Use LLM to generate a proper answer based on the question
            system_prompt = f"""
                你是{self.name}，一个学生，人设：{self.persona}。

                请回答以下考试题目：
                题目：{question_text}
                主题：{topic}

                {memory_context}

                请提供一个详细且准确的答案，保持符合你的角色设定，并基于你的学习记忆回答。
                """
            
            try:
                response = self.llm.invoke([SystemMessage(content=system_prompt)])
                answer = response.content
            except Exception as e:
                print(f"使用LLM生成答案失败: {e}")
                # Fallback to basic answer
                answer = f"对于这个问题，我的回答是关于{topic}的内容。"
            
            answers.append({
                "question_idx": i,
                "question": question['question'],
                "answer": answer,
                "topic": topic
            })
        
        # Generate memory of taking exam
        exam_memory = {
            "id": f"take_exam_{datetime.now().isoformat()}",
            "type": "exam_taken",
            "timestamp": str(datetime.now()),
            "content": f"参加了考试，回答了{len(answers)}道题",
            "details": {
                "answers": answers,
                "question_topics": [q['topic'] for q in answers]
            },
            "weight": 1.8
        }
        
        self.long_term_memory.add_memory(exam_memory)
        
        return answers

    def ask_teacher_for_help(self, teacher, topic: str):
        """Ask the teacher for help on a specific topic using LLM"""
        # Get recent memories to understand what the student is struggling with
        recent_memories = self.long_term_memory.search_memories(limit=5)
        
        # Get specific memories related to the topic
        topic_memories = self.long_term_memory.get_memories_by_topic(topic, limit=3)
        
        system_prompt = f"""
你是{self.name}，一个学生，人设：{self.persona}。

你对"{topic}"这个主题有疑问，需要向老师寻求帮助。
你的近期学习记忆：{[mem.get('content', '') for mem in recent_memories]}
关于{topic}的特定记忆：{[mem.get('content', '') for mem in topic_memories]}

请根据你的学习情况和人设，向老师提出一个具体的学习问题。
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            question = response.content
            
            # Ask the teacher the question
            teacher_response = teacher.answer_question(self, question)
            
            # Create memory of asking for help
            help_memory = {
                "id": f"help_request_{topic}_{datetime.now().isoformat()}",
                "type": "help_request",
                "timestamp": str(datetime.now()),
                "content": f"向老师寻求关于{topic}的帮助",
                "details": {
                    "topic": topic,
                    "question": question,
                    "teacher_response": teacher_response,
                    "context_memories": [mem.get("id") for mem in recent_memories]
                },
                "weight": 1.3
            }
            
            self.long_term_memory.add_memory(help_memory)
            
            return {
                "question": question,
                "teacher_response": teacher_response
            }
        except Exception as e:
            error_msg = f"请求帮助过程中出现错误: {e}"
            print(error_msg)
            return {
                "question": f"我对{topic}有疑问",
                "teacher_response": "请求失败"
            }
    
    def initiate_dialogue(self, participants: List[str], topic: str = None, max_rounds: int = 5, world_simulator=None):
        """Initiate a dialogue with grounding in specific context"""
        if not participants:
            return []
        
        # Get relevant memories related to the topic
        relevant_memories = self.long_term_memory.get_memories_by_topic(topic or "general", limit=3) if topic else []
        
        # Get knowledge from world simulator if available
        kb_context = ""
        if world_simulator:
            expert_agent = None
            for agent_name in world_simulator.agents:
                agent = world_simulator.agents[agent_name]
                if hasattr(agent, 'is_expert') and agent.is_expert:
                    expert_agent = agent
                    break
            
            if expert_agent and topic:
                topic_knowledge = expert_agent.get_kb_content_by_topic(topic)
                if topic_knowledge:
                    kb_context = f"【{topic_knowledge['topic']}】{topic_knowledge['content']}"
        
        # Build context-aware prompt
        context_info = f"""
- 当前话题：{topic or '通用学术讨论'}
- 相关知识库内容：{kb_context}
- 相关记忆：{[mem.get('content', '') for mem in relevant_memories]}
- 对话目标：基于已有知识深入探讨该话题
        """
        
        system_prompt = f"""
你是{self.name}，一个学生，人设：{self.persona}。

{context_info}

现在开始与 {', '.join(participants)} 进行关于 "{topic}" 的对话。确保你的发言基于具体的知识点和上下文，提出具体问题或分享具体见解，推动对话进展。
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            initial_message = response.content
            
            # Create dialogue history
            dialogue_history = [{
                "speaker": self.name,
                "message": initial_message,
                "timestamp": str(datetime.now()),
                "topic": topic
            }]
            
            # Let other participants respond (this would normally be handled by the simulator)
            return dialogue_history
        except Exception as e:
            print(f"发起对话时出现错误: {e}")
            return []
    
    def should_join_dialogue_based_on_context(self, topic: str, participants: List[str], world_simulator, location: str):
        """Decide whether to join a dialogue based on context and relevance"""
        # Check if the topic is relevant to my studies
        topic_relevance = 0
        if topic:
            # Check if I have memories related to this topic
            topic_memories = self.long_term_memory.get_memories_by_topic(topic, limit=1)
            if topic_memories:
                topic_relevance = 2
            else:
                # Check if the topic appears in my recent memories
                recent_memories = self.long_term_memory.search_memories(topic, limit=3)
                if recent_memories:
                    topic_relevance = 1
        
        # Check if the teacher is participating
        has_teacher = any("teacher" in p.lower() or "教授" in p or "老师" in p for p in participants)
        
        # Make decision based on relevance and context
        if topic_relevance > 0 or has_teacher:
            return {
                "should_join": True,
                "reason": f"话题'{topic}'与我的学习相关，或有老师参与",
                "confidence": 0.7 if topic_relevance > 1 else 0.5
            }
        else:
            return {
                "should_join": False,
                "reason": f"话题'{topic}'与我的学习不相关",
                "confidence": 0.2
            }
    
    def _generate_dialogue_memory(self, topic: str, dialogue_history: List[Dict], participants: List[str]):
        """Generate memory of a dialogue for this student"""
        dialogue_content = " ".join([turn.get('message', '') for turn in dialogue_history if turn])
        
        # Extract key learnings from the dialogue
        memory_content = f"参与了关于'{topic}'的对话，参与者：{participants}"
        
        dialogue_memory = {
            "id": f"dialogue_{topic}_{datetime.now().isoformat()}",
            "type": "dialogue",
            "timestamp": str(datetime.now()),
            "content": memory_content,
            "details": {
                "topic": topic,
                "participants": participants,
                "dialogue_summary": dialogue_content[:200] + "..." if len(dialogue_content) > 200 else dialogue_content,
                "key_takeaways": self._extract_key_takeaways(dialogue_content, topic)
            },
            "weight": 1.2
        }
        
        self.long_term_memory.add_memory(dialogue_memory)
    
    def _extract_key_takeaways(self, dialogue_content: str, topic: str):
        """Extract key takeaways from dialogue content"""
        # Simple extraction - in a real implementation, this could use more sophisticated NLP
        takeaways = []
        
        # Look for key terms related to the topic
        if topic.lower() in dialogue_content.lower():
            # Extract sentences that contain the topic
            sentences = dialogue_content.split('。')
            for sentence in sentences:
                if topic.lower() in sentence.lower() and len(sentence.strip()) > 10:
                    takeaways.append(sentence.strip())
        
        return takeaways[:3]  # Return up to 3 key takeaways