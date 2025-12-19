import json
import os
from typing import Dict, List, Any
from datetime import datetime
from langchain_community.chat_models import ChatTongyi

from langchain_core.messages import SystemMessage
from agents.base_agent import BaseAgent
from utils.token_manager import truncate_text_to_token_limit, summarize_memory_content
from memory.vector_memory import KnowledgeBaseManager


class ExpertAgent(BaseAgent):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.is_expert = True  # Mark this agent as an expert
        
        # Initialize knowledge base manager
        print("Initialize knowledge base..")
        self.knowledge_base_manager = KnowledgeBaseManager(collection_name="knowledge_base")
        
        # Load knowledge base from JSONL files if available
        kb_paths = [
            self.config.get("knowledge_base_path", "./data/nin_min.jsonl"),
            "./data/nin_min.jsonl"
        ]
        
        for kb_path in kb_paths:
            if os.path.exists(kb_path):
                self.knowledge_base_manager.load_knowledge_from_jsonl(kb_path)
        
        # Get all topics from knowledge base
        self.knowledge_base_topics = self.knowledge_base_manager.get_all_topics()
        
        # Generate curriculum based on knowledge base
        self.curriculum = self.generate_curriculum()
        # Track teaching progress
        self.teaching_progress = {}
        # Track which topics have been taught in order
        self.taught_topics_order = []
    
    def generate_curriculum(self):
        """Generate a comprehensive curriculum based on knowledge base"""
        if not self.knowledge_base_topics:
            return {"topics": [], "sequence": []}
        
        # Get content for each topic
        topics_content = {}
        for topic in self.knowledge_base_topics:
            topic_content = self.knowledge_base_manager.get_knowledge_by_topic(topic, limit=20)  # Get all content for this topic
            topics_content[topic] = topic_content
        
        topics = list(topics_content.keys())
        
        curriculum = {
            "topics": topics,
            "topic_contents": topics_content,  # Store all content for each topic
            "sequence": topics,  # Define teaching sequence
            "schedule": {}
        }
        
        # Create a detailed schedule mapping topics to days
        for i, topic in enumerate(topics):
            day = i + 1
            curriculum["schedule"][f"day_{day}"] = {
                "topic": topic,
                "content_summary": f"关于{topic}的基础知识和应用",
                "content_items": topics_content[topic]
            }
        
        return curriculum
    
    def get_next_teaching_topic(self, student_name: str):
        """Get the next topic to teach based on curriculum progress"""
        if student_name not in self.teaching_progress:
            self.teaching_progress[student_name] = 0
        
        curriculum_sequence = self.curriculum.get("sequence", [])
        current_index = self.teaching_progress[student_name]
        
        if current_index < len(curriculum_sequence):
            topic = curriculum_sequence[current_index]
            # Move to next topic for this student
            self.teaching_progress[student_name] += 1
            return topic
        else:
            # Reset to first topic if we've covered all
            self.teaching_progress[student_name] = 1  # Start from second topic next time, or reset to 0 for restart
            return curriculum_sequence[0] if curriculum_sequence else "通用"
    
    def get_kb_content_by_topic(self, topic: str):
        """Retrieve knowledge base content for a specific topic"""
        return self.knowledge_base_manager.get_knowledge_by_topic(topic, limit=1)
    
    def teach(self, student: BaseAgent, topic: str = None, all_students: List[BaseAgent] = None):
        """Teach a topic to one or more students, using curriculum if topic is not specified"""
        if all_students is None:
            # If no list of all students is provided, just teach the single student
            student_name = student.name
            if topic is None:
                topic = self.get_next_teaching_topic(student_name)
            
            # Find relevant knowledge in the curriculum
            topic_contents = self.curriculum.get("topic_contents", {}).get(topic, [])
            relevant_knowledge = topic_contents
            
            if not relevant_knowledge:
                # If no specific knowledge, use general knowledge
                relevant_knowledge = self.knowledge_base_manager.get_knowledge_by_topic(topic, limit=5) or []
            
            # Get student agent to retrieve relevant memories
            student_agent = student
            
            # Retrieve relevant memories for the student
            student_relevant_memories = []
            if student_agent:
                student_relevant_memories = student_agent.long_term_memory.get_memories_by_topic(topic, limit=3)
            
            # Format knowledge base content for prompt
            kb_text = ""
            if relevant_knowledge:
                for item in relevant_knowledge:
                    kb_text += f"\n【{item.get('topic', '主题')}】{item.get('content', '')}"
            
            # Apply token limiting to knowledge base content
            kb_text = truncate_text_to_token_limit(kb_text, max_tokens=2000)
            
            # Format student memories for context
            memory_context = ""
            if student_relevant_memories:
                # Apply token limiting to student memories
                memory_context = summarize_memory_content(student_relevant_memories, max_total_length=1000)
                memory_context = f"学生最近的学习记忆：\n{memory_context}"
            
            system_prompt = f"""
你是{self.name}，一名专业教师，人设：{self.persona}。
你的教学风格：{self.dialogue_style}。

你正在教授学生{student_name}关于"{topic}"的知识。
相关知识内容：{kb_text}

{memory_context}

请提供清晰、专业的教学内容，使用启发式方法引导学生思考。确保你的教学基于具体的知识点，而不是泛泛而谈。
"""

            try:
                response = self.llm.invoke([SystemMessage(content=truncate_text_to_token_limit(system_prompt, max_tokens=4000))])
                teaching_content = response.content
                
                # Generate memory of teaching session for teacher
                teaching_memory = {
                    "id": f"teaching_{student_name}_{topic}_{datetime.now().isoformat()}",
                    "type": "teaching",
                    "timestamp": str(datetime.now()),
                    "content": f"向{student_name}教授了{topic}的相关知识",
                    "details": {
                        "student": student_name,
                        "topic": topic,
                        "content": teaching_content,
                        "kb_reference": [item.get("topic") for item in relevant_knowledge] if relevant_knowledge else []
                    },
                    "weight": 1.5
                }
                
                self.long_term_memory.add_memory(teaching_memory)
                
                # Also generate a learning memory for the student
                student_learning_memory = {
                    "id": f"learned_from_teacher_{student_name}_{topic}_{datetime.now().isoformat()}",
                    "type": "learning_from_teacher",
                    "timestamp": str(datetime.now()),
                    "content": f"从{self.name}老师那里学习了{topic}的内容",
                    "details": {
                        "teacher": self.name,
                        "topic": topic,
                        "content": teaching_content,
                        "method": "direct_teaching",
                        "kb_reference": [item.get("topic") for item in relevant_knowledge] if relevant_knowledge else [],
                        "context": memory_context
                    },
                    "weight": 1.5
                }
                
                # Return both the teaching content and the memory that should be added to student
                return {
                    "teaching_content": teaching_content,
                    "student_memory": student_learning_memory,
                    "topic": topic
                }
            except Exception as e:
                error_msg = f"教学过程中出现错误: {e}"
                print(error_msg)
                return {
                    "teaching_content": error_msg,
                    "student_memory": None,
                    "topic": topic
                }
        else:
            # Teaching all students
            if topic is None:
                # Use the next topic in the curriculum
                if all_students:
                    first_student_name = all_students[0].name
                    topic = self.get_next_teaching_topic(first_student_name)
                    # Use the same topic for all students, but update progress for each
                    for student_agent in all_students:
                        student_name = student_agent.name
                        # Update teaching progress for each student
                        if student_name not in self.teaching_progress:
                            self.teaching_progress[student_name] = 0
                        curriculum_sequence = self.curriculum.get("sequence", [])
                        current_index = self.teaching_progress[student_name]
                        if current_index < len(curriculum_sequence):
                            self.teaching_progress[student_name] += 1
                        else:
                            self.teaching_progress[student_name] = 1  # Reset to second topic
            
            # Find relevant knowledge in the curriculum
            topic_contents = self.curriculum.get("topic_contents", {}).get(topic, [])
            relevant_knowledge = topic_contents
            
            if not relevant_knowledge:
                # If no specific knowledge, use general knowledge
                relevant_knowledge = self.knowledge_base_manager.get_knowledge_by_topic(topic, limit=5) or []
            
            # Format knowledge base content for prompt
            kb_text = ""
            if relevant_knowledge:
                for item in relevant_knowledge:
                    kb_text += f"\n【{item.get('topic', '主题')}】{item.get('content', '')}"
            
            # Apply token limiting to knowledge base content
            kb_text = truncate_text_to_token_limit(kb_text, max_tokens=2000)
            
            # Get names of all students
            student_names = [s.name for s in all_students]
            students_str = "、".join(student_names)
            
            system_prompt = f"""
你是{self.name}，一名专业教师，人设：{self.persona}。
你的教学风格：{self.dialogue_style}。

你正在教授学生{students_str}关于"{topic}"的知识。
相关知识内容：{kb_text}

请提供清晰、专业的教学内容，使用启发式方法引导学生思考。确保你的教学基于具体的知识点，而不是泛泛而谈。
"""

            try:
                response = self.llm.invoke([SystemMessage(content=truncate_text_to_token_limit(system_prompt, max_tokens=4000))])
                teaching_content = response.content
                
                # Generate memory of teaching session for teacher
                teaching_memory = {
                    "id": f"teaching_all_{topic}_{datetime.now().isoformat()}",
                    "type": "teaching",
                    "timestamp": str(datetime.now()),
                    "content": f"向所有学生教授了{topic}的相关知识",
                    "details": {
                        "students": student_names,
                        "topic": topic,
                        "content": teaching_content,
                        "kb_reference": [item.get("topic") for item in relevant_knowledge] if relevant_knowledge else []
                    },
                    "weight": 1.5
                }
                
                self.long_term_memory.add_memory(teaching_memory)
                
                # Generate a learning memory for each student
                results = []
                for student_agent in all_students:
                    student_name = student_agent.name
                    
                    # Retrieve relevant memories for this student
                    student_relevant_memories = student_agent.long_term_memory.get_memories_by_topic(topic, limit=3)
                    memory_context = ""
                    if student_relevant_memories:
                        memory_context = summarize_memory_content(student_relevant_memories, max_total_length=1000)
                        memory_context = f"学生最近的学习记忆：\n{memory_context}"
                    
                    student_learning_memory = {
                        "id": f"learned_from_teacher_{student_name}_{topic}_{datetime.now().isoformat()}",
                        "type": "learning_from_teacher",
                        "timestamp": str(datetime.now()),
                        "content": f"从{self.name}老师那里学习了{topic}的内容",
                        "details": {
                            "teacher": self.name,
                            "topic": topic,
                            "content": teaching_content,
                            "method": "group_teaching",
                            "kb_reference": [item.get("topic") for item in relevant_knowledge] if relevant_knowledge else [],
                            "context": memory_context
                        },
                        "weight": 1.5
                    }
                    
                    # Add the learning memory to the student
                    student_agent.long_term_memory.add_memory(student_learning_memory)
                    
                    results.append({
                        "student_name": student_name,
                        "teaching_content": teaching_content,
                        "student_memory": student_learning_memory,
                        "topic": topic
                    })
                
                return results
            except Exception as e:
                error_msg = f"教学过程中出现错误: {e}"
                print(error_msg)
                return {
                    "teaching_content": error_msg,
                    "student_memory": None,
                    "topic": topic
                }
    
    def initiate_dialogue(self, participants: List[str], topic: str = None, max_rounds: int = 5, world_simulator=None):
        """Initiate a dialogue with grounding in specific context"""
        if not participants:
            return []
        
        # Get relevant context for the topic
        topic_knowledge = self.get_kb_content_by_topic(topic) if topic else None
        kb_snippet = f"【{topic_knowledge['topic']}】{topic_knowledge['content']}" if topic_knowledge else ""
        
        # Get recent memories related to the topic
        relevant_memories = self.long_term_memory.get_memories_by_topic(topic or "general", limit=3) if topic else []
        
        # Apply token limiting to knowledge base content
        kb_snippet = truncate_text_to_token_limit(kb_snippet, max_tokens=1000)
        
        # Apply token limiting to relevant memories
        memory_context = summarize_memory_content(relevant_memories, max_total_length=1000)
        
        # Build context-aware prompt
        context_info = f"""
- 当前话题：{topic or '通用学术讨论'}
- 相关知识库内容：{kb_snippet}
- 相关记忆：{memory_context}
- 对话目标：深入探讨该话题的具体方面
        """
        
        system_prompt = f"""
你是{self.name}，一名专业教师，人设：{self.persona}。
你的教学风格：{self.dialogue_style}。

{context_info}

现在开始与 {', '.join(participants)} 进行关于 "{topic}" 的对话。确保你的发言基于具体的知识点和上下文，提出具体问题或分享具体见解，推动对话进展。
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=truncate_text_to_token_limit(system_prompt, max_tokens=4000))])
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
        # Check if the topic is relevant to my expertise
        topic_relevance = 0
        if topic:
            # Check if the topic exists in my knowledge base using vector search
            kb_results = self.knowledge_base_manager.search_knowledge(topic, limit=5)
            if kb_results:
                # If we find relevant knowledge, set relevance based on similarity score
                highest_score = max([item.get("similarity_score", 0) for item in kb_results])
                if highest_score > 0.7:  # High similarity
                    topic_relevance = 2
                elif highest_score > 0.3:  # Medium similarity
                    topic_relevance = 1
        
        # Check if any participants are my students
        student_participants = [p for p in participants if p != self.name]
        
        # Make decision based on relevance and context
        if topic_relevance > 0 or student_participants:
            return {
                "should_join": True,
                "reason": f"话题'{topic}'与我的专业知识相关，或有学生参与",
                "confidence": 0.8 if topic_relevance > 1 else 0.6
            }
        else:
            return {
                "should_join": False,
                "reason": f"话题'{topic}'与我的专业领域不相关",
                "confidence": 0.3
            }
    
    def answer_question(self, student: BaseAgent, question: str):
        student_name = student.name
        """Answer a student's question"""
        # Get relevant knowledge from KB
        relevant_knowledge = []
        relevant_knowledge = self.knowledge_base_manager.search_knowledge(
            query=question,
            limit=5
        )
        
        # Get student's relevant memories
        student_relevant_memories = []
        from world.world_simluator import WorldSimulator
        student_agent = student
        if student_agent:
            student_relevant_memories = student_agent.long_term_memory.search_memories(question, limit=3)
        
        # Format knowledge for prompt
        kb_context = ""
        if relevant_knowledge:
            kb_context = "相关知识库内容：\n"
            for item in relevant_knowledge:
                kb_context += f"【{item.get('topic', '主题')}】{item.get('content', '')}\n"
        
        # Apply token limiting to knowledge base content
        kb_context = truncate_text_to_token_limit(kb_context, max_tokens=2000)
        
        memory_context = ""
        if student_relevant_memories:
            # Apply token limiting to student memories
            memory_context = summarize_memory_content(student_relevant_memories, max_total_length=1000)
            memory_context = f"学生相关记忆：\n{memory_context}"
        
        system_prompt = f"""
你是{self.name}，一名专业教师，人设：{self.persona}。
你的教学风格：{self.dialogue_style}。

学生{student_name}提出了问题："{question}"

{kb_context}
{memory_context}

请提供专业、准确的回答，结合你的知识库内容和学生的历史学习情况。
"""
        
        try:
            response = self.llm.invoke([SystemMessage(content=truncate_text_to_token_limit(system_prompt, max_tokens=4000))])
            answer = response.content
            
            # Generate memory of Q&A session
            qa_memory = {
                "id": f"qa_{student_name}_{question[:20]}_{datetime.now().isoformat()}",
                "type": "question_answer",
                "timestamp": str(datetime.now()),
                "content": f"回答了{student_name}关于'{question[:30]}...'的问题",
                "details": {
                    "student": student_name,
                    "question": question,
                    "answer": answer,
                    "related_kb_topics": [item.get("topic") for item in relevant_knowledge],
                    "related_student_memories": [mem.get("id") for mem in student_relevant_memories]
                },
                "weight": 1.2
            }
            
            self.long_term_memory.add_memory(qa_memory)
            
            return answer
        except Exception as e:
            error_msg = f"回答问题时出现错误: {e}"
            print(error_msg)
            return error_msg

    def create_exam(self, num_questions: int = 5):
        """Create an exam based on the curriculum and knowledge base using LLM"""
        # ✅ 使用 knowledge_base_manager 替代 self.knowledge_base
        all_topics = self.knowledge_base_manager.get_all_topics()
        if not all_topics:
            print("⚠️ 知识库中无主题，无法生成试卷")
            return []

        # 从 curriculum 或 topics 中选择主题
        selected_topics = (self.curriculum.get("sequence", []) or all_topics)[:num_questions]
        if len(selected_topics) < num_questions:
            # 补足不足的题目
            extra = all_topics * ((num_questions // len(all_topics)) + 1)
            selected_topics = selected_topics + extra
        selected_topics = selected_topics[:num_questions]

        exam_questions = []
        for i, topic in enumerate(selected_topics):
            # ✅ 从知识库管理器中获取该 topic 的内容
            topic_items = self.knowledge_base_manager.get_knowledge_by_topic(topic, limit=5)
            if not topic_items:
                # 如果没找到，用 topic 名作为 fallback
                knowledge_content = f"关于主题 '{topic}' 的相关知识。"
            else:
                # 拼接前几条内容（或只取第一条）
                knowledge_content = "\n".join(
                    item.get("content", str(item)) for item in topic_items[:2]
                )

            # 限制 token 长度
            knowledge_content = truncate_text_to_token_limit(knowledge_content, max_tokens=2000)

            system_prompt = f"""
    你是{self.name}，一名专业教师，人设：{self.persona}。
    你的教学风格：{self.dialogue_style}。

    基于以下关于"{topic}"的知识内容：
    {knowledge_content}

    请为学生创建一道关于"{topic}"的考试题目。
    并根据指示内容生成一个参考答案ref_ans
    返回格式为JSON对象：
    {{
        "question": "问题内容",
        "type": "short_answer",
        "topic": "{topic}",
        "reference_answer":"ref_ans"
    }}
    """
            try:
                response = self.llm.invoke([
                    SystemMessage(content=truncate_text_to_token_limit(system_prompt, max_tokens=4000))
                ])
                content = response.content.strip()

                # 安全解析 JSON
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx+1]
                    question = json.loads(json_str)
                    exam_questions.append(question)
                else:
                    raise ValueError("LLM 未返回有效 JSON")

            except Exception as e:
                print(f"为话题 '{topic}' 生成考试题目失败: {e}")
                exam_questions.append({
                    "question": f"请简述关于‘{topic}’的主要知识点。",
                    "type": "short_answer",
                    "topic": topic,
                    "reference_answer": "（参考答案由教师根据知识库生成）"
                })
        print(f"考试题目:{exam_questions}")
        return exam_questions

    def grade_exam(self, student_name: str, answers: List[Dict], exam_questions: List[Dict], student_agent=None):
        """Grade a student's exam answers using LLM"""
        max_score = len(answers) * 10  # 10 points per question
        grading_results = []
        
        for i, (answer, question) in enumerate(zip(answers, exam_questions)):
            answer_text = answer.get('answer', '')
            question_text = question.get('question', '')
            question_topic = question.get('topic', '通用')
            reference_answer = question.get("reference_answer","")
            
            # Get relevant context from knowledge base
            kb_context = ""
            topic_knowledge = self.get_kb_content_by_topic(question_topic)
            if topic_knowledge:
                kb_context = f"【{topic_knowledge[0]['topic']}】{topic_knowledge[0]['content']}"
            
            # Apply token limiting to knowledge base content
            kb_context = truncate_text_to_token_limit(kb_context, max_tokens=1500)
            
            # Get student's relevant memories for this topic
            student_memories_context = ""
            if student_agent:
                student_memories = student_agent.long_term_memory.get_memories_by_topic(question_topic, limit=3)
                if student_memories:
                    # Apply token limiting to student memories
                    memory_summary = summarize_memory_content(student_memories, max_total_length=1000)
                    student_memories_context = f"学生相关学习记忆：\n{memory_summary}"
            
            # Use LLM to grade the answer
            system_prompt = f"""
你是{self.name}，一名专业教师，人设：{self.persona}。
你的教学风格：{self.dialogue_style}。

请对学生的答案进行评分。

题目：{question_text}
学生答案：{answer_text}
主题：{question_topic}
参考答案:{reference_answer}

知识库相关内容：
{kb_context}

{student_memories_context}

评分标准：
- 内容准确性 (0-4分)
- 回答完整性 (0-3分)
- 表达清晰度 (0-3分)

请提供评分和反馈，返回格式为JSON：
{{
    "score": 0-10之间的分数,
    "feedback": "具体的评分反馈和建议",
    "topic": "{question_topic}"
}}
"""
            try:
                response = self.llm.invoke([SystemMessage(content=truncate_text_to_token_limit(system_prompt, max_tokens=4000))])
                import json
                content = response.content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                    score = result.get("score", 5)  # Default to 5 if parsing fails
                    feedback = result.get("feedback", "评分完成")
                else:
                    # Fallback scoring
                    score = 5
                    feedback = f"基于{question_topic}的回答评分"
            except Exception as e:
                print(f"使用LLM评分失败，使用默认评分: {e}")
                # Fallback to basic scoring
                score = 5
                feedback = f"基于{question_topic}的回答评分"
            
            grading_results.append({
                "question_idx": i,
                "score": score,
                "feedback": feedback,
                "topic": question_topic
            })
        
        # Calculate total score
        total_score = sum(result["score"] for result in grading_results)
        overall_score = (total_score / max_score) * 100 if max_score > 0 else 0
        
        # Store grading memory
        grading_memory = {
            "id": f"exam_grade_{student_name}_{datetime.now().isoformat()}",
            "type": "exam_grading",
            "timestamp": str(datetime.now()),
            "content": f"{student_name}的考试成绩：{overall_score:.1f}分",
            "details": {
                "student": student_name,
                "total_score": overall_score,
                "grading_results": grading_results,
                "answers": answers,
                "exam_questions": exam_questions
            },
            "weight": 2.0
        }
        
        self.long_term_memory.add_memory(grading_memory)
        
        # Also create a learning memory for the student about the exam
        if student_agent:
            student_exam_memory = {
                "id": f"exam_result_{student_name}_{datetime.now().isoformat()}",
                "type": "exam_result",
                "timestamp": str(datetime.now()),
                "content": f"考试结果：{overall_score:.1f}分",
                "details": {
                    "exam_score": overall_score,
                    "grading_results": grading_results,
                    "teacher_feedback": [gr.get("feedback") for gr in grading_results],
                    "performance_analysis": f"在{len([r for r in grading_results if r['score'] >= 7])}/{len(grading_results)}题表现良好"
                },
                "weight": 1.8
            }
            student_agent.long_term_memory.add_memory(student_exam_memory)
        
        return {
            "total_score": overall_score,
            "grading_results": grading_results,
            "max_score": max_score
        }