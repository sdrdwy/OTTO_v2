import json
import os
from typing import Dict, List, Any
from datetime import datetime
from langchain_community.chat_models import ChatTongyi

from langchain_core.messages import SystemMessage
from agents.base_agent import BaseAgent


class ExpertAgent(BaseAgent):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load knowledge base if available
        self.knowledge_base = []
        kb_path = self.config.get("knowledge_base_path", "./data/knowledge_base.jsonl")
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.knowledge_base.append(json.loads(line))
        
        # Generate curriculum based on knowledge base
        self.curriculum = self.generate_curriculum()
        # Track teaching progress
        self.teaching_progress = {}
    
    def generate_curriculum(self):
        """Generate a comprehensive curriculum based on knowledge base"""
        if not self.knowledge_base:
            return {"topics": [], "sequence": []}
        
        # Group knowledge by topics
        topics_content = {}
        for item in self.knowledge_base:
            topic = item.get("topic", "通用")
            if topic not in topics_content:
                topics_content[topic] = []
            topics_content[topic].append(item)
        
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
        """Get the next topic to teach based on student's progress"""
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
    
    def teach(self, student_name: str, topic: str = None):
        """Teach a student on a specific topic, using curriculum if topic is not specified"""
        if topic is None:
            topic = self.get_next_teaching_topic(student_name)
        
        # Find relevant knowledge in the curriculum
        topic_contents = self.curriculum.get("topic_contents", {}).get(topic, [])
        relevant_knowledge = topic_contents
        
        if not relevant_knowledge:
            # If no specific knowledge, use general knowledge
            relevant_knowledge = self.knowledge_base[:1] if self.knowledge_base else []
        
        system_prompt = f"""
        你是{self.name}，一名专业教师，人设：{self.persona}。
        你的教学风格：{self.dialogue_style}。
        
        你正在教授学生{student_name}关于"{topic}"的知识。
        相关知识内容：{relevant_knowledge}
        
        请提供清晰、专业的教学内容，使用启发式方法引导学生思考。
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
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
                    "content": teaching_content
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
                    "method": "direct_teaching"
                },
                "weight": 1.5
            }
            
            # Get student agent to add the memory
            from world.world_simluator import WorldSimulator
            # We'll add the student's memory in the calling function since we need access to student object
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
    
    def answer_question(self, student_name: str, question: str):
        """Answer a student's question"""
        system_prompt = f"""
        你是{self.name}，一名专业教师，人设：{self.persona}。
        你的教学风格：{self.dialogue_style}。
        
        学生{student_name}提出了问题："{question}"
        
        请提供专业、准确的回答，结合你的知识库内容。
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
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
                    "answer": answer
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
        if not self.knowledge_base:
            return -1
        
        # Use LLM to generate questions based on knowledge base
        topics = list(set([item.get("topic", "通用") for item in self.knowledge_base]))
        selected_topics = topics[:num_questions] if len(topics) >= num_questions else topics + topics[:num_questions-len(topics)]
        
        exam_questions = []
        for i, topic in enumerate(selected_topics):
            relevant_knowledge = [item for item in self.knowledge_base if item.get("topic") == topic]
            if relevant_knowledge:
                knowledge_content = str(relevant_knowledge[0].get("content", ""))
            else:
                knowledge_content = "相关知识内容"
            
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
                "reference_answer":"ref_ans",
            }}
            """
            try:
                response = self.llm.invoke([SystemMessage(content=system_prompt)])
                import json
                content = response.content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    question = json.loads(json_str)
                    exam_questions.append(question)
                else:
                    # Fallback to simple question
                    exam_questions.append({
                        "question": f"请简述关于{topic}的主要知识点",
                        "type": "short_answer",
                        "topic": topic
                    })
            except Exception as e:
                print(f"为话题'{topic}'生成考试题目失败: {e}")
                exam_questions.append({
                    "question": f"请简述关于{topic}的主要知识点",
                    "type": "short_answer",
                    "topic": topic
                })
        
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
            # Use LLM to grade the answer
            system_prompt = f"""
            你是{self.name}，一名专业教师，人设：{self.persona}。
            你的教学风格：{self.dialogue_style}。
            
            请对学生的答案进行评分。
            
            题目：{question_text}
            学生答案：{answer_text}
            主题：{question_topic}
            参考答案:{reference_answer}
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
                response = self.llm.invoke([SystemMessage(content=system_prompt)])
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
                "answers": answers
            },
            "weight": 2.0
        }
        
        self.long_term_memory.add_memory(grading_memory)
        
        return {
            "total_score": overall_score,
            "grading_results": grading_results,
            "max_score": max_score
        }