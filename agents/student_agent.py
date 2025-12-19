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
        system_prompt = f"""
        你是{self.name}，一个学生，人设：{self.persona}。
        你想向{teacher_name}老师询问关于"{topic}"的问题："{question}"
        
        请以学生的身份提出问题，保持符合你的角色设定。
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
                    "question": question_content
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
        
        system_prompt = f"""
        你是{self.name}，一个学生，人设：{self.persona}。
        你正在学习"{topic}"这个主题。
        学习材料：{study_materials}
        
        请描述你的学习过程和收获，保持符合你的角色设定。
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
                    "materials": study_materials
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
            
            # Use LLM to generate a proper answer based on the question
            system_prompt = f"""
            你是{self.name}，一个学生，人设：{self.persona}。
            
            请回答以下考试题目：
            题目：{question_text}
            主题：{topic}
            
            请提供一个详细且准确的答案，保持符合你的角色设定。
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
                "answers": answers
            },
            "weight": 1.8
        }
        
        self.long_term_memory.add_memory(exam_memory)
        
        return answers

    def ask_teacher_for_help(self, teacher, topic: str):
        """Ask the teacher for help on a specific topic using LLM"""
        # Get recent memories to understand what the student is struggling with
        recent_memories = self.long_term_memory.search_memories(limit=5)
        
        system_prompt = f"""
        你是{self.name}，一个学生，人设：{self.persona}。
        
        你对"{topic}"这个主题有疑问，需要向老师寻求帮助。
        你的近期学习记忆：{recent_memories}
        
        请根据你的学习情况和人设，向老师提出一个具体的学习问题。
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=system_prompt)])
            question = response.content
            
            # Ask the teacher the question
            teacher_response = teacher.answer_question(self.name, question)
            
            # Create memory of asking for help
            help_memory = {
                "id": f"help_request_{topic}_{datetime.now().isoformat()}",
                "type": "help_request",
                "timestamp": str(datetime.now()),
                "content": f"向老师寻求关于{topic}的帮助",
                "details": {
                    "topic": topic,
                    "question": question,
                    "teacher_response": teacher_response
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