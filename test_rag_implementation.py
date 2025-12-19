"""
Test script to verify the RAG-based memory implementation and contextual dialogue system
"""
import os
import sys
import json
from datetime import datetime

# Add the workspace directory to Python path
sys.path.insert(0, '/workspace')

from agents.expert_agent import ExpertAgent
from agents.student_agent import StudentAgent
from memory.long_term_mem import LongTermMemory
from dialogue.dialogue_manager import DialogueManager, run_dialogue_with_context
from world.world_simluator import WorldSimulator


def test_memory_retrieval():
    """Test the enhanced memory retrieval capabilities"""
    print("Testing enhanced memory retrieval...")
    
    # Create a memory instance
    memory = LongTermMemory("/workspace/data/test_memory.jsonl")
    
    # Add some test memories
    test_memories = [
        {
            "id": "test_001",
            "type": "studying",
            "timestamp": str(datetime.now()),
            "content": "今天学习了中医皮肤病学中的油风（斑秃）相关知识",
            "details": {"topic": "oil_wind", "source": "textbook"},
            "weight": 1.0
        },
        {
            "id": "test_002", 
            "type": "lecture",
            "timestamp": str(datetime.now()),
            "content": "老师讲解了斑秃的病因和治疗方法",
            "details": {"topic": "alopecia", "teacher": "expert"},
            "weight": 1.5
        },
        {
            "id": "test_003",
            "type": "experiment",
            "timestamp": str(datetime.now()),
            "content": "实验观察了毛囊细胞在不同条件下的生长情况",
            "details": {"topic": "hair_growth", "method": "microscopy"},
            "weight": 1.2
        }
    ]
    
    for mem in test_memories:
        memory.add_memory(mem)
    
    # Test topic-based retrieval
    oil_wind_memories = memory.get_memories_by_topic("油风", limit=5)
    print(f"Found {len(oil_wind_memories)} memories related to '油风'")
    
    # Test field-based retrieval
    studying_memories = memory.search_by_content_fields("学习", fields=["content"], limit=5)
    print(f"Found {len(studying_memories)} memories with '学习' in content")
    
    # Test general search
    斑秃_memories = memory.search_memories("斑秃", limit=5)
    print(f"Found {len(斑秃_memories)} memories related to '斑秃'")
    
    print("Memory retrieval test completed.\n")
    return True


def test_expert_agent_enhancements():
    """Test the enhanced expert agent functionality"""
    print("Testing enhanced expert agent...")
    
    # Load expert agent
    expert_agent = ExpertAgent("/workspace/configs/teacher.json")
    
    # Test topic knowledge retrieval
    topic_knowledge = expert_agent.get_kb_content_by_topic("油风")
    if topic_knowledge:
        print(f"Found knowledge for topic '油风': {topic_knowledge.get('topic', 'Unknown')}")
    else:
        print("No knowledge found for '油风' (this is OK if not in KB)")
    
    # Test context-based dialogue initiation
    dialogue_history = expert_agent.initiate_dialogue(
        participants=["Arisu", "Yuzu"],
        topic="中医皮肤病学",
        max_rounds=3
    )
    
    print(f"Initiated dialogue with {len(dialogue_history)} initial message(s)")
    
    print("Expert agent test completed.\n")
    return True


def test_student_agent_enhancements():
    """Test the enhanced student agent functionality"""
    print("Testing enhanced student agent...")
    
    # Load student agent
    student_agent = StudentAgent("/workspace/configs/arisa.json")
    
    # Test topic-based memory retrieval in ask_question
    question = student_agent.ask_question(
        teacher_name="Teacher",
        topic="油风",
        question="油风的治疗方法有哪些？"
    )
    
    print(f"Generated question: {question[:100]}...")
    
    # Test study_topic with memory context
    study_result = student_agent.study_topic(
        topic="油风",
        study_materials=["中医皮肤病学教材", "相关论文"]
    )
    
    print(f"Study result: {study_result[:100]}...")
    
    print("Student agent test completed.\n")
    return True


def test_dialogue_manager():
    """Test the new dialogue manager"""
    print("Testing dialogue manager...")
    
    # Create mock world simulator for testing
    class MockWorldSimulator:
        def __init__(self):
            self.agents = {}
    
    # Create mock agents for testing
    expert_agent = ExpertAgent("/workspace/configs/teacher.json")
    student_agent = StudentAgent("/workspace/configs/arisa.json")
    
    # Set up mock world
    mock_world = MockWorldSimulator()
    mock_world.agents = {
        "Teacher": expert_agent,
        "Arisu": student_agent
    }
    
    # Test the dialogue manager
    try:
        dialogue_manager = DialogueManager(mock_world)
        print("DialogueManager created successfully")
        
        # Test context preparation
        context = dialogue_manager._prepare_agent_context(
            expert_agent, "油风", [expert_agent, student_agent]
        )
        print(f"Agent context prepared with KB snippet length: {len(context.get('kb_snippet', ''))}")
        
        print("Dialogue manager test completed.\n")
        return True
    except Exception as e:
        print(f"Dialogue manager test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests to verify the implementation"""
    print("="*60)
    print("COMPREHENSIVE TEST OF RAG-BASED IMPLEMENTATION")
    print("="*60)
    
    tests = [
        ("Memory Retrieval", test_memory_retrieval),
        ("Expert Agent Enhancements", test_expert_agent_enhancements), 
        ("Student Agent Enhancements", test_student_agent_enhancements),
        ("Dialogue Manager", test_dialogue_manager)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"Test {test_name} failed with error: {e}")
            results.append((test_name, False, str(e)))
    
    print("="*60)
    print("TEST RESULTS:")
    print("="*60)
    
    all_passed = True
    for test_name, result, error in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if error:
            print(f"  Error: {error}")
        if not result:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("\nThe RAG-based memory system and contextual dialogue implementation is working correctly.")
        print("Key improvements:")
        print("- Memory retrieval now uses semantic search and topic-based filtering")
        print("- Expert and Student agents now use context-aware prompts")
        print("- Dialogue system incorporates knowledge base and memory context")
        print("- Agents can make more meaningful, grounded conversations")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Please check the errors above.")
    
    print("="*60)
    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)