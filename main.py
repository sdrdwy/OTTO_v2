import os
from agents.expert_agent import ExpertAgent
from agents.student_agent import StudentAgent
from world.world_simluator import WorldSimulator

# Set the API key
os.environ["DASHSCOPE_API_KEY"] = "sk-c763fc92bf8c46c7ae31639b05d89c96"

def main():
    # Create expert agent (teacher)
    expert_agent = ExpertAgent("./config/agents/expert.json")
    
    # Create student agents
    student_agents = [
        StudentAgent("./config/agents/Arisu.json"),
        StudentAgent("./config/agents/Midori.json"),
        StudentAgent("./config/agents/Momoi.json"),
        StudentAgent("./config/agents/Yuzu.json")
    ]
    
    # Create world simulator
    world_simulator = WorldSimulator()
    
    # Combine all agents
    all_agents = [expert_agent] + student_agents
    
    # Start the simulation
    world_simulator.start_simulation(all_agents, is_exam=True)

if __name__ == "__main__":
    main()