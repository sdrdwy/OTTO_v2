"""
Utility functions for managing token length and memory content
"""

def truncate_text_to_token_limit(text: str, max_tokens: int = 4000) -> str:
    """
    Truncate text to fit within a token limit.
    This is a simple approximation - in practice, tokenization varies by model.
    We'll use character count as a rough proxy for tokens (typically 1 char ≈ 1 token for Chinese).
    """
    if len(text) <= max_tokens:
        return text
    
    # Truncate to max_tokens and add indicator
    truncated = text[:max_tokens]
    return truncated + "... [内容已截断]"

def summarize_memory_content(memories: list, max_total_length: int = 2000) -> str:
    """
    Summarize memory content to fit within length limits.
    """
    if not memories:
        return ""
    
    # First, create a basic summary string
    memory_texts = []
    for mem in memories:
        content = mem.get('content', '')
        if content:
            memory_texts.append(f"- {content}")
    
    full_text = "\n".join(memory_texts)
    
    # If it's too long, we need to summarize
    if len(full_text) > max_total_length:
        # Keep the most important/recent memories and truncate the content
        truncated_memories = []
        current_length = 0
        
        for mem in memories:
            content = mem.get('content', '')
            if content:
                # Add a truncated version of the content
                available_space = max_total_length - current_length
                if available_space <= 100:  # Leave some space for formatting
                    break
                
                if len(content) <= available_space:
                    truncated_content = content
                else:
                    truncated_content = content[:available_space-50] + "...[记忆截断]"
                
                truncated_memories.append(f"- {truncated_content}")
                current_length += len(truncated_content) + 3  # +3 for "- " prefix
        
        return "\n".join(truncated_memories)
    
    return full_text