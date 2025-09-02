
from src.states.blogstate import BlogState

class BlogNode:
    """A class to represent the blog node"""

    def __init__(self, llm):
        self.llm = llm

    def title_creation(self, state: BlogState):
        """Create a Title for Blog"""

        if "topic" in state and state["topic"]:
            prompt = """You are an expert blog title writer. Use Markdown formatting.\
                Generate exactly one single-line blog title for the {topic}.\
                The title must be creative, SEO-friendly, and concise.\
                Do not include explanations or multiple options—output only the title."""
            
            system_message = prompt.format(topic=state['topic'])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": response.content}}
    
    def content_generation(self, state: BlogState):
        """Generate a content for Blog"""
        
        if "topic" in state and state["topic"]:
            prompt = """You are an expert blog content writer. Use Markdown formatting.\
                Generate a detailed blog content with the detailed breakdown for the {topic}."""
            
            system_message = prompt.format(topic=state['topic'])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": state['blog']['title'], "content": response.content}}
        
            