from langchain_core.messages import HumanMessage
from src.states.blogstate import BlogState, Blog

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
        
    def translation(self, state:BlogState):
        """Translate the content to the specific language,"""
        translation_prompt = """Translate the following content into {current_language}.
        - Maintain the original tone, style, and formatting.
        - Adapt cultural references and idioms to be appropriate for {current_language}.
        ORIGINAL_CONTENT"
        {blog_content}"""

        blog_content = state['blog']['content']
        message=[
            HumanMessage(
                translation_prompt.format(
                    current_language=state['current_language'],
                    blog_content=blog_content
                )
            )
        ]

        translation_content = self.llm.with_structured_output(Blog).invoke(message)
        
    def route(self, state: BlogState):
        return {"current_language": state['current_language']}
    

    def route_decision(self, state:BlogState):
        """Route the content to the respective translation function."""

        if state['current_language'] == "hindi":
            return "hindi"
        elif state['current_language'] == "french":
            return "french"
        else:
            return state['current_language']