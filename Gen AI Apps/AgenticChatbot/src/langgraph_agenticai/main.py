import streamlit as st
from src.langgraph_agenticai.ui.streamlitui.load_ui import LoadStreamlitUI
from src.langgraph_agenticai.llms.groq_llm import GroqLLM
from src.langgraph_agenticai.graph.graph_builder import GraphBuilder
from src.langgraph_agenticai.ui.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    """
    Loads and runs the Langgraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM Model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.
    """

    ui = LoadStreamlitUI()

    user_input = ui.load_streamlit_ui()


    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return
    
    user_message = st.chat_input("Enter your message: ")

    if user_message:
        try:
            obj_llm_config = GroqLLM(user_controls_input=user_input)
            model = obj_llm_config.get_llm_models()
            if not model:
                st.error("Error: LLM model could not be initialized.")
                return
            
            usecase = user_input.get("selected_usecase")
            if not usecase:
                st.error("Error: No usecase selected.")
                return
            
            graph_builder = GraphBuilder(model)
            try:
                graph = graph_builder.setup_graph(usecase)
                DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()

            except Exception as e:
                st.error(f"Error: Graph set up failed- {e}")
                return

        except Exception as e:
            st.error(f"Error: Graph set up failed- {e}")
            return