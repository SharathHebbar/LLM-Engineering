from dotenv import load_dotenv
from crew import med_crew

load_dotenv()

def run(user_input: str):
    result = med_crew.kickoff(inputs={"user_input": user_input})
    print(result)

if __name__ == "__main__":
    # Example patient report or question
    sample_input = """
    Hi, I recently had a blood test and received this report:
    Hemoglobin: 10.5 g/dL
    Fasting Blood Sugar: 135 mg/dL
    LDL Cholesterol: 160 mg/dL
    Vitamin D: 15 ng/mL
    Can you help me understand this report?
    """
    
    run(sample_input)