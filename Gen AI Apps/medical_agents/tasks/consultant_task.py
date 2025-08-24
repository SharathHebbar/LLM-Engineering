from crewai import Task
from agents.consultant_agent import consultant_doctor_agent

consultant_task = Task(
    description=(
        "Review the medical report or query provided by the user: {user_input}. "
        "Extract the key findings and explain them in simple, patient-friendly language. "
        "Your primary goal is to help the user understand what the medical report means, why it matters, "
        "and what general steps they might take next. "
        "If the document contains complex findings or requires clinical interpretation beyond general knowledge, "
        "**delegate the task to the `doctor_agent` for detailed medical analysis.**"
    ),
    expected_output=(
        "A clear and empathetic explanation of the medical document, tailored for a non-medical audience. The output should include:\n"
        "- Summary of findings (in simple language)\n"
        "- Possible implications for the patient’s health\n"
        "- General next steps or lifestyle advice\n"
        "- **If escalation occurs**, a referral note explaining why deeper analysis is required and what the `doctor_agent` will assess."
    ),
    agent=consultant_doctor_agent
)
