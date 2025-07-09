from crewai import Task
from agents.doctor_agent import doctor_agent
from tasks.consultant_task import consultant_task

doctor_task = Task(
    description=(
        "Analyze the uploaded medical report or record in depth. Focus on identifying patterns in lab values, "
        "diagnostic statements, or prescriptions. Cross-check against known medical knowledge to provide a possible "
        "explanation of the condition, associated risks, and clinical significance. "
        "Your output will be used internally or summarized by the `consultant_doctor_agent` for patient-facing delivery."
    ),
    expected_output=(
        "A medically grounded interpretation of the document, including:\n"
        "- Diagnostic reasoning or possible medical explanations\n"
        "- Identification of abnormal values, red flags, or inconsistencies\n"
        "- Clinical recommendations or potential differential diagnoses (non-binding)\n"
        "- Flag if further tests or human doctor intervention is needed."
    ),
    agent=doctor_agent,
    # context=consultant_task
)
