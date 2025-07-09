from crewai import Crew

from tasks.doctor_task import doctor_task
from tasks.consultant_task import consultant_task
from agents.doctor_agent import doctor_agent
from agents.consultant_agent import consultant_doctor_agent

med_crew = Crew(
    agents=[doctor_agent, consultant_doctor_agent],
    tasks=[doctor_task, consultant_task],
    verbose=True
)
