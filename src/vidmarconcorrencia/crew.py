# crew.py

import os
import openai
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, ScrapeElementFromWebsiteTool, ScrapeWebsiteTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

openai.api_key = api_key
# Initialize the LLM
llm                   = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
gpt_mini              = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
gpt4o_mini_2024_07_18 = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", openai_api_key=api_key)
gpt4o                 = ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key)
gpt_o1                = ChatOpenAI(model_name="o1-preview", openai_api_key=api_key)
gpt_o1_mini           = ChatOpenAI(model_name="o1-mini", openai_api_key=api_key)

# Initialize tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
scrape_element_tool = ScrapeElementFromWebsiteTool()
website_search_tool = WebsiteSearchTool()

@CrewBase
class VidmarconcorrenciaCrew():
	"""Vidmarconcorrencia crew"""

	@agent
	def price_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['price_agent'],
			tools=[search_tool, scrape_tool],
			memory=True,
			verbose=True,
			llm=gpt4o_mini_2024_07_18
		)

	@agent
	def product_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['product_agent'],
			tools=[search_tool, scrape_tool, scrape_element_tool],
			memory=True,
			allow_delegation=True,
			verbose=True,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,      # Permite fallback caso precise de um especialista em tecnologia ou satisfação do cliente
			llm=gpt4o_mini_2024_07_18
		)

	@agent
	def marketing_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['marketing_agent'],
			tools=[search_tool, scrape_tool, website_search_tool],
			memory=True,
			verbose=True,
			allow_interruption=True,
			llm=gpt4o_mini_2024_07_18
		)

	@agent
	def customer_satisfaction_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['customer_satisfaction_agent'],
			tools=[search_tool, scrape_tool],
			memory=True,
			verbose=True,
			allow_interruption=True,  # Permite interrupções para reagir rapidamente a mudanças nos produtos dos concorrentes
			allow_fallback=True,      # Permite fallback caso precise de um especialista em tecnologia ou satisfação do cliente
			llm=gpt4o_mini_2024_07_18
		)

	@agent
	def technology_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['technology_agent'],
			tools=[search_tool, website_search_tool],
			memory=True,
			verbose=True,
			allow_delegation=True,
			allow_interruption=True,
			llm=gpt4o_mini_2024_07_18
		)

	@agent
	def sustainability_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['sustainability_agent'],
			tools=[search_tool, website_search_tool],
			memory=True,
			verbose=True,
			allow_delegation=True,
			allow_fallback=True, 
			llm=gpt4o_mini_2024_07_18
		)


	@task
	def price_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['price_analysis'],
			output_file='price_analysis_report.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 20000}]
		)

	@task
	def product_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['product_analysis'],
			output_file='product_analysis_report.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 20000}]
		)

	@task
	def marketing_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['marketing_analysis'],
			output_file='marketing_analysis_report.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 20000}]
		)		

	@task
	def customer_satisfaction_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['customer_satisfaction_analysis'],
			output_file='customer_satisfaction_report.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 20000}]
		)

	@task
	def technology_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['technology_analysis'],
			output_file='technology_analysis_report.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 20000}]
		)

	@task
	def sustainability_analysis(self) -> Task:
		return Task(
			config=self.tasks_config['sustainability_analysis'],
			output_file='sustainability_report.md',
			guardrails=[{"output_format": "markdown"}, {"max_length": 20000}]
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Vidmarconcorrencia crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			memory=True  # Activate memory
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)