import os
import re
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import replicate
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.llms.replicate import Replicate as LlamaIndexReplicate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Define the state for LangGraph
class AgentState(TypedDict):
    requirements_text: str
    tasks_text: str
    tasks_estimates: Dict[str, List[str]]
    role_outputs: Dict[str, str]
    generated_requirements: List[str]

class ExperimentRunner:
    def __init__(self):
        self.experiments = [
            
            {"platform": "Ollama", "model": "llama3.2:3b", "framework": "LangChain/LangGraph"},  # Experiment 1
            {"platform": "Replicate", "model": "meta/meta-llama-3-70b-instruct", "framework": "LangChain/LangGraph"},  # Experiment 2
            {"platform": "OpenAI", "model": "gpt-4o-mini", "framework": "LangChain/LangGraph"},  # Experiment 3
            {"platform": "Ollama", "model": "llama3.2:3b", "framework": "LlamaIndex"},  # Experiment 4
            {"platform": "Replicate", "model": "meta/meta-llama-3-70b-instruct", "framework": "LlamaIndex"},  # Experiment 5
            {"platform": "OpenAI", "model": "gpt-4o-mini", "framework": "LlamaIndex"},  # Experiment 6
           {"platform": "Replicate", "model": "deepseek-ai/deepseek-r1", "framework": "LangChain/LangGraph"},  # Experiment 7  # Experiment 7
            {"platform": "Replicate", "model": "deepseek-ai/deepseek-r1", "framework": "LlamaIndex"},  # Experiment 8


        ]
        self.roles = [
            "Project Manager",
            "Requirements Engineer",
            "System Engineer",
            "Software Engineer",
            "Test Engineer",
            "Documentation Engineer"
        ]

    def setup_llm(self, experiment):
        platform = experiment["platform"]
        model = experiment["model"]
        framework = experiment["framework"]

        try:
            if framework == "LangChain/LangGraph":
                if platform == "Ollama":
                    llm = Ollama(model=model.split(":")[0])
                elif platform == "Replicate":
                    llm = None
                else:  # OpenAI
                    llm = ChatOpenAI(model_name=model, api_key=os.getenv("OPENAI_API_KEY"))
            else:  # LlamaIndex
                if platform == "Ollama":
                    llm = LlamaIndexOllama(model=model.split(":")[0])
                elif platform == "Replicate":
                    llm = LlamaIndexReplicate(model=model, api_key=os.getenv("REPLICATE_API_TOKEN"))
                else:  # OpenAI
                    llm = LlamaIndexOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            raise Exception(f"Error setting up LLM for {platform} with {framework}: {str(e)}")

        return llm

    def generate_requirements_langchain(self, experiment, requirements_text):
        llm = self.setup_llm(experiment)
        platform = experiment["platform"]

        prompt = PromptTemplate(
            input_variables=["input"],
            template="Generate tagged requirements and use-cases from the following specification:\n{input}\nFormat each requirement as 'REQ-XXX: Description' with a use-case.\nExample: REQ-001: The system shall allow users to log in with a username and password. Use-case: A user enters their credentials to access the system."
        )

        try:
            if platform == "Replicate":
                response = replicate.run(experiment["model"], input={"prompt": prompt.format(input=requirements_text)})
                if isinstance(response, list):
                    response = " ".join(response)
                print(f"Experiment {experiment['id']} Requirements Raw Response:\n{response}")
            else:
                chain = RunnableSequence(prompt | llm)
                response = chain.invoke({"input": requirements_text})
                # Handle AIMessage object for OpenAI models
                if hasattr(response, 'content'):  # Check if response is an AIMessage object
                    response = response.content  # Extract the text content
                print(f"Experiment {experiment['id']} Requirements Raw Response:\n{response}")
            if not response.strip():  # Now safe to call .strip() on the string
                response = "No requirements generated"
        except Exception as e:
            response = f"Error generating requirements: {str(e)}"

        return [response]

    def generate_tasks_langchain(self, experiment, tasks_text, role, previous_outputs):
        llm = self.setup_llm(experiment)
        platform = experiment["platform"]

        previous_tasks = "\n".join([f"{r}: {output}" for r, output in previous_outputs.items() if output])
        prompt = PromptTemplate(
            input_variables=["input", "role", "previous_tasks"],
            template="You are an AI agent for the role of {role}. Generate tasks with time estimates for a project plan based on the following data:\n{input}\nPrevious tasks generated by other roles:\n{previous_tasks}\nAssign tasks for your role in the following format:\n{role}: [task1 (X hours), task2 (Y hours)]\nExample:\nProject Manager: [Oversee project timeline (5 hours), Coordinate team meetings (3 hours)]\nRequirements Engineer: [Gather requirements (10 hours), Validate requirements (8 hours)]\nSystem Engineer: [Perform system analysis (12 hours), Design system architecture (15 hours)]\nSoftware Engineer: [Develop login module (20 hours), Write unit tests for login module (10 hours)]\nTest Engineer: [Create test plan (8 hours), Execute integration tests (12 hours)]\nDocumentation Engineer: [Write user manual (15 hours), Update API documentation (10 hours)]"
        )

        try:
            if platform == "Replicate":
                response = replicate.run(experiment["model"], input={"prompt": prompt.format(input=tasks_text, role=role, previous_tasks=previous_tasks)})
                if isinstance(response, list):
                    response = " ".join(response)
                print(f"Experiment {experiment['id']} Raw Task Response for {role}:\n{response}")
            else:
                chain = RunnableSequence(prompt | llm)
                response = chain.invoke({"input": tasks_text, "role": role, "previous_tasks": previous_tasks})
                # Handle AIMessage object for OpenAI models
                if hasattr(response, 'content'):  # Check if response is an AIMessage object
                    response = response.content  # Extract the text content
                print(f"Experiment {experiment['id']} Raw Task Response for {role}:\n{response}")
            if not response.strip():
                response = f"{role}: [No tasks generated]"
        except Exception as e:
            response = f"Error generating tasks for {role}: {str(e)}"

        return response

    # LlamaIndex methods remain unchanged as requested
    def generate_requirements_llamaindex(self, experiment, requirements_text):
        llm = self.setup_llm(experiment)
        document = Document(text=requirements_text)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        index = VectorStoreIndex.from_documents([document], embed_model=embed_model, llm=llm)

        query_engine = index.as_query_engine()
        prompt = "Generate tagged requirements and use-cases from the specification. Format each requirement as 'REQ-XXX: Description' with a use-case. Example: REQ-001: The system shall allow users to log in with a username and password. Use-case: A user enters their credentials to access the system."
        try:
            response = query_engine.query(prompt)
            response_text = str(response)
            print(f"Experiment {experiment['id']} Requirements Raw Response:\n{response_text}")
            if not response_text.strip():
                # Fallback to direct LLM call
                print(f"Fallback: Using direct LLM call for requirements generation in Experiment {experiment['id']}")
                response_text = llm.complete(prompt + "\n\nSpecification:\n" + requirements_text).text
                print(f"Experiment {experiment['id']} Fallback Requirements Raw Response:\n{response_text}")
            if not response_text.strip():
                response_text = "No requirements generated"
        except Exception as e:
            response_text = f"Error generating requirements: {str(e)}"

        return [response_text]

    def generate_tasks_llamaindex(self, experiment, tasks_text, role, previous_outputs):
        llm = self.setup_llm(experiment)
        document = Document(text=tasks_text)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        index = VectorStoreIndex.from_documents([document], embed_model=embed_model, llm=llm)

        query_engine = index.as_query_engine()
        previous_tasks = "\n".join([f"{r}: {output}" for r, output in previous_outputs.items() if output])
        prompt = f"You are a {role}. Generate tasks with time estimates based on the data. Previous tasks:\n{previous_tasks}\nFormat: {role}: [task1 (X hours), task2 (Y hours)]"
        try:
            response = query_engine.query(prompt)
            response_text = str(response)
            print(f"Experiment {experiment['id']} Raw Task Response for {role}:\n{response_text}")
            if not response_text.strip():
                # Fallback to direct LLM call
                print(f"Fallback: Using direct LLM call for task generation in Experiment {experiment['id']} for {role}")
                response_text = llm.complete(prompt + "\n\nData:\n" + tasks_text).text
                print(f"Experiment {experiment['id']} Fallback Raw Task Response for {role}:\n{response_text}")
            if not response_text.strip():
                response_text = f"{role}: [No tasks generated]"
        except Exception as e:
            response_text = f"Error generating tasks for {role}: {str(e)}"

        return response_text

    # def parse_tasks(self, role_outputs):
    #     tasks_estimates = {
    #         "Project Manager": [],
    #         "Requirements Engineer": [],
    #         "System Engineer": [],
    #         "Software Engineer": [],
    #         "Test Engineer": [],
    #         "Documentation Engineer": []
    #     }

    #     for role, response in role_outputs.items():
    #         lines = response.split("\n")
    #         for line in lines:
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             line = line.lstrip("* ").strip()
    #             if line.lower().startswith(role.lower() + ":"):
    #                 tasks_str = line[len(role) + 1:].strip()
    #                 if tasks_str.startswith("[") and tasks_str.endswith("]"):
    #                     tasks = []
    #                     task_items = tasks_str[1:-1].split(",")
    #                     for item in task_items:
    #                         item = item.strip()
    #                         if item:
    #                             if "(" in item and ")" in item:
    #                                 task = item[:item.rfind("(")].strip()
    #                                 estimate = item[item.rfind("(")+1:item.rfind(")")].strip()
    #                                 if task:
    #                                     tasks.append(f"{task} ({estimate})")
    #                             else:
    #                                 tasks.append(item)
    #                 else:
    #                     tasks = []
    #                     parts = tasks_str.split("+")
    #                     for part in parts:
    #                         part = part.strip()
    #                         if ":" in part:
    #                             task = part.split(":")[0].strip()
    #                             if task:
    #                                 tasks.append(task)
    #                         else:
    #                             sub_tasks = part.split(",")
    #                             for sub_task in sub_tasks:
    #                                 sub_task = sub_task.strip()
    #                                 if sub_task:
    #                                     tasks.append(sub_task)
    #                 tasks_estimates[role] = tasks
    #                 break

    #     for role in tasks_estimates:
    #         if not tasks_estimates[role]:
    #             tasks_estimates[role] = ["No tasks assigned"]

    #     return tasks_estimates

    

    def parse_tasks(self, role_outputs):
        tasks_estimates = {role: [] for role in self.roles}

        for role, response in role_outputs.items():
            lines = response.split("\n")
            tasks = []

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                # Remove markdown bullets or numbering
                line = re.sub(r"^\*?\s*\d*[\.\)]?\s*", "", line)

                # Match any pattern with (X hours)
                match = re.match(r"(.*?)(?:\s*\(\s*(\d+(?:\.\d+)?)\s*hours?\))", line, re.IGNORECASE)
                if match:
                    task = match.group(1).strip()
                    hours = match.group(2).strip()
                    if task:
                        tasks.append(f"{task} ({hours} hours)")

            tasks_estimates[role] = tasks if tasks else ["No tasks assigned"]

        return tasks_estimates


    def langgraph_agent(self, experiment, requirements_text, tasks_text):
        def requirements_node(state: AgentState):
            requirements = self.generate_requirements_langchain(experiment, state["requirements_text"])
            return {"generated_requirements": requirements}

        def agent_node_factory(role):
            def agent_node(state: AgentState):
                role_output = self.generate_tasks_langchain(experiment, state["tasks_text"], role, state["role_outputs"])
                state["role_outputs"][role] = role_output
                return {"role_outputs": state["role_outputs"]}
            return agent_node

        # Define the LangGraph workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("requirements", requirements_node)

        # Add nodes for each role
        for role in self.roles:
            workflow.add_node(role, agent_node_factory(role))

        # Project Manager coordinates the process
        workflow.set_entry_point("requirements")
        workflow.add_edge("requirements", "Project Manager")
        previous_role = "Project Manager"
        for role in self.roles[1:]:  # Start from Requirements Engineer
            workflow.add_edge(previous_role, role)
            previous_role = role
        workflow.add_edge(previous_role, END)

        app = workflow.compile()
        result = app.invoke({
            "requirements_text": requirements_text,
            "tasks_text": tasks_text,
            "tasks_estimates": {},
            "role_outputs": {role: "" for role in self.roles},
            "generated_requirements": []
        })

        tasks_estimates = self.parse_tasks(result["role_outputs"])
        return result["generated_requirements"], tasks_estimates

    def llamaindex_agent(self, experiment, requirements_text, tasks_text):
        requirements = self.generate_requirements_llamaindex(experiment, requirements_text)
        role_outputs = {role: "" for role in self.roles}

        # Simulate agent coordination
        for i, role in enumerate(self.roles):
            previous_outputs = {r: role_outputs[r] for r in self.roles[:i]}
            role_outputs[role] = self.generate_tasks_llamaindex(experiment, tasks_text, role, previous_outputs)

        tasks_estimates = self.parse_tasks(role_outputs)
        return requirements, tasks_estimates

    def run_experiment(self, experiment_id, requirements_text, tasks_text):
        experiment = self.experiments[experiment_id - 1]
        experiment["id"] = experiment_id

        try:
            if experiment["framework"] == "LangChain/LangGraph":
                generated_requirements, tasks_estimates = self.langgraph_agent(experiment, requirements_text, tasks_text)
                return generated_requirements, tasks_estimates
            else:  # LlamaIndex
                return self.llamaindex_agent(experiment, requirements_text, tasks_text)
        except Exception as e:
            return [f"Error in Experiment {experiment_id}: {str(e)}"], {}