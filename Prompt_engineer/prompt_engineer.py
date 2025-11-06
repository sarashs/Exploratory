from typing import TypedDict, List, Dict, Any, Literal, Callable, NamedTuple
import json
from pydantic import BaseModel, Field
import random
from langgraph.graph import StateGraph, END
from openai import OpenAI
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import notebook
from tenacity import retry, stop_after_attempt, wait_exponential
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import glob
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np
import os
##############################################
# Configuration
##############################################
EPSILON = 0.5  # exploration rate
DROP_OUT = 0.5 # dropout rate for improvement actions
NUM_FAILURE_CASES_TOSHOW = 10
#RANDOM_SEED = 42  # or any integer you prefer
TARGET_SCORE = 0.99
RECURSION_LIMIT = 5000
##############################################

client = OpenAI()

# Define named functions that can be pickled
def simple_format_function(x):
    return x

def text_format_function(x, y):
    return x.format(text=y)

def kwargs_format_function(prompt, **kwargs):
    return prompt.format(**kwargs)

class PromptTemplateData:
    def __init__(
        self, 
        prompt: str, 
        system_prompt: str, 
        prompt_format_function: Callable[[str], str] = simple_format_function,
        output_format: BaseModel = None
    ):
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.prompt_format_function = prompt_format_function
        self.output_format = output_format
    
    def __call__(self, *args, **kwargs):
        """Make the template callable to execute prompt_format_function"""
        ret = PromptTemplateData(self.prompt_format_function(self.prompt, *args, **kwargs),
                                 self.system_prompt,
                                 self.prompt_format_function,
                                 self.output_format)
        return ret


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=10))
def llm_call(prompt: PromptTemplateData, temperature=0.7, model: str = "gpt-4o"):
    llm_out = client.responses.parse(
        temperature=temperature,
        model=model,
        input=[{"role": "system", "content": prompt.system_prompt},
               {"role": "user", "content": prompt.prompt}],  # Use prompt.prompt directly instead of prompt()
        text_format=prompt.output_format,
    )
    llm_out = llm_out.output_parsed
    return llm_out

# Available improvement actions
IMPROVEMENT_ACTIONS = [
    "Add examples: This is pertaining to few-shot prompting. Try not to add examples directly from the feedback. Instead, create new examples that are similar in nature to the feedback.",
    "Remove examples: If you think the current examples are not helpful or are irrelevant, consider removing them.",
    "Add Chain of Thought reasoning: You can ask the model to think step-by-step, and output its reasoning.",
    "Remove Chain of Thought reasoning: If the current prompt includes chain of thought reasoning, consider removing it.",
    "Add specificity: Give more specific instructions based on the feedback.",
    "Remove specificity: Make the prompt more generic so that it considers a wider range of responses.",
    "Change wording: Rephrase parts of the prompt to improve clarity or effectiveness.",
    "Add context: Context can be added to provide more information to the model.",
    "Remove context: Context can be removed if you don't find it relevant to the feedback.",
    "Break the problem into subtasks: Consider breaking a complex problem into smaller, manageable subproblems that collectively address the main issue.",
    "Define roles: Clearly specify the role the model should assume when generating a response.",
    "Remove roles: Remove the role definition if you don't find it relevant to the feedback.",
    "Add constraints: Introduce constraints to guide the model's responses more effectively.",
    "Remove constraints: Remove constraints if you don't find them relevant to the feedback.",
    "Include negative examples: Provide examples of undesirable responses to help the model learn from mistakes.",
    "Add verification steps: Include steps for the model to verify its own answers.",
    "Remove verification steps: If the prompt includes verification steps, consider removing them.",
    "Add including edge cases: Encourage the model to consider edge cases in its responses.",
    "Remove including edge cases: If the prompt includes edge cases, consider removing them.",
    "Add quality criteria: Specify the criteria that the model's responses should meet.",
    "Remove quality criteria: If the prompt includes quality criteria, consider removing them.",
    "Add requesting explanations: Ask the model to explain its reasoning or thought process.",
    "Remove requesting explanations: If the prompt includes requesting explanations, consider removing them.",
    "Add decision trees: Write a prompt such that the model goes through a series of decisions to reach a conclusion.",
    "Remove decision trees: If the prompt includes a decision tree process, consider removing it.",
    "Other: This is a catch-all for any other improvement action that may not fit the above categories.",
]

##############################################
# Epsilon-greedy strategy for prompt selection
##############################################

# State definition
class PromptEngineerState(TypedDict):
    current_prompt: PromptTemplateData
    #evaluation_dataset: List[Dict[str, Any]]
    performance_metrics_string: str
    performance_metrics: Dict[str, float]
    failure_analysis: List[Dict[str, Any]]
    improvement_actions: List[str]
    iteration_count: int
    best_prompt: PromptTemplateData
    best_score: float
    search_history: List[Dict[str, Any]]
    selected_action: str
    epsilon_choice: str  # "explore" or "exploit"
    selected_action_reason: str  # Add this missing field

# Define the output formats for LLM nodes
class ActionOutputFormat(BaseModel):
    """Output format for the action selection step."""
    cot: str = Field(..., description="Chain of thought reasoning for the selected action.")
    action: str = Field(..., description="The selected action to apply to the current prompt. You must return the complete action text as provided in the list of improvement actions.")
    action_reason: str = Field(..., description="Reason for selecting this action.")

class ActionApplicationOutputFormat(BaseModel):
    """Output format for the action application step."""
    cot: str = Field(..., description="Chain of thought reasoning for the action application.")
    new_prompt: str = Field(..., description="The new prompt after applying the selected action.")

# Define the action selection and action execution prompts
# Define the action selection and action execution prompts

movie_review_action_selection_prompt = PromptTemplateData(
        prompt="""Select an improvement action based on failure analysis: 
        Task:
        The prompt that you are engineering is for sentiment analysis of movie reviews. If the movie review is positive, it should return true, otherwise it should return false.
        Current Prompt:
        {current_prompt}
        Improvement Actions:
        {improvement_actions}
        Performance Metrics:
        {performance_metrics}
        Some Failure Examples:
        {failure_cases}
        """,
        system_prompt=f"""You are an expert prompt engineer. Your task is to optimize prompts for the given task.
        You will be provided with a current prompt, performance metrics for the task, and a set of improvement actions.
        Your goal is to iteratively improve the prompt based on performance metrics and failure analysis.
        - The performance metrics may help you identify a lack of balance in the model's performance (e.g., precision vs. recall).
        - The failure examples will help you understand some of the specific cases where the model is failing.
        You will select an action to apply to the current prompt. A separate step will apply that action to generate a new prompt.
        Note that these actions will be used later to modify the prompt that an LLM will use to perform the task.
        You should think about why the model is failing on the specific failure cases provided and suggest an action accordingly.
        """,
        output_format=ActionOutputFormat,
        prompt_format_function=kwargs_format_function
    )

movie_review_action_application = PromptTemplateData(
        prompt="""Apply the selected action to the prompt for the following task, and performance metrics:
        Task:
        Sentiment analysis of movie reviews. If the movie review is positive, return true, otherwise return false.
        Selected Action: 
        {selected_action}
        Selected Action Reason:
        {selected_action_reason} 
        Current Prompt: 
        {current_prompt}
        Performance Metrics:
        {performance_metrics}
        Some Failure Examples:
        {failure_cases}""",
        system_prompt=f"""You are a prompt engineer. 
        You are provided with a prompt which you will help improve such that an llm performs well on a given task. 
        Your job is to improve a prompt based on a preselection improvement action and a set of existing performance metrics performance metrics.
        - The performance metrics may help you identify a lack of balance in the model's performance (e.g., precision vs. recall).
        - The failure examples will help you understand some of the specific cases where the model is failing.
        You should provide the new prompt that results from applying the action that you are provided with to the current prompt.
        Make sure the new prompt is clear, concise, and effectively incorporates the selected improvement action.
        Do not change any other aspect of the prompt except for applying the selected improvement action.
        You should think about why the model is failing on the specific failure cases provided and modify the prompt accordingly.
        Make sure that while applying the action, you do not remove the input text place holder at the end of the prompt.
        """,
        output_format=ActionApplicationOutputFormat,
        prompt_format_function=kwargs_format_function
    )

esnli_action_selection_prompt = PromptTemplateData(
        prompt="""Select an improvement action based on failure analysis: 
        Task:
        The prompt that you are engineering is for natural language inference on the e-SNLI dataset. Given a premise and a hypothesis, the model should classify the relationship as entailment, contradiction, or neutral.
        each failure example is formatted as follows:
        Premise: <premise text> | Hypothesis: <hypothesis text> | Predicted: <predicted label> | Actual: <actual label> | Explanation: <explanation of the prediction>
        Current Prompt:
        {current_prompt}
        Improvement Actions:
        {improvement_actions}
        Performance Metrics:
        {performance_metrics}
        Some Failure Examples:
        {failure_cases}
        """,
        system_prompt=f"""You are an expert prompt engineer. Your task is to optimize prompts for the given task.
        You will be provided with a current prompt, performance metrics for the task, and a set of improvement actions.
        Your goal is to iteratively improve the prompt based on performance metrics and failure analysis.
        - The performance metrics may help you identify a lack of balance in the model's performance (e.g., precision vs. recall).
        - The failure examples will help you understand some of the specific cases where the model is failing.
        You will select an action to apply to the current prompt. A separate step will apply that action to generate a new prompt.
        Note that these actions will be used later to modify the prompt that an LLM will use to perform the task.
        You should think about why the model is failing on the specific failure cases provided and suggest an action accordingly.
        """,
        output_format=ActionOutputFormat,
        prompt_format_function=kwargs_format_function
    )

esnli_action_application = PromptTemplateData(
        prompt="""Apply the selected action to the prompt for the following task, and performance metrics:
        Task:
        The prompt that you are engineering is for natural language inference on the e-SNLI dataset. Given a premise and a hypothesis, the model should classify the relationship as entailment, contradiction, or neutral.
        each failure example is formatted as follows:
        Premise: <premise text> | Hypothesis: <hypothesis text> | Predicted: <predicted label> | Actual: <actual label> | Explanation: <explanation of the prediction>
        Selected Action: 
        {selected_action}
        Selected Action Reason:
        {selected_action_reason} 
        Current Prompt: 
        {current_prompt}
        Performance Metrics:
        {performance_metrics}
        Some Failure Examples:
        {failure_cases}""",
        system_prompt=f"""You are a prompt engineer. 
        You are provided with a prompt which you will help improve such that an llm performs well on a given task. 
        Your job is to improve a prompt based on a preselection improvement action and a set of existing performance metrics performance metrics.
        - The performance metrics may help you identify a lack of balance in the model's performance (e.g., precision vs. recall).
        - The failure examples will help you understand some of the specific cases where the model is failing.
        You should provide the new prompt that results from applying the action that you are provided with to the current prompt.
        Make sure the new prompt is clear, concise, and effectively incorporates the selected improvement action.
        Do not change any other aspect of the prompt except for applying the selected improvement action.
        You should think about why the model is failing on the specific failure cases provided and modify the prompt accordingly.
        Make sure that while applying the action, you do not remove the input text place holder at the end of the prompt.
        """,
        output_format=ActionApplicationOutputFormat,
        prompt_format_function=kwargs_format_function
    )

##############################################
# RAG Prompt
##############################################

class RAGOutputFormat(BaseModel):
    """Output format for the action application step."""
    cot: str = Field(..., description="Chain of thought reasoning for the query generation.")
    query_1: str = Field(..., description="first query to retrieve relevant documents.")
    query_2: str = Field(..., description="second query to retrieve relevant documents.")
    query_3: str = Field(..., description="third query to retrieve relevant documents.")


rag_prompt = PromptTemplateData(
        prompt="""Apply the selected action to the prompt for the following task, and performance metrics:
        Task:
        {task_description}
        Current Prompt: 
        {current_prompt}
        Some Failure Examples:
        {failure_cases}""",
        system_prompt=f"""You are helping a prompt engineer by retrieving relevant documents that can help improve a prompt for a given task. 
        You will be provided with a current prompt, a set of failure cases, and the task description.
        Your job is to generate 3 key questions that can help retrieve relevant documents from a knowledge base. These documents will be used to improve the prompt.
        Make sure the questions are clear, concise, and relevant to the task and failure cases.
        """,
        output_format=RAGOutputFormat,
        prompt_format_function=kwargs_format_function
    )

rag_output_evaluator = PromptTemplateData(
        prompt="""Apply the selected action to the prompt for the following task, and performance metrics:
        Task:
        {task_description}
        Current Prompt: 
        {current_prompt}
        Some Failure Examples:
        {failure_cases}""",
        system_prompt=f"""You are helping a prompt engineer by retrieving relevant documents that can help improve a prompt for a given task. 
        You will be provided with a current prompt, a set of failure cases, and the task description.
        Your job is to generate 3 key questions that can help retrieve relevant documents from a knowledge base. These documents will be used to improve the prompt.
        Make sure the questions are clear, concise, and relevant to the task and failure cases.
        """,
        output_format=RAGOutputFormat,
        prompt_format_function=kwargs_format_function
    )
##############################################
# Prompt Optimizer Class
##############################################
class PromptOptimizer:
    def __init__(
        self,
        #failure_analysis: PromptTemplateData,
        action_selection: PromptTemplateData,
        action_application: PromptTemplateData,
        initial_prompt: PromptTemplateData,
        evaluation_method: Callable[[PromptTemplateData, List[Any]], Dict[str, float]],
        training_dataset: List[Any],
        action_list: List[str] = IMPROVEMENT_ACTIONS,
        train_test_ratio: float = 0.5,
        epsilon: float = EPSILON,  # exploration rate
        epsilon_decay: float = 0.98,  # decay rate for epsilon
        min_epsilon: float = 0,  # minimum exploration rate
        drop_out: float = DROP_OUT,  # dropout rate for improvement actions
        max_iterations: int = 20,
        model: str = "gpt-4o",
        rag: bool = False,
    ):
        #self.failure_analysis = failure_analysis
        self.action_selection = action_selection
        self.action_application = action_application
        self.initial_action_list = action_list
        self.action_list = action_list
        self.initial_prompt = initial_prompt
        self.evaluation_method = evaluation_method
        self.training_dataset, self.evaluation_dataset = train_test_split(
            training_dataset, test_size=1 - train_test_ratio, random_state=42
        )

        # Epsilon-greedy parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon if epsilon > 0 else 0

        # Dropout parameters
        self.drop_out = drop_out
        # Max iterations
        self.max_iterations = max_iterations
        # LLM model
        self.model = model

        # Build the workflow
        self.workflow = StateGraph(PromptEngineerState)
        self.workflow.add_node("evaluate", self.evaluate_prompt_node)
        #self.workflow.add_node("analyze_failures", self.analyze_failures_node)
        self.workflow.add_node("epsilon_greedy_choice", self.epsilon_greedy_choice_node)
        if rag:
            self.workflow.add_node("rag_retrieve", self.rag_retrieve_node)
        self.workflow.add_node("select_action", self.select_action_node)
        self.workflow.add_node("apply_action", self.apply_action_node)

        #self.workflow.add_edge("evaluate", "analyze_failures")
        self.workflow.add_edge("evaluate", "epsilon_greedy_choice")
        if not rag:
            self.workflow.add_edge("epsilon_greedy_choice", "select_action")
        else:
            self.workflow.add_edge("epsilon_greedy_choice", "rag_retrieve")
            self.workflow.add_edge("rag_retrieve", "select_action")
        self.workflow.add_edge("select_action", "apply_action")
        self.workflow.add_conditional_edges("apply_action", self.should_continue)
        self.workflow.set_entry_point("evaluate")
        self.app = self.workflow.compile()

    def rag_retrieve_node(self, state: PromptEngineerState) -> PromptEngineerState:
        # Placeholder for RAG retrieval logic
        print("🔍 Retrieving relevant documents using RAG")
        # For now, we just pass the state through
        return state

    def evaluate_prompt_node(self, state: PromptEngineerState) -> PromptEngineerState:
        print(f"current prompt: {state['current_prompt'].prompt}")
        print("========================================")
        print(f"🔍 Evaluating prompt (iteration {state['iteration_count']})")
        
        # Generate metrics
        metrics = self.evaluation_method(state["current_prompt"], self.training_dataset)
        validation_metrics = self.evaluation_method(state["current_prompt"], self.evaluation_dataset)
        accuracy = metrics.get("accuracy", 0)
        recall = metrics.get("recall", 0)
        precision = metrics.get("precision", 0)
        false_positive_rate = metrics.get("false_positive_rate", 0)
        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        # calculate f1 score for validation dataset
        if validation_metrics.get("precision", 0) + validation_metrics.get("recall", 0) > 0:
            validation_f1_score = 2 * (validation_metrics.get("precision", 0) * validation_metrics.get("recall", 0)) / (validation_metrics.get("precision", 0) + validation_metrics.get("recall", 0))
        else:
            validation_f1_score = 0
        # Extract failure cases
        if "failure_cases" in metrics:  # Ensure failure_cases is in metrics
            failure_cases = metrics["failure_cases"]
        else:
            failure_cases = []
        metrics = {
            "validation_accuracy": validation_metrics.get("accuracy", 0),
            "validation_recall": validation_metrics.get("recall", 0),
            "validation_precision": validation_metrics.get("precision", 0),
            "validation_f1_score": validation_f1_score,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            #"false_positive_rate": false_positive_rate,
            "failure_cases": failure_cases
        }
        
        # Track best prompt globally
        new_best_prompt = state["best_prompt"]
        new_best_score = state["best_score"]
        
        if validation_f1_score > state["best_score"]:
            new_best_prompt = state["current_prompt"]
            new_best_score = validation_f1_score
            print(f"🎉 New best score: {validation_f1_score:.3f}")

        #performance_metrics_string = ''
        #for item in metrics:
        #    performance_metrics_string += f"{item}:\n {metrics[item]}\n"
        
        performance_metrics_string = ''
        for item in metrics:
            performance_metrics_string += f"{item}:\n {metrics[item]}\n"
        
        return {
            **state,
            "performance_metrics_string": performance_metrics_string,
            "performance_metrics": metrics,
            "best_prompt": new_best_prompt,
            "best_score": new_best_score
        }

    def epsilon_greedy_choice_node(self, state: PromptEngineerState) -> PromptEngineerState:
        """Epsilon-greedy choice: continue current prompt with prob epsilon, or use best prompt with prob 1-epsilon"""
        current_epsilon = max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** state["iteration_count"]))
        
        if random.random() < current_epsilon:
            # Explore: continue with current prompt
            print(f"🔍 Exploring - continuing with current prompt (ε={current_epsilon:.3f})")
            choice = "explore"
            chosen_prompt = state["current_prompt"]
        else:
            # Exploit: use best known prompt
            print(f"💰 Exploiting - using best known prompt (ε={current_epsilon:.3f})")
            choice = "exploit"
            chosen_prompt = state["best_prompt"] if state["best_prompt"] else state["current_prompt"]
        
        return {
            **state,
            "current_prompt": chosen_prompt,
            "epsilon_choice": choice
        }

    def select_action_node(self, state: PromptEngineerState) -> PromptEngineerState:
        print("🤖 Selecting improvement action")
        
        metrics = state["performance_metrics"]
        
        # LLM based action selection
        # Randomly drop self.drop_out portion of the actions
        if self.drop_out > 0:
            num_to_keep = int(len(self.initial_action_list) * (1 - self.drop_out))
            num_to_keep = max(1, num_to_keep)  # Ensure at least one action remains
            self.action_list = random.sample(self.initial_action_list, num_to_keep)
        else:
            self.action_list = self.initial_action_list.copy()
        failure_cases = random.sample(state["performance_metrics"].get("failure_cases", []), min(len(state["performance_metrics"].get("failure_cases", [])), NUM_FAILURE_CASES_TOSHOW)) if state["performance_metrics"].get("failure_cases") else []
        
        random.shuffle(self.action_list) # We shuffle to debias the llm's potential location bias
        
        # Using validation metrics but training failure cases to avoid overfitting
        llm_action_output = llm_call(
            model=self.model,
            prompt=self.action_selection(
            current_prompt=state["current_prompt"].prompt,
            improvement_actions='\n'.join([f"> {action}" for action in self.action_list]),
            performance_metrics=json.dumps({k: v for k, v in metrics.items() if k.startswith("validation_")}, indent=2),
            failure_cases='\n'.join(failure_cases) if failure_cases else "None"
            ))
        

        selected_action = llm_action_output.action
        assert selected_action is not None, "LLM did not return a valid action. Please check the action selection prompt and the LLM response format."

        selected_action_reason = llm_action_output.action_reason
        assert selected_action_reason is not None, "LLM did not return a valid action reason. Please check the action selection prompt and the LLM response format."

        print(f"🎯 Selected action: {selected_action}")
        return {
            **state,
            "selected_action": selected_action,
            "selected_action_reason": selected_action_reason
        }

    def apply_action_node(self, state: PromptEngineerState) -> PromptEngineerState:
        print(f"⚡ Applying action: {state['selected_action']}")
        action = state["selected_action"]
        selected_action_reason = state["selected_action_reason"]
        current_prompt = state["current_prompt"].prompt  # Extract the prompt string
        metrics = state["performance_metrics"]
        failure_cases = random.sample(state["performance_metrics"].get("failure_cases", []), min(len(state["performance_metrics"].get("failure_cases", [])), NUM_FAILURE_CASES_TOSHOW)) if state["performance_metrics"].get("failure_cases") else []
        
        # Using validation metrics but training failure cases to avoid overfitting
        llm_new_prompt_output = llm_call(
            model=self.model,
            prompt=self.action_application(
                selected_action=action,
                selected_action_reason = selected_action_reason,
                current_prompt=current_prompt,
                performance_metrics=json.dumps({k: v for k, v in metrics.items() if k.startswith("validation_")}, indent=2), 
                failure_cases='\n'.join(failure_cases) if failure_cases else "None"
                ))
        new_prompt_str = llm_new_prompt_output.new_prompt
        assert new_prompt_str is not None, "LLM did not return a valid new prompt. Please check the action application prompt and the LLM response format."

        # Create new PromptTemplateData object with the updated prompt
        new_prompt = PromptTemplateData(
            prompt=new_prompt_str,
            system_prompt=state["current_prompt"].system_prompt,
            output_format=state["current_prompt"].output_format,
            prompt_format_function=state["current_prompt"].prompt_format_function
        )

        new_history = state["search_history"] + [{
            "iteration": state["iteration_count"],
            "action": action,
            "prompt": new_prompt_str,
            "metrics": state["performance_metrics"], 
        }]

        print(f"iteration: {state['iteration_count']} | action: {action} | training f1_score: {state['performance_metrics']['f1_score']:.3f} | validation f1_score: {state['performance_metrics']['validation_f1_score']:.3f}")
        
        return {
            **state,
            "current_prompt": new_prompt,
            "iteration_count": state["iteration_count"] + 1,
            "search_history": new_history
        }

    def should_continue(self, state: PromptEngineerState) -> Literal["evaluate", "__end__"]:
        if state["iteration_count"] >= self.max_iterations:
            print(f"🛑 Max iterations ({self.max_iterations}) reached")
            return "__end__"
        if state["performance_metrics"]["f1_score"] >= TARGET_SCORE:
            print(f"🎯 Target score ({TARGET_SCORE}) achieved!")
            return "__end__"
        return "evaluate"

    def visualize(self):
        try:
            from IPython.display import Image, display
            # Try to use local rendering first to avoid API issues
            try:
                from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
                display(Image(self.app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.PYPPETEER
                )))
            except:
                # Fallback to API with retry settings
                display(Image(self.app.get_graph().draw_mermaid_png(
                    max_retries=5, 
                    retry_delay=2.0
                )))
        except ImportError:
            print("To visualize the graph, install: pip install grandalf")
            print("Or use: app.get_graph().print_ascii()")
            print("\nWorkflow Graph (ASCII):")
            self.app.get_graph().print_ascii()
        except Exception as e:
            print(f"Graph visualization failed: {e}")
            print("Falling back to ASCII representation:")
            self.app.get_graph().print_ascii()
    
    def run(self) -> PromptEngineerState:
        initial_state: PromptEngineerState = {
            "current_prompt": self.initial_prompt,
            "performance_metrics_string": "",
            "performance_metrics": {},
            "failure_analysis": [],
            "improvement_actions": [],
            "iteration_count": 0,
            "best_prompt": self.initial_prompt,
            "best_score": 0.0,
            "search_history": [],
            "selected_action": "",
            "selected_action_reason": "",
            "epsilon_choice": ""
        }
        final_state = self.app.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})
        return final_state
    
##############################################
# Helper function
##############################################
    
def save_final_state(final_state: Dict[str, Any], EPSILON: float, DROP_OUT: float, MAX_ITERATIONS: int):
    # Create filename with metadata
    filename = f"epsilon_{EPSILON}_max_{DROP_OUT}_iter_{MAX_ITERATIONS}.json"

    # Convert final_state to a serializable format
    serializable_state = {}
    for key, value in final_state.items():
        if key in ['current_prompt', 'best_prompt']:
            # Convert PromptTemplateData objects to dictionaries, excluding output_format
            serializable_state[key] = {
                'prompt': value.prompt,
                'system_prompt': value.system_prompt
            }
        else:
            serializable_state[key] = value

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(serializable_state, f, indent=2)

    print(f"Final state saved to: {filename}")

def plot_validation_f1_scores():
    # Find all JSON files matching the pattern
    json_files = glob.glob("epsilon_*_max_*_iter_*.json")
    
    if not json_files:
        print("No JSON files found matching the pattern 'epsilon_*_max_*_iter_*.json'")
        return
    
    plt.figure(figsize=(12, 8))
    
    for json_file in json_files:
        try:
            # Extract epsilon and dropout from filename
            # Pattern: epsilon_{EPSILON}_max_{DROP_OUT}_iter_{MAX_ITERATIONS}.json
            match = re.search(r'epsilon_([0-9.]+)_max_([0-9.]+)_iter_([0-9]+)\.json', json_file)
            if match:
                epsilon = float(match.group(1))
                dropout = float(match.group(2))
                max_iter = int(match.group(3))
            else:
                print(f"Could not parse filename: {json_file}")
                continue
            
            # Load the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract validation F1 scores from search history
            iterations = []
            validation_f1_scores = []
            
            for entry in data.get('search_history', []):
                iteration = entry.get('iteration', 0)
                metrics = entry.get('metrics', {})
                val_f1 = metrics.get('validation_f1_score', 0)
                
                iterations.append(iteration)
                validation_f1_scores.append(val_f1)
            
            # Create label for the plot
            label = f"ε={epsilon}, dropout={dropout}"
            
            # Plot the line
            plt.plot(iterations, validation_f1_scores, marker='o', markersize=4, 
                    label=label, linewidth=2, alpha=0.8)
            
            print(f"Loaded {json_file}: {len(iterations)} iterations, "
                  f"best validation F1: {max(validation_f1_scores) if validation_f1_scores else 0:.3f}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Customize the plot
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Validation F1 Score', fontsize=12)
    plt.title('Validation F1 Score Over Iterations\nby Epsilon and Dropout Parameters', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Also save the plot
    plt.savefig('validation_f1_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'validation_f1_comparison.png'")

def plot_cumulative_max_f1_scores():
    # Find all JSON files matching the pattern
    json_files = glob.glob("epsilon_*_max_*_iter_*.json")
    
    if not json_files:
        print("No JSON files found matching the pattern 'epsilon_*_max_*_iter_*.json'")
        return
    
    plt.figure(figsize=(12, 8))
    
    for json_file in json_files:
        try:
            # Extract epsilon and dropout from filename
            match = re.search(r'epsilon_([0-9.]+)_max_([0-9.]+)_iter_([0-9]+)\.json', json_file)
            if match:
                epsilon = float(match.group(1))
                dropout = float(match.group(2))
                max_iter = int(match.group(3))
            else:
                print(f"Could not parse filename: {json_file}")
                continue
            
            # Load the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract validation F1 scores from search history
            iterations = []
            validation_f1_scores = []
            
            for entry in data.get('search_history', []):
                iteration = entry.get('iteration', 0)
                metrics = entry.get('metrics', {})
                val_f1 = metrics.get('validation_f1_score', 0)
                
                iterations.append(iteration)
                validation_f1_scores.append(val_f1)
            
            if not validation_f1_scores:
                print(f"No validation F1 scores found in {json_file}")
                continue
            
            # Calculate cumulative maximum - THIS IS THE KEY LINE!
            cumulative_max_scores = np.maximum.accumulate(validation_f1_scores)
            
            # Create label for the plot
            label = f"ε={epsilon}, dropout={dropout}"
            
            # Plot the CUMULATIVE MAXIMUM line (not the raw scores)
            plt.plot(iterations, cumulative_max_scores, marker='o', markersize=4, 
                    label=label, linewidth=2, alpha=0.8)
            
            print(f"Loaded {json_file}: {len(iterations)} iterations, "
                  f"final best F1: {cumulative_max_scores[-1]:.3f}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Customize the plot
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Validation F1 Score So Far', fontsize=12)
    plt.title('Cumulative Best Validation F1 Score Over Iterations\nby Epsilon and Dropout Parameters', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Also save the plot
    plt.savefig('cumulative_max_f1_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'cumulative_max_f1_comparison.png'")