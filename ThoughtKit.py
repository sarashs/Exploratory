from typing import Any
from pydantic import BaseModel, Field
from openai import OpenAI
from collections import deque
import numpy as np

# PARAMETERS
DEBUG = False
#--------------------------------------------------------------------------------
# Structure for the outputs

class Status(BaseModel):
    status: str = Field(description="The status of the chain of thoughts. Either we should continue to reach a final asnwer, we should terminate this chain because it won't reach a final answer and it is going stray. If a we are ready for a final answer (regardless of correctness) return ready", enum=["continue", "terminate", "ready"])

class Thought(BaseModel):
    """A thought/step on how to solve the problem"""
    thought: str = Field(description="The next step or thought to solve the problem")

class Solution(BaseModel):
    """A solution to the problem"""
    solution: str = Field(description="The solution to the problem")

class SolutionEvaluation(BaseModel):
    """Evaluate and compare two solutions"""
    reason: str = Field(description="Describe your reasoning for the evaluation")
    choice: str = Field(description="The best solution", enum=["solution1", "solution2"])

class Equivalence(BaseModel):
    """Check if two solutions are equivalent"""
    equivalent: bool = Field(description="True if the two solutions are equivalent, False otherwise")

# OpenAI models low T, high T, and evaluator
class OpenAIParse(object):

    def __init__(self, model, response_format, system_prompt):
        self.client = OpenAI()
        self.model = model
        self.response_format= response_format
        self.system_prompt = system_prompt

    def __call__(self, prompt, temperature = 0):     
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format=self.response_format,
            #logprobs=True,
            #top_logprobs=3,
        )
        return completion.choices[0].message.parsed
#--------------------------------------------------------------------------------
Thought_system_prompt = """Given the user query and the chain of thoughts, generate the next step to solve the problem but do not generate the final solution.
Do not repeat the same step twice. Your step/thought must follow the previous Chain of thoughts. If you are ready to generate a solution, return I am ready.
You must use your own abilities and knowledge to generate the next step. Remember to generate only one step/thought at a time.
Here is an example of a chain of thoughts:
User query: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
Chain of thoughts:
- Roger has 5 tennis balls.
- 2 cans of 3 tennis balls each is 6 tennis balls.
- 5 + 6 = 11 tennis balls.
- I am ready.
"""
Thought_user_prompt = "User query: {query}\nChain of thoughts: {chain_of_thoughts}"
Thought_model = OpenAIParse("gpt-4o-mini", Thought, Thought_system_prompt)
#--------------------------------------------------------------------------------
Status_system_prompt = """
Given the user query and the chain of thoughts, evaluate the chain of thoughts and determine the status of the chain of thoughts.
The chain of thoughts is a series of thoughts that are generated to solve a problem.
The chain of thoughts can be in one of three states: continue, terminate, ready."""
Status_user_prompt = "User query: {query}\nChain of thoughts: {chain_of_thoughts}"
Status_model = OpenAIParse("gpt-4o-mini", Status, Status_system_prompt)
#--------------------------------------------------------------------------------
Solution_system_prompt = """
Given the user query and the chain of thoughts, generate the solution to the problem.
The solution is the final answer to the problem."""
Solution_user_prompt = "User query: {query}\nChain of thoughts: {chain_of_thoughts}"
Solution_model = OpenAIParse("gpt-4o-mini", Solution, Solution_system_prompt)
#--------------------------------------------------------------------------------
Equivalence_system_prompt = """
Given two solutions and a user query, evaluate if the two solutions are equivalent."""
Equivalence_user_prompt = "User query: {query}\nSolution 1: {solution_1}\nSolution 2: {solution_2}"
Equivalence_model = OpenAIParse("gpt-4o-mini", Equivalence, Equivalence_system_prompt)
#--------------------------------------------------------------------------------
Compare_system_prompt = """
Given the user query, chain of thought and solution 1 and, chain of thought and solution 2, evaluate the solutions and determine the best solution.
Note that the solutions are already generated and you are evaluating them. You should consider the logical reasoning, logical flow, and correctness of the solutions."""
Compare_user_prompt = "User query: {query}\nSolutions 1: {solutions_1}\nSolutions 2: {solutions_2}"
Compare_model = OpenAIParse("gpt-4o-mini", SolutionEvaluation, Compare_system_prompt)
#--------------------------------------------------------------------------------
# chain of thoughts

class link(object):
    """A link in a chain of thoughts"""
    def __init__(self, thought, prev = None , status: Status = None):
        self.thought = thought
        self.thoughts = [thought]
        self.status = status
        self.next = None
        self.prev = None
        if prev:
            self.prev = prev
            self.thoughts = prev.thoughts + [thought]

class Chain(object):
    """A chain of thoughts to solve a problem"""
    def __init__(self, status_model, thought_model, solution_model):
        self.head = None
        self.tail = None
        self.status_model = status_model
        self.thought_model = thought_model
        self.solution_model = solution_model
        self.solution = None

    def set_status(self, node, query):
        """Set the status of the chain of thoughts"""
        node.status = self.status_model(
            Status_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts))
            )

    def __call__(self, query, T = 0 , max_length: int = 5):
        self.head = link(thought = query, status = Status(status="continue"))
        node = self.head
        for i in range(max_length):
            if DEBUG:
                print(f"Current Length: {i} out of {max_length}")
            thought = self.thought_model(
                Thought_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts)), 
                temperature = T)
            node.next = link(thought = thought.thought, prev = node)
            node = node.next
            self.set_status(node, query)
            if node.status.status == "terminate" or node.status.status == "ready":
                break
        self.tail = node
        self.solution = self.solution_model(
            Solution_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(self.tail.thoughts))
            ).solution
        return self.solution

class AnnealChain(Chain):
    """A chain of thoughts to solve a problem using annealing"""
    def __init__(self, status_model, thought_model, solution_model):
        super().__init__(status_model, thought_model, solution_model)

    def __call__(self, query, T_init = 1.5, max_length: int = 5):
        """Solve the problem using annealing"""
        self.head = link(thought = query, status = Status(status="continue"))
        node = self.head
        T_list = np.linspace(T_init, 0.0, max_length).tolist()
        for T in T_list:
            if DEBUG:
                print(f"Current temperature: {T} from {T_init}")
            thought = self.thought_model(
                Thought_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts)), 
                temperature = T)
            node.next = link(thought = thought.thought, prev = node)
            node = node.next
            self.set_status(node, query)
            if node.status.status == "terminate" or node.status.status == "ready":
                break
        self.tail = node
        self.solution = self.solution_model(
            Solution_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(self.tail.thoughts))
            ).solution
        return self.solution
#--------------------------------------------------------------------------------

# tree of thoughts

class Node(object):
    """A node in a tree of thoughts"""
    def __init__(self, thought, parent = None, status: Status = None):
        self.thought = thought
        self.thoughts = [thought]
        self.status = status
        self.children = []
        self.parent = parent
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
            self.parent = parent
            self.thoughts = parent.thoughts + [thought]

class BinNode(Node):
    """A node in a binary tree of thoughts"""
    def __init__(self, thought, parent = None, status: Status = None, low_temp = None, high_temp = None, node_type = "Hot"):
        super().__init__(thought, parent, status)
        self.node_type_chain = [node_type] # list of node types to keep track of the path
        self.low_temp = low_temp
        self.high_temp = high_temp
        if parent:
            self.node_type_chain = parent.node_type_chain + [node_type]

class Tree(object):
    """A tree of thoughts to solve a problem"""
    def __init__(self, status_model, thought_model, solution_model,
                 equivalence_model = None, best_solution_model = None, max_depth: int = 5, max_children: int = 2):
        self.root = None
        self.leaves = deque([self.root]) # list of leaves
        self.status_model = status_model
        self.solution_nodes = []
        self.solutions = []
        self.solution_thoughts = []
        self.max_depth = max_depth
        self.max_children = max_children
        self.thought_model = thought_model
        self.solution_model = solution_model
        self.equivalence_model = equivalence_model
        self.best_solution_model = best_solution_model

    def set_status(self, query):
        """Set the status of the chain of thoughts"""
        for node in self.leaves:
            if not node.status:
                node.status = self.status_model(
                    Status_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts))
                    )

    def prune(self):
        """Prune the tree by removing nodes that are ready to generate a solution or are terminated due to going stray."""
        removal = []
        for node in self.leaves:
            if node.status.status == "terminate":
                removal.append(node)
            elif node.status.status == "ready":
                self.solution_nodes.append(node)
                removal.append(node)
        for node in removal:
            self.leaves.remove(node)

    def check_equivalence(self, query, solution_1, solution_2):
        """Check if two solutions are equivalent"""
        return self.equivalence_model(
            Equivalence_user_prompt.format(query=query, solution_1=solution_1, solution_2=solution_2)).equivalent
    
    def unique_solutions(self, query):
        """Remove duplicate solutions"""
        unique_solutions = []
        unique_solutions_thoughts = []
        for num, solution in enumerate(self.solutions):
            if not unique_solutions:
                unique_solutions.append(solution)
                unique_solutions_thoughts.append(self.solution_thoughts[num])
            else:
                for unique_solution in unique_solutions:
                    if self.check_equivalence(query, solution, unique_solution):
                        break
                else:
                    unique_solutions.append(solution)
                    unique_solutions_thoughts.append(self.solution_thoughts[num])
        self.solutions = unique_solutions
        self.solution_thoughts = unique_solutions_thoughts

    def compare_solution(self, query, solutions_1, solutions_2):
        """Evaluate and compare two solutions"""
        return self.best_solution_model(
            Compare_user_prompt.format(query=query, solutions_1=solutions_1, solutions_2=solutions_2)
            ).choice
    
    def best_solution(self, query):
        """Find the best solution among the unique solutions"""
        if len(self.solutions) == 1:
            return (self.solution_thoughts[0], self.solutions[0]) if self.solution_thoughts else ([] , self.solutions[0])
        else:
            best = self.solutions[0]
            best_thoughts = self.solution_thoughts[0]
            for i in range(1, len(self.solutions)):
                best_thoughts, best = (
                    (self.solution_thoughts[i], self.solutions[i]) if self.compare_solution(
                        query, best, '\n-'.join(self.solution_thoughts[i])+"\n"+self.solutions[i]
                        ) == "solution2" else (best_thoughts, best)
                        )
            return (best_thoughts, best)

    def __call__(self, query, T=0):
        self.root = Node(thought = query, status = Status(status="continue"))
        node = self.root
        self.leaves = deque([self.root])
        for depth in range(self.max_depth):
            while self.leaves and node.depth <= depth:
                if DEBUG:
                    print(f"Current Depth: {node.depth} out of {self.max_depth}")
                node = self.leaves.popleft()
                for i in range(self.max_children):
                    thought = self.thought_model(
                        Thought_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts)), 
                        temperature = T)
                    node.children.append(Node(thought = thought.thought, parent = node))
                self.leaves.extend(node.children)
            self.set_status(query)
            self.prune()
        if self.solution_nodes:
            self.solutions = [self.solution_model(
                Solution_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts))
                ).solution for node in self.solution_nodes]
            self.solution_thoughts = [node.thoughts for node in self.solution_nodes]
        elif self.leaves:
            self.solutions = [self.solution_model(
                Solution_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts))
                ).solution for node in self.leaves]
            self.solution_thoughts = [node.thoughts for node in self.leaves]
        else:
            self.solutions = ["No solution found"]
        if self.equivalence_model:
            self.unique_solutions(query)
        if self.best_solution_model:
            best_thoughts, best = self.best_solution(query)
        if not best:
            return None, self.solutions
        return best_thoughts, best

class BinTree(Tree):
    """A binary tree of thoughts to solve a problem"""
    def __init__(
            self, status_model, thought_model, solution_model,
            equivalence_model = None, best_solution_model = None,
            max_depth: int = 5, low_temp = 0, high_temp = 1.2
            ):
        super().__init__(
            status_model=status_model, thought_model=thought_model,
            solution_model=solution_model, equivalence_model=equivalence_model,
            best_solution_model=best_solution_model, max_depth=max_depth
            )
        self.root = None
        self.leaves = deque([self.root]) # list of leaves
        self.low_temp = low_temp
        self.high_temp = high_temp

    def __call__(self, query):
        self.root = BinNode(thought = query, status = Status(status="continue"))
        node = self.root
        self.leaves = deque([self.root])
        for depth in range(self.max_depth):
            while self.leaves and node.depth <= self.max_depth:
                if DEBUG:
                    print(f"Current Depth: {node.depth} out of {self.max_depth}")
                node = self.leaves.popleft()
                thought = self.thought_model(
                    Thought_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts)), 
                    temperature = self.low_temp)
                node.low_temp = BinNode(thought = thought.thought, parent = node, node_type = "Cold")
                thought = self.thought_model(
                    Thought_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts)), 
                    temperature = self.high_temp)
                node.high_temp = BinNode(thought = thought.thought, parent = node, node_type = "Hot")
                self.leaves.extend([node.low_temp, node.high_temp])
            self.set_status(query)
            self.prune()
        if self.solution_nodes:
            self.solutions = [self.solution_model(
                Solution_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts))
                ).solution for node in self.solution_nodes]
            self.solution_thoughts = [node.thoughts for node in self.solution_nodes]
        elif self.leaves:
            self.solutions = [self.solution_model(
                Solution_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts))
                ).solution for node in self.leaves]
            self.solution_thoughts = [node.thoughts for node in self.leaves]
        else:
            self.solutions = ["No solution found"]        
        if self.equivalence_model:
            self.unique_solutions(query)
        if self.best_solution_model:
            best_thoughts, best = self.best_solution(query)
        if not best:
            return None, self.solutions
        return best_thoughts, best
    
class AnnealTree(Tree):
    """A tree of thoughts to solve a problem using annealing"""
    def __init__(self, status_model, thought_model, solution_model,
                 equivalence_model = None, best_solution_model = None, max_depth: int = 5):
        super().__init__(status_model, thought_model, solution_model, equivalence_model, best_solution_model, max_depth)

    def __call__(self, query, T_init = 1.5):
        """Solve the problem using annealing"""
        self.root = Node(thought = query, status = Status(status="continue"))
        node = self.root
        self.leaves = deque([self.root])
        temp_list = np.linspace(T_init, 0.0, self.max_depth).tolist()
        for depth, T in zip(range(self.max_depth), temp_list):
            while self.leaves and node.depth <= depth:
                if DEBUG:
                    print(f"Current Depth: {node.depth} out of {self.max_depth}; Temperature: {T}")
                node = self.leaves.popleft()
                for i in range(self.max_children):
                    thought = self.thought_model(
                        Thought_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts)), 
                        temperature = T)
                    node.children.append(Node(thought = thought.thought, parent = node))
                self.leaves.extend(node.children)
            self.set_status(query)
            self.prune()
        if self.solution_nodes:
            self.solutions = [self.solution_model(
                Solution_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts))
                ).solution for node in self.solution_nodes]
            self.solution_thoughts = [node.thoughts for node in self.solution_nodes]
        elif self.leaves:
            self.solutions = [self.solution_model(
                Solution_user_prompt.format(query=query, chain_of_thoughts='\n-'.join(node.thoughts))
                ).solution for node in self.leaves]
            self.solution_thoughts = [node.thoughts for node in self.leaves]
        else:
            self.solutions = ["No solution found"]
        if self.equivalence_model:
            self.unique_solutions(query)
        if self.best_solution_model:
            best_thoughts, best = self.best_solution(query)
        if not best:
            return None, self.solutions
        return best_thoughts, best