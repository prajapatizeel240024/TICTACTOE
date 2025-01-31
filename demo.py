from typing import TypedDict, Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from anthropic import Client as AnthropicClient
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import time


load_dotenv()

# Define the state structure
class GameState(TypedDict):
    messages: Sequence[BaseMessage]
    next_agent: str
    board: List[str]
    game_over: bool

def print_board(board: List[str]) -> str:
    return f"""
    {board[0]} | {board[1]} | {board[2]}
    -----------
    {board[3]} | {board[4]} | {board[5]}
    -----------
    {board[6]} | {board[7]} | {board[8]}
    """

def check_winner(board: List[str]) -> str:
    # Winning combinations
    wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for w in wins:
        if board[w[0]] == board[w[1]] == board[w[2]] != " ":
            return board[w[0]]
    if " " not in board:
        return "Tie"
    return ""

def is_valid_move(board: List[str], position: int) -> bool:
    return 0 <= position <= 8 and board[position] == " "

# Initialize the AI models with timeouts
player_x = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    request_timeout=30  # Add timeout
)

anthropic_client = AnthropicClient(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=30  # Add timeout
)

def create_game_prompt(board: List[str], player_symbol: str) -> str:
    return f"""You are playing Tic Tac Toe. You are player {player_symbol}.
Current board state:
{print_board(board)}

Valid moves are positions 0-8 (empty spaces only).
Positions are numbered like this:
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8

Provide only the number (0-8) for your next move. Just the number, nothing else."""

def player_x_node(state: GameState) -> GameState:
    board = state["board"]
    max_retries = 3  # Limit retries
    retries = 0
    
    while retries < max_retries:
        try:
            prompt = create_game_prompt(board, "X")
            response = player_x.invoke([HumanMessage(content=prompt)])
            move = int(response.content.strip())
            
            if is_valid_move(board, move):
                board[move] = "X"
                winner = check_winner(board)
                print(f"\nOpenAI placed X at position {move}")  # Add progress indicator
                print(print_board(board))  # Show current board state
                return {
                    "messages": state["messages"] + [HumanMessage(content=f"Player X placed at {move}")],
                    "next_agent": "player_o" if winner == "" else END,
                    "board": board,
                    "game_over": winner != "",
                }
            retries += 1
        except Exception as e:
            print(f"Retry {retries + 1}/{max_retries} for OpenAI...")
            retries += 1
            time.sleep(1)  # Add small delay between retries
    
    # If all retries failed, make a random valid move
    valid_moves = [i for i, val in enumerate(board) if val == " "]
    if valid_moves:
        move = valid_moves[0]
        board[move] = "X"
        winner = check_winner(board)
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Player X placed at {move} (fallback)")],
            "next_agent": "player_o" if winner == "" else END,
            "board": board,
            "game_over": winner != "",
        }

def player_o_node(state: GameState) -> GameState:
    board = state["board"]
    max_retries = 3  # Limit retries
    retries = 0
    
    while retries < max_retries:
        try:
            prompt = create_game_prompt(board, "O")
            response = anthropic_client.completions.create(
                model="claude-2",  # Use claude-2 instead of claude-v1.3
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                max_tokens_to_sample=100,
                temperature=0.7,
                timeout=30
            )
            move = int(response.completion.strip())
            
            if is_valid_move(board, move):
                board[move] = "O"
                winner = check_winner(board)
                print(f"\nAnthropic placed O at position {move}")  # Add progress indicator
                print(print_board(board))  # Show current board state
                return {
                    "messages": state["messages"] + [HumanMessage(content=f"Player O placed at {move}")],
                    "next_agent": "player_x" if winner == "" else END,
                    "board": board,
                    "game_over": winner != "",
                }
            retries += 1
        except Exception as e:
            print(f"Retry {retries + 1}/{max_retries} for Anthropic...")
            retries += 1
            time.sleep(1)  # Add small delay between retries
    
    # If all retries failed, make a random valid move
    valid_moves = [i for i, val in enumerate(board) if val == " "]
    if valid_moves:
        move = valid_moves[0]
        board[move] = "O"
        winner = check_winner(board)
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Player O placed at {move} (fallback)")],
            "next_agent": "player_x" if winner == "" else END,
            "board": board,
            "game_over": winner != "",
        }

# Create the graph
workflow = StateGraph(GameState)

# Add nodes
workflow.add_node("player_x", player_x_node)
workflow.add_node("player_o", player_o_node)

# Define routing logic
def route_step(state: GameState) -> str:
    if state["game_over"]:
        return END
    return state["next_agent"]

# Add edges
workflow.add_edge(START, "player_x")
workflow.add_conditional_edges("player_x", route_step)
workflow.add_conditional_edges("player_o", route_step)

# Compile the graph
chain = workflow.compile()

# Initialize game
initial_state = {
    "messages": [],
    "next_agent": "player_x",
    "board": [" "] * 9,
    "game_over": False,
}

# Run the game
print("\nStarting Tic Tac Toe game between OpenAI (Player X) and Anthropic (Player O)...")
result = chain.invoke(initial_state)

# Print the game history
print("\nGame History:")
for message in result["messages"]:
    print(message.content)

# Print final board state
print("\nFinal Board State:")
print(print_board(result["board"]))

# Determine the winner
winner = check_winner(result["board"])
if winner == "X":
    print("\nPlayer X (OpenAI) wins!")
elif winner == "O":
    print("\nPlayer O (Anthropic) wins!")
else:
    print("\nIt's a tie!")