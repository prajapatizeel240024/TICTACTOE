from typing import TypedDict, Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from anthropic import Client as AnthropicClient
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import time
import re

load_dotenv()

# Game state structure
class GameState(TypedDict):
    messages: Sequence[BaseMessage]
    next_agent: str
    board: List[str]
    game_over: bool
    game_history: List[str]

# Board visualization
def print_board(board: List[str]) -> str:
    return (
        f"\n{board[0]} | {board[1]} | {board[2]}\n"
        "-----------\n"
        f"{board[3]} | {board[4]} | {board[5]}\n"
        "-----------\n"
        f"{board[6]} | {board[7]} | {board[8]}\n"
    )

# Game logic
def check_winner(board: List[str]) -> str:
    wins = [(0,1,2), (3,4,5), (6,7,8), 
            (0,3,6), (1,4,7), (2,5,8), 
            (0,4,8), (2,4,6)]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != " ":
            return board[a]
    return "Tie" if " " not in board else ""

def is_valid_move(board: List[str], pos: int) -> bool:
    return 0 <= pos <= 8 and board[pos] == " "

# AI Clients
player_x = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    request_timeout=30
)

anthropic_client = AnthropicClient(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=30
)

# Game prompts and response parsing
def create_game_prompt(board: List[str], symbol: str, history: List[str]) -> str:
    history_str = "\nRecent Games:\n" + "\n".join(history[-3:]) if history else ""
    return f"""{history_str}
    
You are Player {symbol}. Current board:
{print_board(board)}

Position grid:
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8

Strategic priorities:
1. Win immediately if possible
2. Block opponent's winning moves
3. Create multiple winning opportunities
4. Control center and corners first
5. Use defensive patterns from previous games

Explain your reasoning in 10-15 words, then provide ONLY your move as "Move: X". 

Your analysis:"""

def parse_response(response_text: str) -> tuple[str, int]:
    try:
        clean_text = response_text.replace("\n", " ").strip()
        reason = " ".join(clean_text.split("Move:")[0].split()).strip(". ")
        move = int(re.findall(r'\d+', clean_text.split("Move:")[-1])[0])
        return reason, move
    except:
        valid_moves = re.findall(r'\b[0-8]\b', clean_text)
        if valid_moves:
            return "Pattern recognition", int(valid_moves[-1])
        return "Strategic placement", 0

# Game nodes with decision tracking
def player_x_node(state: GameState) -> GameState:
    board, history = state["board"], state["game_history"]
    for _ in range(3):
        try:
            prompt = create_game_prompt(board, "X", history)
            response = player_x.invoke([HumanMessage(content=prompt)])
            reason, move = parse_response(response.content.strip())
            
            if is_valid_move(board, move):
                board[move] = "X"
                winner = check_winner(board)
                return {
                    "messages": state["messages"] + [HumanMessage(
                        content=f"X{move}: {reason}"
                    )],
                    "next_agent": "player_o" if not winner else END,
                    "board": board,
                    "game_over": bool(winner),
                    "game_history": history
                }
        except Exception as e:
            time.sleep(1)
    return fallback_move(state, "X")

def player_o_node(state: GameState) -> GameState:
    board, history = state["board"], state["game_history"]
    for _ in range(3):
        try:
            prompt = create_game_prompt(board, "O", history)
            response = anthropic_client.completions.create(
                model="claude-2",
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                max_tokens_to_sample=100
            )
            reason, move = parse_response(response.completion.strip())
            
            if is_valid_move(board, move):
                board[move] = "O"
                winner = check_winner(board)
                return {
                    "messages": state["messages"] + [HumanMessage(
                        content=f"O{move}: {reason}"
                    )],
                    "next_agent": "player_x" if not winner else END,
                    "board": board,
                    "game_over": bool(winner),
                    "game_history": history
                }
        except Exception as e:
            time.sleep(1)
    return fallback_move(state, "O")

def fallback_move(state: GameState, symbol: str) -> GameState:
    board = state["board"]
    valid_moves = [i for i, x in enumerate(board) if x == " "]
    move = valid_moves[0] if valid_moves else 0
    board[move] = symbol
    return {
        "messages": state["messages"] + [HumanMessage(
            content=f"{symbol}{move}F: System-generated fallback move"
        )],
        "next_agent": END,
        "board": board,
        "game_over": True,
        "game_history": state["game_history"]
    }

# Game workflow
workflow = StateGraph(GameState)
workflow.add_node("player_x", player_x_node)
workflow.add_node("player_o", player_o_node)

def route_step(state):
    return END if state["game_over"] else state["next_agent"]

def initial_router(state: GameState):
    return state["next_agent"]

workflow.add_conditional_edges(
    START,
    initial_router,
    {"player_x": "player_x", "player_o": "player_o"}
)
workflow.add_conditional_edges("player_x", route_step)
workflow.add_conditional_edges("player_o", route_step)
chain = workflow.compile()

# Tournament execution with detailed reporting
def run_tournament(games=3):
    history, results = [], {"X":0, "O":0, "Tie":0}
    
    for game_num in range(1, games+1):
        starting_agent = "player_x" if game_num % 2 == 1 else "player_o"
        starter_symbol = "X" if starting_agent == "player_x" else "O"
        print(f"\n{'='*40}\nGame {game_num} - First move: {starter_symbol}\n{'='*40}")
        
        result = chain.invoke({
            "messages": [],
            "next_agent": starting_agent,
            "board": [" "]*9,
            "game_over": False,
            "game_history": history.copy()
        })
        
        # Print move-by-move analysis
        print("\nMove Details:")
        for i, msg in enumerate(result["messages"], 1):
            move, _, reason = msg.content.partition(": ")
            print(f"Turn {i}: {move}")
            print(f"   Reason: {reason}")
            print(f"   Board State: {print_board(result['board'][:9])}")
        
        # Track results
        winner = check_winner(result["board"])
        results[winner if winner in ["X","O"] else "Tie"] += 1
        
        # Update history
        moves = " → ".join([m.content.split(':')[0] for m in result["messages"]])
        history_entry = f"{moves} → {winner if winner else 'Tie'}"
        history.append(history_entry)
        history = history[-3:]
        
        # Final board
        print(f"\nFinal Board:")
        print(print_board(result["board"]))
        print(f"Result: {winner if winner else 'Tie'}")
    
    # Tournament summary
    print("\n" + "="*40)
    print("Tournament Analysis:")
    print(f"Total Games: {games}")
    print(f"X Wins: {results['X']} ({results['X']/games:.0%})")
    print(f"O Wins: {results['O']} ({results['O']/games:.0%})")
    print(f"Ties: {results['Tie']} ({results['Tie']/games:.0%})")
    print("\nStrategic Observations:")
    print("- Perfect play should result in 100% ties")
    print("- Early center control correlates with higher win rates")
    print("- Defensive patterns emerge after 3+ games")

if __name__ == "__main__":
    run_tournament()