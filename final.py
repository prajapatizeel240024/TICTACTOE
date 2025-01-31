from typing import TypedDict, Sequence, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from anthropic import Anthropic
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import time
import json
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GameState(TypedDict):
    messages: Sequence[BaseMessage]
    next_agent: str
    board: List[str]
    game_over: bool
    game_history: Dict
    current_game: int
    last_winner: str
    reasoning: List[str]

def print_board(board: List[str]) -> str:
    return f"""
    {board[0]} | {board[1]} | {board[2]}
    -----------
    {board[3]} | {board[4]} | {board[5]}
    -----------
    {board[6]} | {board[7]} | {board[8]}
    """

def check_winner(board: List[str]) -> str:
    wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for w in wins:
        if board[w[0]] == board[w[1]] == board[w[2]] != " ":
            return board[w[0]]
    if " " not in board:
        return "Tie"
    return ""

def is_valid_move(board: List[str], position: int) -> bool:
    try:
        return 0 <= position <= 8 and board[position] == " "
    except (IndexError, ValueError, TypeError):
        return False

def get_valid_moves(board: List[str]) -> List[int]:
    return [i for i, val in enumerate(board) if val == " "]

def analyze_board(board: List[str], player_symbol: str) -> str:
    opponent = "O" if player_symbol == "X" else "X"
    
    # Check for winning moves
    for i in range(9):
        if board[i] == " ":
            temp_board = board.copy()
            temp_board[i] = player_symbol
            if check_winner(temp_board) == player_symbol:
                return f"Taking position {i} to win the game"
    
    # Block opponent's winning moves
    for i in range(9):
        if board[i] == " ":
            temp_board = board.copy()
            temp_board[i] = opponent
            if check_winner(temp_board) == opponent:
                return f"Blocking opponent's winning move at position {i}"
    
    # Take center if available
    if board[4] == " ":
        return "Taking center position for strategic advantage"
    
    # Take corners
    corners = [0, 2, 6, 8]
    for corner in corners:
        if board[corner] == " ":
            return f"Taking corner position {corner} for better control"
    
    # Take sides
    sides = [1, 3, 5, 7]
    for side in sides:
        if board[side] == " ":
            return f"Taking side position {side} as best available move"
    
    return "Making a random valid move"

def extract_move_number(response_text: str) -> int:
    """Extract a valid move number from AI response text."""
    response_text = response_text.strip()
    
    # First try direct integer conversion
    if response_text.isdigit():
        move = int(response_text)
        if 0 <= move <= 8:
            return move
    
    # Look for single digits in the text
    digits = re.findall(r'\b[0-8]\b', response_text)
    if digits:
        return int(digits[0])
    
    # Look for position/move keywords
    pattern = r'(?:position|move|space|square)\s*(\d)'
    match = re.search(pattern, response_text.lower())
    if match:
        return int(match.group(1))
    
    raise ValueError(f"Could not extract valid move from response: {response_text}")

# Initialize AI clients
try:
    player_x = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=60
    )
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

try:
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {str(e)}")
    raise

def create_game_prompt(board: List[str], player_symbol: str, game_history: Dict, current_game: int) -> str:
    """Create improved game prompt with better context."""
    history_summary = ""
    if current_game > 1:
        wins_x = sum(1 for game in game_history.values() if game.get('winner') == 'X')
        wins_o = sum(1 for game in game_history.values() if game.get('winner') == 'O')
        ties = sum(1 for game in game_history.values() if game.get('winner') == 'Tie')
        
        last_game = game_history.get(current_game - 1, {})
        last_winner = last_game.get('winner', 'Unknown')
        last_moves = last_game.get('moves', [])
        
        history_summary = f"""
Previous Game Summary:
- Game {current_game-1} Winner: {last_winner}
- Last Game Moves: {', '.join(f"{m['player']}:{m['position']}" for m in last_moves)}

Current Series Stats:
- X (OpenAI) wins: {wins_x}
- O (Anthropic) wins: {wins_o}
- Ties: {ties}
"""

    return f"""You are playing a game of Tic Tac Toe as player {player_symbol}. This is game {current_game} of 10.

{history_summary}
Current board state:
{print_board(board)}

CRITICAL INSTRUCTION: Respond with ONLY a single digit (0-8) representing your move.
Do not include any explanations or additional text.

Valid positions are numbered:
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8

Only empty positions are valid moves. Respond with just the position number."""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception))
)
def get_ai_move(client, prompt: str, is_openai: bool = True) -> tuple[int, str]:
    """Get move from AI player with improved error handling."""
    try:
        if is_openai:
            response = client.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
        else:
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text.strip()
            
        logger.debug(f"Raw AI response: {response_text}")
        move = extract_move_number(response_text)
        
        if not (0 <= move <= 8):
            raise ValueError(f"Move {move} out of valid range (0-8)")
            
        return move, "Success"
        
    except Exception as e:
        logger.error(f"AI move error: {str(e)}")
        raise

def make_move(state: GameState, player: str, client, is_openai: bool = True) -> GameState:
    """Make a move in the game."""
    board = state["board"]
    valid_moves = get_valid_moves(board)
    
    if not valid_moves:
        if not check_winner(board):
            state["game_history"][state["current_game"]]["winner"] = "Tie"
        raise ValueError("No valid moves available")
    
    try:
        prompt = create_game_prompt(board, player, state["game_history"], state["current_game"])
        move, status = get_ai_move(client, prompt, is_openai)
        
        if not is_valid_move(board, move):
            move = valid_moves[0]
            status = "Fallback move - invalid position returned"
            logger.warning(f"Invalid move by {player}, using fallback")
        
    except Exception as e:
        logger.error(f"Error in make_move for {player}: {str(e)}")
        move = valid_moves[0]
        status = f"Emergency fallback: {str(e)}"
    
    board[move] = player
    reasoning = analyze_board(board, player)
    winner = check_winner(board)
    
    if state["current_game"] not in state["game_history"]:
        state["game_history"][state["current_game"]] = {
            "moves": [],
            "winner": "",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    state["game_history"][state["current_game"]]["moves"].append({
        "player": player,
        "position": move,
        "reasoning": reasoning,
        "status": status
    })
    
    if winner:
        state["game_history"][state["current_game"]]["winner"] = winner
    
    logger.info(f"\nPlayer {player} placed at position {move}")
    logger.info(f"Reasoning: {reasoning}")
    if status != "Success":
        logger.info(f"Note: {status}")
    print(print_board(board))
    
    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=f"Player {player} placed at {move} - {reasoning}")],
        "next_agent": "player_o" if player == "X" and not winner else "player_x" if not winner else END,
        "board": board,
        "game_over": bool(winner),
        "last_winner": winner if winner else state["last_winner"],
        "reasoning": state["reasoning"] + [reasoning]
    }

def player_x_node(state: GameState) -> GameState:
    return make_move(state, "X", player_x, True)

def player_o_node(state: GameState) -> GameState:
    return make_move(state, "O", anthropic, False)

def play_games(num_games: int = 10):
    """Play multiple games between AI players."""
    workflow = StateGraph(GameState)
    workflow.add_node("player_x", player_x_node)
    workflow.add_node("player_o", player_o_node)
    
    def route_step(state: GameState) -> str:
        if state["game_over"]:
            if state["current_game"] < num_games:
                state["board"] = [" "] * 9
                state["game_over"] = False
                state["current_game"] += 1
                state["messages"] = []
                state["reasoning"] = []
                
                winner = state["last_winner"]
                if winner == "Tie":
                    return "player_x" if state["current_game"] % 2 == 1 else "player_o"
                return "player_x" if winner == "O" else "player_o"
            return END
        return state["next_agent"]
    
    workflow.add_edge(START, "player_x")
    workflow.add_conditional_edges("player_x", route_step)
    workflow.add_conditional_edges("player_o", route_step)
    
    chain = workflow.compile()
    
    initial_state = {
        "messages": [],
        "next_agent": "player_x",
        "board": [" "] * 9,
        "game_over": False,
        "game_history": {},
        "current_game": 1,
        "last_winner": "",
        "reasoning": []
    }
    
    logger.info(f"\nStarting {num_games} games between OpenAI (X) and Anthropic (O)...")
    
    try:
        result = chain.invoke(initial_state)
        
        # Print final statistics
        print("\nFinal Statistics:")
        wins_x = sum(1 for game in result["game_history"].values() if game["winner"] == "X")
        wins_o = sum(1 for game in result["game_history"].values() if game["winner"] == "O")
        ties = sum(1 for game in result["game_history"].values() if game["winner"] == "Tie")
        
        print(f"\nTotal Games: {num_games}")
        print(f"OpenAI (X) wins: {wins_x}")
        print(f"Anthropic (O) wins: {wins_o}")
        print(f"Ties: {ties}")
        
        print("\nDetailed Game History:")
        for game_num, game_data in result["game_history"].items():
            print(f"\nGame {game_num}:")
            print(f"Winner: {game_data['winner']}")
            print("Moves:")
            for move in game_data["moves"]:
                print(f"- Player {move['player']} -> Position {move['position']}")
                print(f"  Reasoning: {move['reasoning']}")
                if move['status'] != "Success":
                    print(f"  Note: {move['status']}")
        
        # Save game history
        filename = f"game_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(result["game_history"], f, indent=2)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during game execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        play_games(10)
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")