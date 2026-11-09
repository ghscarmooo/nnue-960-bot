import chess, requests, json, time, pickle, numpy as np, threading
from flask import Flask

TOKEN = "lip_COLE_TUA_CHAVE_AQUI"
DEPTH = 11
TIME_LIMIT = 15
NNUE_PATH = "nnue-960-treinada.pkl"

headers = {"Authorization": f"Bearer {TOKEN}"}

app = Flask(__name__)

@app.route("/")
def home():
    return "NNUE-PVS-960 ONLINE!"

# --- CARREGA NNUE ---
print("Carregando NNUE...")
with open(NNUE_PATH, "rb") as f:
    net = pickle.load(f)
W1, b1, W2, b2, W3, b3 = net["W1"], net["b1"], net["W2"], net["b2"], net["W3"], net["b3"]

def board_to_vec(board):
    vec = np.zeros(768, dtype=np.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            idx = ((0 if p.color == chess.WHITE else 1)*6 + (p.piece_type-1))*64 + sq
            vec[idx] = 1.0
    return vec

def nnue_eval(board):
    if board.is_checkmate():
        return -999999 if board.turn else 999999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    vec = board_to_vec(board)
    h1 = np.clip(W1 @ vec + b1, 0, 1)
    h2 = np.clip(W2 @ h1 + b2, 0, 1)
    out = float(np.tanh(W3 @ h2 + b3[0]))
    vals = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900}
    mat = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            mat += vals.get(p.piece_type,0) if p.color==chess.WHITE else -vals.get(p.piece_type,0)
    score = int(out * 800 * 0.7 + mat * 0.3)
    return score if board.turn == chess.WHITE else -score

TT = {}
VAL = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:20000}
nodes = 0
start_time = 0

def order_moves(board):
    def sc(m):
        if board.is_capture(m):
            v = board.piece_at(m.to_square)
            a = board.piece_at(m.from_square)
            if v and a:
                return 10*VAL[v.piece_type] - VAL[a.piece_type]
        return 0
    return sorted(board.legal_moves, key=sc, reverse=True)

def quiescence(board, alpha, beta):
    global nodes
    nodes += 1
    stand = nnue_eval(board)
    if stand >= beta: return beta
    if stand > alpha: alpha = stand
    for m in order_moves(board):
        if not board.is_capture(m): continue
        board.push(m)
        s = -quiescence(board, -beta, -alpha)
        board.pop()
        if s >= beta: return beta
        if s > alpha: alpha = s
    return alpha

def pvs(board, depth, alpha, beta):
    global nodes, start_time
    nodes += 1
    if nodes % 2048 == 0 and time.time() - start_time > TIME_LIMIT:
        raise TimeoutError
    if depth == 0: return quiescence(board, alpha, beta)
    key = board._transposition_key()
    if key in TT:
        td, ts, tf = TT[key]
        if td >= depth:
            if tf == "exact": return ts
            elif tf == "lower" and ts >= beta: return beta
            elif tf == "upper" and ts <= alpha: return alpha
    moves = order_moves(board)
    best = -9999999
    a_orig = alpha
    for i, m in enumerate(moves):
        board.push(m)
        if i == 0:
            s = -pvs(board, depth-1, -beta, -alpha)
        else:
            s = -pvs(board, depth-1, -alpha-1, -alpha)
            if alpha < s < beta:
                s = -pvs(board, depth-1, -beta, -alpha)
        board.pop()
        if s > best: best = s
        if s > alpha: alpha = s
        if alpha >= beta: break
    flag = "exact"
    if best <= a_orig: flag = "upper"
    elif best >= beta: flag = "lower"
    TT[key] = (depth, best, flag)
    return best

def find_best(board):
    global nodes, start_time
    nodes = 0
    start_time = time.time()
    best_move = None
    best_score = -99999999
    alpha = -99999999
    beta = 99999999
    try:
        for m in order_moves(board):
            board.push(m)
            s = -pvs(board, DEPTH-1, -beta, -alpha)
            board.pop()
            print(f"  {m} NNUE={s} nodes={nodes}")
            if s > best_score:
                best_score = s
                best_move = m
                alpha = s
    except TimeoutError:
        print("Tempo acabou")
    print(f"Melhor: {best_move} score {best_score}")
    return best_move

def bot_loop():
    while True:
        try:
            print("BOT NNUE-960 CONECTANDO...")
            r = requests.get("https://lichess.org/api/stream/event", headers=headers, stream=True, timeout=60)
            for line in r.iter_lines():
                if not line: continue
                event = json.loads(line)
                print("EVENTO:", event)
                if event["type"] == "challenge":
                    cid = event["challenge"]["id"]
                    requests.post(f"https://lichess.org/api/challenge/{cid}/accept", headers=headers)
                if event["type"] == "gameStart":
                    gid = event["game"]["id"]
                    gs = requests.get(f"https://lichess.org/api/bot/game/stream/{gid}", headers=headers, stream=True)
                    board = chess.Board()
                    my_color = None
                    for gline in gs.iter_lines():
                        if not gline: continue
                        st = json.loads(gline)
                        if st["type"] == "gameFull":
                            try:
                                me = requests.get("https://lichess.org/api/account", headers=headers).json()
                                my_id = me["id"]
                                my_color = chess.WHITE if st["white"]["id"] == my_id else chess.BLACK
                            except: pass
                            moves = st["state"]["moves"].split() if st["state"]["moves"] else []
                            board = chess.Board()
                            for mv in moves: board.push_uci(mv)
                        elif st["type"] == "gameState":
                            moves = st["moves"].split() if st["moves"] else []
                            board = chess.Board()
                            for mv in moves: board.push_uci(mv)
                            if st.get("status") in ["mate","resign","stalemate","draw","outoftime","aborted"]:
                                break
                            if my_color is not None and board.turn == my_color and not board.is_game_over():
                                mv = find_best(board)
                                if mv:
                                    requests.post(f"https://lichess.org/api/bot/game/{gid}/move/{mv}", headers=headers)
        except Exception as e:
            print(f"Caiu: {e} - reconectando em 5s...")
            time.sleep(5)

# Roda bot em thread separada
threading.Thread(target=bot_loop, daemon=True).start()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)