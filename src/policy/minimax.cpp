#include <utility>
#include "state.hpp"
#include "minimax.hpp"

#include <algorithm>
#include <vector>
/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax without pruning. Caller manages memory.
 *============================================================*/

namespace {
    static const int val_of_piece[7] = {
        0,
        100,
        500,
        300,
        300,
        900,
        10000
    };

    //if eat opponent piece
    bool is_capture_move(
        const State *state,
        const Move& move
    ){
        Point dest = move.second;
        int opponent = 1 - state->player;
        return state->board.board[opponent][dest.first][dest.second] != 0;
    }

    //if become a queen
    bool pawn_to_queen(
        const State *state,
        const Move& move
    ){
        Point now = move.first;
        Point dest = move.second;
        int piece = state->board.board[state->player][now.first][now.second];

        if(piece != 1) return false;

        return dest.first == 0 || dest.first == BOARD_H - 1;
    }

    int move_order_score(
        const State *state,
        const Move& move
    ){
        Point now = move.first;
        Point dest = move.second;

        int self = state->player;
        int opponent = 1 - self;

        int atk = state->board.board[self][now.first][now.second];
        int be_atked = state->board.board[opponent][dest.first][dest.second];

        int score = 0;

        if(be_atked != 0){
            score += 10 * val_of_piece[be_atked] - val_of_piece[atk];
        }
        if(pawn_to_queen(state, move)){
            score += 3000;
        }
        return score;
    }

    std::vector<Move> ordered_moves(const State *state){
        std::vector<Move> moves = state->legal_actions;
        std::stable_sort(
            moves.begin(),
            moves.end(),
            [state](const Move& a, const Move& b){
                return move_order_score(state, a) > move_order_score(state, b);
            }
        );
        return moves;
    }
}

int MiniMax::eval_ctx(
    State *state,
    int depth,
    int alpha, int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if((ctx.nodes & 2047ULL) == 0){
        if(ctx.time_up()){
            ctx.stop = true;
        }
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [ Hackathon TODO 3-1 ]
    // return the score for a winning terminal state
    // Hint: prefer faster wins by using ply.

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }
    

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score;
        if(p.use_quie){
            score = MiniMax::quiescence(
                state,
                alpha, beta,
                p.quie_depth,
                history,
                ply,
                ctx,
                p
            );
        }
        else{
            score = state->evaluate(
                p.use_kp_eval, 
                p.use_eval_mobility, 
                &history
            ); 
        }

        history.pop(state->hash());
        return score;
    }
    

    /* === Negamax loop === */
    int best_score = M_MAX;
    bool first_move = true;

    std::vector<Move> moves = ordered_moves(state);
    for(const Move& action : moves){
        // [ Hackathon TODO 3-2 ]
        // create the child state after applying action
        if(ctx.stop) break;
        State *next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();

        // [Hackathon TODO 3-3]
        // search the child one level deeper
        // [Hackathon TODO 3-4]
        // convert raw to the current player's perspective.
        int score;
        if(first_move || !p.use_PVS){
            int raw;

            if(same){
                raw = MiniMax::eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
                score = raw;
            }
            else{
                raw = MiniMax::eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
                score = -raw;
            }
        }
        else{
            int raw;

            if(same){
                raw = MiniMax::eval_ctx(next, depth - 1, alpha, alpha + 1, history, ply + 1, ctx, p);
                score = raw;
            }
            else{
                raw = MiniMax::eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, ply + 1, ctx, p);
                score = -raw;
            }

            if(score > alpha && score < beta){
                if(same){
                    raw = MiniMax::eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
                    score = raw;
                }
                else{
                    raw = MiniMax::eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
                    score = -raw;
                }
            }
        }
        
        delete next;

        first_move = false;

        // [ Hackathon TODO 3-5 ]
        // update best_score if this child is better.
        if(score > best_score){
            best_score = score;
        }
        if(score > alpha){
            alpha = score;
        }
        if(alpha >= beta){
            break;
        }
    }

    history.pop(state->hash());
    return best_score;
}

int MiniMax::quiescence(
    State *state,
    int alpha,  int beta,
    int quie_depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if((ctx.nodes & 2047ULL) == 0){
        if(ctx.time_up()){
            ctx.stop = true;
        }
    }
    if(ctx.stop){
        return 0;
    }
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }
    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    bool in_danger = state->king_in_check();

    int stand_pat = state->evaluate(p.use_kp_eval, false, &history);

    if(quie_depth <= 0){
        return stand_pat;
    }

    if(!in_danger){
        if(stand_pat >= beta){
            return stand_pat;
        }
        if(stand_pat > alpha){
            alpha = stand_pat;
        }
    }
    

    std::vector<Move> tac_moves;

    for(const Move& move : state->legal_actions){
        if(in_danger || is_capture_move(state, move) || pawn_to_queen(state, move)){
            tac_moves.push_back(move);
        }
    }

    std::sort(
        tac_moves.begin(),
        tac_moves.end(),
        [state](const Move& a, const Move& b){
            return move_order_score(state, a) > move_order_score(state, b);
        }
    );

    for(const Move& action : tac_moves){
        if(ctx.stop) break;
        
        State *next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();

        int rep_score;
        if(next->check_repetition(history, rep_score)){
            int score = same ? rep_score : -rep_score;
            delete next;

            if(score >= beta){
                return score;
            }
            if(score > alpha){
                alpha = score;
            }
            continue;
        }

        history.push(next->hash());

        int raw;
        if(same){
            raw = MiniMax::quiescence(next, alpha, beta, quie_depth - 1, history, ply + 1, ctx, p);
        }
        else{
            raw = MiniMax::quiescence(next, -beta, -alpha, quie_depth - 1, history, ply + 1, ctx, p);
        }

        history.pop(next->hash());
        int score = same ? raw : -raw;
        delete next;

        if(score >= beta){
            return score;
        }
        if(score > alpha){
            alpha = score;
        }
    }
    return alpha;
}

/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(state->legal_actions.empty()){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        result.best_move = state->legal_actions.front();
        result.score = P_MAX;
        result.depth = depth;
        result.nodes = ctx.nodes;
        return result;
    }
    if(state->game_state == DRAW){
        result.score = 0;
        result.depth = depth;
        result.nodes = ctx.nodes;
        return result;
    }

    history.push(state->hash());
    

    int alpha = M_MAX;
    int beta = P_MAX;
    int best_score = M_MAX;

    int move_index = 0;
    int total_moves = static_cast<int>(state->legal_actions.size());


    std::vector<Move> moves = ordered_moves(state);
    bool first_move = true;

    for(const Move& action : moves){
        /* [ Hackathon TODO 4-1 ]
         * search this move like TODO 3, but starting from the root */
        if(ctx.stop){
            break;
        }

        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        
        int score;
        if(first_move || !p.use_PVS){
            int raw;

            if(same){
                raw = MiniMax::eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
                score = raw;
            }
            else{
                raw = MiniMax::eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
                score = -raw;
            }
        }
        else{
            int raw;

            if(same){
                raw = MiniMax::eval_ctx(next, depth - 1, alpha, alpha + 1, history, 1, ctx, p);
                score = raw;
            }
            else{
                raw = MiniMax::eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, 1, ctx, p);
                score = -raw;
            }

            if(score > alpha && score < beta){
                if(same){
                    raw = MiniMax::eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
                    score = raw;
                }
                else{
                    raw = MiniMax::eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
                    score = -raw;
                }
            }
        }
        
        delete next;
        first_move = false;
        if(score > best_score){
            // [ Hackathon TODO 4-2 ]
            // keep this move if it is the best so far
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({
                    result.best_move,
                    best_score,
                    depth,
                    move_index + 1,
                    total_moves
                });  
            }
        }

        if(score > alpha){
            alpha = score;
        }

        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    // update result and return
    history.pop(state->hash());
    result.score = best_score;
    result.depth = depth;
    result.nodes = ctx.nodes;
    return result;
} 


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
        {"UsePVS", "true"},
        {"UseQuiescence", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "true"},
        {"UseQuiescence", ParamDef::CHECK, "true"},
    };
}
