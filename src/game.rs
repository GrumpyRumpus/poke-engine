use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::mcts::{perform_mcts, MctsResult};
use crate::state::State;
use rand::prelude::*;
use rand::rng;
use std::time::Duration;

/// Result of a completed game.
pub struct GameResult {
    /// 1.0 = side_one won, -1.0 = side_two won, 0.0 = draw/timeout
    pub winner: f32,
    /// Number of turns played
    pub turns: u32,
}

/// Play a complete game: MCTS (side_one) vs MCTS (side_two).
///
/// side_one uses `s1_search_ms` per turn, side_two uses `s2_search_ms`.
/// Set s2_search_ms to a small value (e.g. 50) for a heuristic-level opponent.
pub fn play_game(
    state: &mut State,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> GameResult {
    let mut rng = rng();

    for turn in 0..max_turns {
        // check if battle is over
        let result = state.battle_is_over();
        if result != 0.0 {
            return GameResult {
                winner: result,
                turns: turn,
            };
        }

        // get available moves for both sides
        let (s1_options, s2_options) = state.get_all_options();

        // if either side has no options, game is stuck
        if s1_options.is_empty() || s2_options.is_empty() {
            return GameResult {
                winner: state.battle_is_over(),
                turns: turn,
            };
        }

        // P1: MCTS search
        let s1_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s1_search_ms),
        );
        let s1_move = s1_result
            .s1
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s1_options[0].clone());

        // P2: MCTS search (short time = heuristic)
        let s2_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s2_search_ms),
        );
        let s2_move = s2_result
            .s2
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s2_options[0].clone());

        // generate all possible outcomes for this move pair
        let instructions = generate_instructions_from_move_pair(state, &s1_move, &s2_move, true);

        if instructions.is_empty() {
            return GameResult {
                winner: 0.0,
                turns: turn,
            };
        }

        // sample an outcome weighted by probability
        let total_pct: f32 = instructions.iter().map(|i| i.percentage).sum();
        let roll = rng.random::<f32>() * total_pct;
        let mut cumulative = 0.0;
        let mut chosen_idx = 0;
        for (i, inst) in instructions.iter().enumerate() {
            cumulative += inst.percentage;
            if roll <= cumulative {
                chosen_idx = i;
                break;
            }
        }

        // apply the chosen outcome
        state.apply_instructions(&instructions[chosen_idx].instruction_list);
    }

    // hit max turns = draw
    GameResult {
        winner: 0.0,
        turns: max_turns,
    }
}

/// Play multiple games and return (s1_wins, s2_wins, draws).
pub fn play_games(
    state_template: &State,
    n_games: u32,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> (u32, u32, u32) {
    let mut s1_wins = 0u32;
    let mut s2_wins = 0u32;
    let mut draws = 0u32;

    for _ in 0..n_games {
        let mut state = state_template.clone();
        let result = play_game(&mut state, s1_search_ms, s2_search_ms, max_turns);
        match result.winner {
            w if w > 0.0 => s1_wins += 1,
            w if w < 0.0 => s2_wins += 1,
            _ => draws += 1,
        }
    }

    (s1_wins, s2_wins, draws)
}
