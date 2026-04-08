// benchmark: expectiminimax vs MCTS across opponents
//
// Usage:
//   cargo run --release --features gen2,policy --example expectiminimax_bench -- \
//     --states opponent_states.tsv --games 4 --s1-ms 1000 --s2-ms 500

use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::mcts::perform_mcts;
use poke_engine::search::iterative_deepen_expectiminimax;
use poke_engine::search::pick_safest;
use poke_engine::state::State;
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;
use std::time::{Duration, Instant};

struct GameResult {
    winner: f32,
}

/// Play a game: side_one uses expectiminimax, side_two uses MCTS
fn play_game_emm_vs_mcts(
    state: &mut State,
    s1_ms: u64,
    s2_ms: u64,
    max_turns: u32,
) -> GameResult {
    let mut rng = rng();

    for _turn in 0..max_turns {
        let result = state.battle_is_over();
        if result != 0.0 {
            return GameResult { winner: result };
        }

        let (s1_options, s2_options) = state.get_all_options();
        if s1_options.is_empty() || s2_options.is_empty() {
            return GameResult { winner: state.battle_is_over() };
        }

        // P1: expectiminimax
        let (emm_s1, emm_s2, scores, depth) = iterative_deepen_expectiminimax(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s1_ms),
        );
        let (best_s1_idx, _) = pick_safest(&scores, emm_s1.len(), emm_s2.len());
        let s1_move = emm_s1[best_s1_idx].clone();

        // P2: MCTS
        let s2_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s2_ms),
        );
        let s2_move = s2_result
            .s2
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s2_options[0].clone());

        let instructions = generate_instructions_from_move_pair(state, &s1_move, &s2_move, true);
        if instructions.is_empty() {
            return GameResult { winner: 0.0 };
        }

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

        state.apply_instructions(&instructions[chosen_idx].instruction_list);
    }

    GameResult { winner: 0.0 }
}

struct MatchupResult {
    name: String,
    emm_wins: u32,
    emm_losses: u32,
    mcts_wins: u32,
    mcts_losses: u32,
    games: u32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut states_path = String::new();
    let mut games_per: u32 = 4;
    let mut s1_ms: u64 = 1000;
    let mut s2_ms: u64 = 500;
    let mut max_turns: u32 = 200;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--states" => { states_path = args[i + 1].clone(); i += 2; }
            "--games" => { games_per = args[i + 1].parse().unwrap(); i += 2; }
            "--s1-ms" => { s1_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--s2-ms" => { s2_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--max-turns" => { max_turns = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    if states_path.is_empty() {
        eprintln!("usage: --states <tsv> [--games N] [--s1-ms N] [--s2-ms N]");
        std::process::exit(1);
    }

    let content = std::fs::read_to_string(&states_path).expect("failed to read states file");
    let matchups: Vec<(String, State)> = content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let mut parts = line.splitn(2, '\t');
            let name = parts.next().unwrap().to_string();
            let state_str = parts.next().unwrap();
            (name, State::deserialize(state_str))
        })
        .collect();

    println!(
        "{} opponents, {} games each, EMM {}ms vs MCTS {}ms (parallel)\n",
        matchups.len(), games_per, s1_ms, s2_ms,
    );

    let t0 = Instant::now();

    // also run MCTS vs MCTS for comparison
    let results: Vec<MatchupResult> = matchups
        .par_iter()
        .map(|(name, state_template)| {
            // EMM (side one) vs MCTS (side two)
            let mut emm_w = 0u32;
            let mut emm_l = 0u32;
            for _ in 0..games_per {
                let mut state = state_template.clone();
                let result = play_game_emm_vs_mcts(&mut state, s1_ms, s2_ms, max_turns);
                if result.winner > 0.0 { emm_w += 1; }
                else if result.winner < 0.0 { emm_l += 1; }
            }

            // MCTS vs MCTS for baseline
            let mut mcts_w = 0u32;
            let mut mcts_l = 0u32;
            for _ in 0..games_per {
                let mut state = state_template.clone();
                let result = poke_engine::game::play_game(&mut state, s1_ms, s2_ms, max_turns);
                if result.winner > 0.0 { mcts_w += 1; }
                else if result.winner < 0.0 { mcts_l += 1; }
            }

            MatchupResult {
                name: name.clone(),
                emm_wins: emm_w,
                emm_losses: emm_l,
                mcts_wins: mcts_w,
                mcts_losses: mcts_l,
                games: games_per,
            }
        })
        .collect();

    println!(
        "{:<14} {:>6} {:>6}  {:>5} {:>5}",
        "Opponent", "MCTS%", "EMM%", "MctW", "EmmW",
    );
    println!("{}", "-".repeat(46));

    let mut t_mcts_w = 0u32;
    let mut t_emm_w = 0u32;
    let mut t_games = 0u32;

    for r in &results {
        let mcts_pct = if r.mcts_wins + r.mcts_losses > 0 {
            r.mcts_wins as f64 / (r.mcts_wins + r.mcts_losses) as f64 * 100.0
        } else { 50.0 };
        let emm_pct = if r.emm_wins + r.emm_losses > 0 {
            r.emm_wins as f64 / (r.emm_wins + r.emm_losses) as f64 * 100.0
        } else { 50.0 };

        let delta = if emm_pct > mcts_pct { "+" } else if emm_pct < mcts_pct { "-" } else { "=" };

        println!(
            "{:<14} {:>5.0}% {:>5.0}%  {:>2}/{:<2} {:>2}/{:<2} {}",
            r.name, mcts_pct, emm_pct,
            r.mcts_wins, r.games, r.emm_wins, r.games, delta,
        );

        t_mcts_w += r.mcts_wins;
        t_emm_w += r.emm_wins;
        t_games += r.games;
    }

    let elapsed = t0.elapsed();
    println!("{}", "-".repeat(46));
    println!(
        "{:<14} {:>5.0}% {:>5.0}%  {:>2}/{:<2} {:>2}/{:<2}",
        "TOTAL",
        t_mcts_w as f64 / t_games as f64 * 100.0,
        t_emm_w as f64 / t_games as f64 * 100.0,
        t_mcts_w, t_games,
        t_emm_w, t_games,
    );
    println!("\nDone in {:.0}s", elapsed.as_secs_f64());
}
