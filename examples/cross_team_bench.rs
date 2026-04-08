// cross-team benchmark: policy MCTS vs standard MCTS against multiple opponents
// runs all matchups in parallel via rayon
//
// Usage:
//   cargo run --release --features gen2,policy --example cross_team_bench -- \
//     --model policy_net_v3.onnx --states opponent_states.tsv --games 4

use poke_engine::game::{play_game, play_game_with_policy};
use poke_engine::policy::PolicyNet;
use poke_engine::state::State;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

struct MatchupResult {
    name: String,
    std_wins: u32,
    std_losses: u32,
    pol_wins: u32,
    pol_losses: u32,
    games: u32,
}

fn run_matchup(
    name: &str,
    state_template: &State,
    policy: &PolicyNet,
    games: u32,
    s1_ms: u64,
    s2_ms: u64,
    max_turns: u32,
) -> MatchupResult {
    let mut std_w = 0u32;
    let mut std_l = 0u32;
    for _ in 0..games {
        let mut state = state_template.clone();
        let result = play_game(&mut state, s1_ms, s2_ms, max_turns);
        if result.winner > 0.0 { std_w += 1; }
        else if result.winner < 0.0 { std_l += 1; }
    }

    let mut pol_w = 0u32;
    let mut pol_l = 0u32;
    for _ in 0..games {
        let mut state = state_template.clone();
        let result = play_game_with_policy(&mut state, policy, s1_ms, s2_ms, max_turns);
        if result.winner > 0.0 { pol_w += 1; }
        else if result.winner < 0.0 { pol_l += 1; }
    }

    MatchupResult {
        name: name.to_string(),
        std_wins: std_w,
        std_losses: std_l,
        pol_wins: pol_w,
        pol_losses: pol_l,
        games,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut states_path = String::new();
    let mut games_per: u32 = 4;
    let mut s1_ms: u64 = 500;
    let mut s2_ms: u64 = 50;
    let mut max_turns: u32 = 200;
    let mut temperature: f32 = 1.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model_path = args[i + 1].clone(); i += 2; }
            "--states" => { states_path = args[i + 1].clone(); i += 2; }
            "--games" => { games_per = args[i + 1].parse().unwrap(); i += 2; }
            "--s1-ms" => { s1_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--s2-ms" => { s2_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--max-turns" => { max_turns = args[i + 1].parse().unwrap(); i += 2; }
            "--temp" => { temperature = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    if model_path.is_empty() || states_path.is_empty() {
        eprintln!("usage: --model <onnx> --states <tsv> [--games N] [--s1-ms N] [--s2-ms N] [--temp F]");
        std::process::exit(1);
    }

    println!("Loading policy: {} (temp={})", model_path, temperature);
    let policy = Arc::new(
        PolicyNet::with_temperature(&model_path, temperature).expect("failed to load model")
    );

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
        "{} opponents, {} games each, {}ms vs {}ms (parallel)\n",
        matchups.len(), games_per, s1_ms, s2_ms,
    );

    let t0 = Instant::now();

    // run all matchups in parallel
    let results: Vec<MatchupResult> = matchups
        .par_iter()
        .map(|(name, state_template)| {
            run_matchup(name, state_template, &policy, games_per, s1_ms, s2_ms, max_turns)
        })
        .collect();

    // print results in original order
    println!(
        "{:<14} {:>6} {:>6}  {:>5} {:>5}",
        "Opponent", "Std%", "Pol%", "StdW", "PolW",
    );
    println!("{}", "-".repeat(44));

    let mut total_std_w = 0u32;
    let mut total_pol_w = 0u32;
    let mut total_games = 0u32;

    for r in &results {
        let std_pct = if r.std_wins + r.std_losses > 0 {
            r.std_wins as f64 / (r.std_wins + r.std_losses) as f64 * 100.0
        } else { 50.0 };
        let pol_pct = if r.pol_wins + r.pol_losses > 0 {
            r.pol_wins as f64 / (r.pol_wins + r.pol_losses) as f64 * 100.0
        } else { 50.0 };

        let delta = if pol_pct > std_pct { "+" } else if pol_pct < std_pct { "-" } else { "=" };

        println!(
            "{:<14} {:>5.0}% {:>5.0}%  {:>2}/{:<2} {:>2}/{:<2} {}",
            r.name, std_pct, pol_pct,
            r.std_wins, r.games, r.pol_wins, r.games, delta,
        );

        total_std_w += r.std_wins;
        total_pol_w += r.pol_wins;
        total_games += r.games;
    }

    let elapsed = t0.elapsed();
    println!("{}", "-".repeat(44));
    println!(
        "{:<14} {:>5.0}% {:>5.0}%  {:>2}/{:<2} {:>2}/{:<2}",
        "TOTAL",
        total_std_w as f64 / total_games as f64 * 100.0,
        total_pol_w as f64 / total_games as f64 * 100.0,
        total_std_w, total_games,
        total_pol_w, total_games,
    );
    println!("\nDone in {:.0}s", elapsed.as_secs_f64());
}
