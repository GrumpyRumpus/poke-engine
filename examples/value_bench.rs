// benchmark: policy + value MCTS vs standard MCTS
//
// Usage:
//   cargo run --release --features gen2,policy --example value_bench -- \
//     --policy policy.onnx --value value.onnx --states states.tsv \
//     --games 4 --s1-ms 1000 --s2-ms 500 --temp 5.0

use poke_engine::game::{play_game, play_game_with_policy_and_value};
use poke_engine::policy::{PolicyNet, ValueNet};
use poke_engine::state::State;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut policy_path = String::new();
    let mut value_path = String::new();
    let mut states_path = String::new();
    let mut games_per: u32 = 4;
    let mut s1_ms: u64 = 1000;
    let mut s2_ms: u64 = 500;
    let mut max_turns: u32 = 200;
    let mut temperature: f32 = 5.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--policy" => { policy_path = args[i + 1].clone(); i += 2; }
            "--value" => { value_path = args[i + 1].clone(); i += 2; }
            "--states" => { states_path = args[i + 1].clone(); i += 2; }
            "--games" => { games_per = args[i + 1].parse().unwrap(); i += 2; }
            "--s1-ms" => { s1_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--s2-ms" => { s2_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--max-turns" => { max_turns = args[i + 1].parse().unwrap(); i += 2; }
            "--temp" => { temperature = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    if policy_path.is_empty() || value_path.is_empty() || states_path.is_empty() {
        eprintln!("usage: --policy <onnx> --value <onnx> --states <tsv>");
        std::process::exit(1);
    }

    println!("Policy: {} (temp={})", policy_path, temperature);
    println!("Value:  {}", value_path);
    let policy = Arc::new(
        PolicyNet::with_temperature(&policy_path, temperature).expect("failed to load policy")
    );
    let value_net = Arc::new(
        ValueNet::load(&value_path).expect("failed to load value net")
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

    let results: Vec<(String, u32, u32, u32, u32, u32)> = matchups
        .par_iter()
        .map(|(name, state_template)| {
            // standard MCTS vs MCTS
            let mut std_w = 0u32;
            let mut std_l = 0u32;
            for _ in 0..games_per {
                let mut state = state_template.clone();
                let result = play_game(&mut state, s1_ms, s2_ms, max_turns);
                if result.winner > 0.0 { std_w += 1; }
                else if result.winner < 0.0 { std_l += 1; }
            }

            // policy + value vs MCTS
            let mut pv_w = 0u32;
            let mut pv_l = 0u32;
            for _ in 0..games_per {
                let mut state = state_template.clone();
                let result = play_game_with_policy_and_value(
                    &mut state, &policy, &value_net, s1_ms, s2_ms, max_turns,
                );
                if result.winner > 0.0 { pv_w += 1; }
                else if result.winner < 0.0 { pv_l += 1; }
            }

            (name.clone(), std_w, std_l, pv_w, pv_l, games_per)
        })
        .collect();

    println!(
        "{:<14} {:>6} {:>6}  {:>5} {:>5}",
        "Opponent", "Std%", "P+V%", "StdW", "P+VW",
    );
    println!("{}", "-".repeat(46));

    let mut t_std_w = 0u32;
    let mut t_pv_w = 0u32;
    let mut t_games = 0u32;

    for (name, std_w, std_l, pv_w, pv_l, games) in &results {
        let std_pct = if std_w + std_l > 0 {
            *std_w as f64 / (std_w + std_l) as f64 * 100.0
        } else { 50.0 };
        let pv_pct = if pv_w + pv_l > 0 {
            *pv_w as f64 / (pv_w + pv_l) as f64 * 100.0
        } else { 50.0 };

        let delta = if pv_pct > std_pct { "+" } else if pv_pct < std_pct { "-" } else { "=" };

        println!(
            "{:<14} {:>5.0}% {:>5.0}%  {:>2}/{:<2} {:>2}/{:<2} {}",
            name, std_pct, pv_pct,
            std_w, games, pv_w, games, delta,
        );

        t_std_w += std_w;
        t_pv_w += pv_w;
        t_games += games;
    }

    let elapsed = t0.elapsed();
    println!("{}", "-".repeat(46));
    println!(
        "{:<14} {:>5.0}% {:>5.0}%  {:>2}/{:<2} {:>2}/{:<2}",
        "TOTAL",
        t_std_w as f64 / t_games as f64 * 100.0,
        t_pv_w as f64 / t_games as f64 * 100.0,
        t_std_w, t_games,
        t_pv_w, t_games,
    );
    println!("\nDone in {:.0}s", elapsed.as_secs_f64());
}
