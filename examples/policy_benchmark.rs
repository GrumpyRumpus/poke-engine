// benchmark: policy-guided MCTS vs standard MCTS
//
// Usage:
//   cargo run --release --features gen2,policy --example policy_benchmark -- \
//     --model /path/to/policy_net.onnx --games 20 --s1-ms 500 --s2-ms 50

use poke_engine::game::{play_game, play_game_with_policy};
use poke_engine::policy::PolicyNet;
use poke_engine::state::State;
use std::sync::Arc;
use std::time::Instant;

const STATE_STR: &str = "SNORLAX,100,NORMAL,TYPELESS,NORMAL,TYPELESS,524,524,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,319,229,229,319,159,NONE,0,0,0,DOUBLEEDGE;false;16,CURSE;false;16,SLEEPTALK;false;16,REST;false;16,false,TYPELESS=ZAPDOS,100,ELECTRIC,FLYING,NORMAL,TYPELESS,384,384,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,279,269,349,279,299,NONE,0,0,0,THUNDERBOLT;false;16,NONE;false;16,REST;false;16,SLEEPTALK;false;16,false,TYPELESS=CLOYSTER,100,WATER,ICE,NORMAL,TYPELESS,304,304,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,289,459,269,189,239,NONE,0,0,0,SPIKES;false;16,SURF;false;16,TOXIC;false;16,EXPLOSION;false;16,false,TYPELESS=NIDOKING,100,POISON,GROUND,NORMAL,TYPELESS,366,366,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,283,253,269,249,269,NONE,0,0,0,EARTHQUAKE;false;16,ICEBEAM;false;16,THUNDERBOLT;false;16,THIEF;false;16,false,TYPELESS=GENGAR,100,GHOST,POISON,NORMAL,TYPELESS,324,324,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,229,219,359,249,319,NONE,0,0,0,DYNAMICPUNCH;false;16,THUNDERBOLT;false;16,ICEPUNCH;false;16,EXPLOSION;false;16,false,TYPELESS=TYRANITAR,100,ROCK,DARK,NORMAL,TYPELESS,404,404,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,367,319,289,299,221,NONE,0,0,0,ROCKSLIDE;false;16,PURSUIT;false;16,EARTHQUAKE;false;16,ROAR;false;16,false,TYPELESS=0=0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0==0;0;0;0;0;0=0=0=0=0=0=0=0=0=0=0=0=0=false=NONE=false=false=false=move:none=false/SNORLAX,100,NORMAL,TYPELESS,NORMAL,TYPELESS,524,524,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,319,229,229,319,159,NONE,0,0,0,DOUBLEEDGE;false;16,CURSE;false;16,SLEEPTALK;false;16,REST;false;16,false,TYPELESS=ZAPDOS,100,ELECTRIC,FLYING,NORMAL,TYPELESS,384,384,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,279,269,349,279,299,NONE,0,0,0,THUNDERBOLT;false;16,NONE;false;16,REST;false;16,SLEEPTALK;false;16,false,TYPELESS=CLOYSTER,100,WATER,ICE,NORMAL,TYPELESS,304,304,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,289,459,269,189,239,NONE,0,0,0,SPIKES;false;16,SURF;false;16,TOXIC;false;16,EXPLOSION;false;16,false,TYPELESS=NIDOKING,100,POISON,GROUND,NORMAL,TYPELESS,366,366,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,283,253,269,249,269,NONE,0,0,0,EARTHQUAKE;false;16,ICEBEAM;false;16,THUNDERBOLT;false;16,THIEF;false;16,false,TYPELESS=GENGAR,100,GHOST,POISON,NORMAL,TYPELESS,324,324,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,229,219,359,249,319,NONE,0,0,0,DYNAMICPUNCH;false;16,THUNDERBOLT;false;16,ICEPUNCH;false;16,EXPLOSION;false;16,false,TYPELESS=TYRANITAR,100,ROCK,DARK,NORMAL,TYPELESS,404,404,NOABILITY,NOABILITY,LEFTOVERS,SERIOUS,85;85;85;85;85;85,367,319,289,299,221,NONE,0,0,0,ROCKSLIDE;false;16,PURSUIT;false;16,EARTHQUAKE;false;16,ROAR;false;16,false,TYPELESS=0=0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0==0;0;0;0;0;0=0=0=0=0=0=0=0=0=0=0=0=0=false=NONE=false=false=false=move:none=false/NONE;0/NONE;0/false;0/false";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::new();
    let mut n_games: u32 = 10;
    let mut s1_ms: u64 = 500;
    let mut s2_ms: u64 = 50;
    let mut max_turns: u32 = 200;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = args[i + 1].clone();
                i += 2;
            }
            "--games" => {
                n_games = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--s1-ms" => {
                s1_ms = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--s2-ms" => {
                s2_ms = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--max-turns" => {
                max_turns = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    if model_path.is_empty() {
        eprintln!("usage: --model <path/to/policy_net.onnx> [--games N] [--s1-ms N] [--s2-ms N]");
        std::process::exit(1);
    }

    println!("loading policy net: {}", model_path);
    let policy = Arc::new(PolicyNet::load(&model_path).expect("failed to load model"));
    println!("loaded\n");

    let state_template = State::deserialize(STATE_STR);

    // ---- standard MCTS ----
    println!(
        "=== Standard MCTS: {}ms vs {}ms, {} games ===",
        s1_ms, s2_ms, n_games
    );
    let t0 = Instant::now();
    let mut std_wins = 0u32;
    let mut std_losses = 0u32;
    let mut std_draws = 0u32;
    for g in 0..n_games {
        let mut state = state_template.clone();
        let result = play_game(&mut state, s1_ms, s2_ms, max_turns);
        match result.winner {
            w if w > 0.0 => std_wins += 1,
            w if w < 0.0 => std_losses += 1,
            _ => std_draws += 1,
        }
        let total = g + 1;
        print!(
            "\r  game {}/{}: {}W {}L {}D ({:.0}%)",
            total,
            n_games,
            std_wins,
            std_losses,
            std_draws,
            std_wins as f64 / total as f64 * 100.0
        );
    }
    let std_elapsed = t0.elapsed();
    println!(
        "\n  result: {}W {}L {}D ({:.0}%) in {:.1}s ({:.1}s/game)\n",
        std_wins,
        std_losses,
        std_draws,
        std_wins as f64 / n_games as f64 * 100.0,
        std_elapsed.as_secs_f64(),
        std_elapsed.as_secs_f64() / n_games as f64,
    );

    // ---- policy-guided MCTS ----
    println!(
        "=== Policy MCTS: {}ms vs {}ms, {} games ===",
        s1_ms, s2_ms, n_games
    );
    let t0 = Instant::now();
    let mut pol_wins = 0u32;
    let mut pol_losses = 0u32;
    let mut pol_draws = 0u32;
    for g in 0..n_games {
        let mut state = state_template.clone();
        let result = play_game_with_policy(&mut state, &policy, s1_ms, s2_ms, max_turns);
        match result.winner {
            w if w > 0.0 => pol_wins += 1,
            w if w < 0.0 => pol_losses += 1,
            _ => pol_draws += 1,
        }
        let total = g + 1;
        print!(
            "\r  game {}/{}: {}W {}L {}D ({:.0}%)",
            total,
            n_games,
            pol_wins,
            pol_losses,
            pol_draws,
            pol_wins as f64 / total as f64 * 100.0
        );
    }
    let pol_elapsed = t0.elapsed();
    println!(
        "\n  result: {}W {}L {}D ({:.0}%) in {:.1}s ({:.1}s/game)\n",
        pol_wins,
        pol_losses,
        pol_draws,
        pol_wins as f64 / n_games as f64 * 100.0,
        pol_elapsed.as_secs_f64(),
        pol_elapsed.as_secs_f64() / n_games as f64,
    );

    // ---- summary ----
    println!("=== Summary ===");
    println!(
        "  Standard: {:.0}% winrate, {:.1}s/game",
        std_wins as f64 / n_games as f64 * 100.0,
        std_elapsed.as_secs_f64() / n_games as f64,
    );
    println!(
        "  Policy:   {:.0}% winrate, {:.1}s/game",
        pol_wins as f64 / n_games as f64 * 100.0,
        pol_elapsed.as_secs_f64() / n_games as f64,
    );
    let overhead_pct =
        (pol_elapsed.as_secs_f64() / std_elapsed.as_secs_f64() - 1.0) * 100.0;
    println!("  Policy overhead: {:.1}%", overhead_pct);
}
