# Poke Engine (Fork)

A fork of [pmariglia/poke-engine](https://github.com/pmariglia/poke-engine) -- a Rust engine for searching through competitive Pokemon battles (singles).

This fork adds neural network policy integration, an improved evaluation function, and a full game simulation driver, focused on Gen 2 OU.

## What's Added

### Policy Module (`src/policy.rs`)

ONNX-based neural network inference for guiding MCTS search, gated behind the `policy` Cargo feature.

- **579-dimensional feature extraction** from game state: active move properties (type, power, accuracy, STAB, effectiveness, effect flags), per-pokemon stats with boost application, side conditions, weather
- **PolicyNet**: loads an ONNX model, produces move priors via softmax with configurable temperature scaling
- **ValueNet**: loads an ONNX model, predicts win probability from position (sigmoid output)
- **PUCT selection** in MCTS: AlphaZero-style prior integration where policy network outputs bias the UCB exploration term

```rust
let policy = PolicyNet::load("model.onnx")?;
// or with temperature to soften confident priors:
let policy = PolicyNet::with_temperature("model.onnx", 5.0)?;

let (s1_priors, s2_priors) = policy.get_priors(&state, &s1_options, &s2_options);
```

### Evaluation Improvements (`src/gen2/evaluate.rs`)

- **Matchup-aware boost scoring**: attack/special attack boost values are multiplied by `threat_vs()`, which checks whether the boosted Pokemon's physical/special moves can actually hit the current defender. Fixes overvaluing boosts in immune matchups (e.g., Snorlax Curse boosts vs Ghost types).
- **Sleep Talk awareness**: sleep penalty reduced by 60% for Pokemon carrying Sleep Talk

### Search Improvements (`src/search.rs`)

- **Immune move deprioritization**: attacking moves that deal 0 damage to the defender's type are pushed to the back of the move list for alpha-beta ordering, allowing faster pruning of dead branches

### Game Driver (`src/game.rs`)

Full game simulation framework for running MCTS vs MCTS games:

- `play_game()` / `play_games()`: run games with configurable search times per side, parallelized with rayon
- `play_game_recorded()`: captures per-turn state strings and visit distributions for both sides, producing training data for policy networks
- `play_game_with_policy()`: policy-guided MCTS (PUCT priors) vs standard MCTS
- `play_game_with_policy_and_value()`: full AlphaZero-style search with policy priors + value net leaf evaluation

### MCTS Extensions (`src/mcts.rs`)

- `mcts_with_priors()`: PUCT selection using policy network priors
- `perform_mcts_with_priors()`: top-level MCTS with policy integration
- `mcts_multi()`: root-parallel MCTS over multiple sampled opponent team states
- `do_mcts_with_value_net()`: value network leaf evaluation (replaces handcrafted eval)

### Benchmark Examples (`examples/`)

- `cross_team_bench.rs`: parallel cross-team winrate matrix (policy MCTS vs standard MCTS)
- `expectiminimax_bench.rs`: EMM vs MCTS comparison across team matchups
- `policy_benchmark.rs`: policy-guided vs standard MCTS in mirror matches
- `value_bench.rs`: policy + value net vs standard MCTS

## Building

Requires Rust / Cargo. Features are used to select the Pokemon generation:

```shell
# standard build (e.g., gen2)
cargo build --release --features gen2

# with neural network policy support
cargo build --release --features gen2,policy
```

The `policy` feature pulls in [ort](https://github.com/pykeio/ort) (ONNX Runtime bindings) for model inference.

## Links

- [Upstream repository](https://github.com/pmariglia/poke-engine)
- [Python Bindings](poke-engine-py)
- [CHANGELOG](CHANGELOG.md)

## Usage

The engine can be used through several subcommands:

| Subcommand | Description |
|---|---|
| `generate-instructions` | Generate possible state transitions for a move pair |
| `expectiminimax` | Fixed-depth expectiminimax search with alpha-beta pruning |
| `iterative-deepening` | Time-limited iterative deepening expectiminimax |
| `monte-carlo-tree-search` | Time-limited MCTS |
| `calculate-damage` | Damage roll calculation |
| (no subcommand) | Interactive mode |

### Interactive Mode

```shell
poke-engine --state <state-string>
```

| Command | Short | Description |
|---|:-:|---|
| **state** *str* | s | Reset state |
| **matchup** | m | Display current state info |
| **generate-instructions** *s1* *s2* | g | Generate instructions for move pair |
| **instructions** | i | Show last generated instructions |
| **apply** *index* | a | Apply instructions to state |
| **pop** | p | Undo last applied instructions |
| **pop-all** | pa | Undo all applied instructions |
| **evaluate** | ev | Evaluate current position |
| **calculate-damage** *s1* *s2* | d | Calculate damage rolls |
| **expectiminimax** *depth* *[ab]* | e | Run expectiminimax search |
| **iterative-deepening** *ms* | id | Run iterative deepening search |
| **monte-carlo-tree-search** *ms* | mcts | Run MCTS |
| **serialize** | ser | Print serialized state string |
| **exit/quit** | q | Quit |

### State Representation

See the doctest for `State::deserialize` in [state.rs](src/state.rs) for the state string format.
