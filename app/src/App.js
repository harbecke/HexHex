// TODO
// * monte carlo move selection

import { Client } from "boardgame.io/react";
import { HexGrid, Layout, Hexagon, Text } from "react-hexgrid";
import React from "react";
import { Tensor, InferenceSession } from "onnxjs";

import "./App.css";

const board_size = 11;
const sess = new InferenceSession();
const url = "./11_2w4_1100.onnx";

let sess_init = false;
let info = "";
let agent_is_blue = true;
let max_rating;

if (!Array.prototype.last){
    Array.prototype.last = function(){
        return this[this.length - 1];
    };
};

async function load_model() {
  await sess.loadModel(url);
  sess_init = true;
}

function PosToId(x, y) {
  return x + board_size * y;
}
function IdToPos(id) {
  const x = id % board_size;
  const y = (id - x) / board_size;
  return [x, y];
}

function Neighbors(id) {
  const [x, y] = IdToPos(id);

  let neighbors = [];
  if (x !== 0) {
    neighbors.push(PosToId(x - 1, y));
  }
  if (x !== 0 && y !== board_size - 1) {
    neighbors.push(PosToId(x - 1, y + 1));
  }
  if (y !== 0) {
    neighbors.push(PosToId(x, y - 1));
  }
  if (y !== board_size - 1) {
    neighbors.push(PosToId(x, y + 1));
  }
  if (x !== board_size - 1 && y !== 0) {
    neighbors.push(PosToId(x + 1, y - 1));
  }
  if (x !== board_size - 1) {
    neighbors.push(PosToId(x + 1, y));
  }

  return neighbors;
}

function AddStone(connected_sets, connected_set_rows, id, player_color) {
  const [x, y] = IdToPos(id);
  let new_cc = new Set([id]);
  let new_cc_rows = player_color === "0" ? new Set([y]) : new Set([x]);

  const neighbors = new Set(Neighbors(id));
  for (let idx = 0; idx < connected_sets[player_color].length; idx++) {
    let cc = new Set(connected_sets[player_color][idx]);
    let cc_rows = new Set(connected_set_rows[player_color][idx]);
    const intersection = new Set([...cc].filter((x) => neighbors.has(x)));
    if (intersection.size > 0) {
      new_cc = new Set([...new_cc, ...cc]);
      new_cc_rows = new Set([...new_cc_rows, ...cc_rows]);
    }
  }

  connected_sets[player_color].push(Array.from(new_cc));
  connected_set_rows[player_color].push(Array.from(new_cc_rows));
  return new_cc_rows.has(0) && new_cc_rows.has(board_size - 1);
}

class Toggle extends React.Component {
  constructor(props) {
    super(props);
    this.state = { showRatings: props.initial };

    // This binding is necessary to make `this` work in the callback
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick() {
    this.props.toggle(!this.state.showRatings);
    this.setState((state) => ({
      showRatings: !state.showRatings,
    }));
  }

  render() {
    return (
      <button onClick={this.handleClick}>
        {this.state.showRatings ? "Hide Ratings" : "Show Ratings"}
      </button>
    );
  }
}

const HexGame = {
  setup: () => ({
    cells: Array(board_size * board_size).fill(null),
    connected_sets: [[], []],
    connected_set_rows: [[], []],
    winner: null,
    model_output: Array(board_size * board_size).fill(null),
    model_display: Array(board_size * board_size).fill(null),
    // index 0 will always be red independent of swap
    // save a pair of sets for each player:
    //   * the first set is a connected component of stones
    //   * the second set is the indices of these stones in the direction
    //     of the player thus the winning condition is having
    //     0 and (size-1) in one of the second sets.
  }),

  moves: {
    clickCell: (G, ctx, id) => {
      const num_moves = G.cells.reduce((x, y) => x + (y ? 1 : 0), 0);
      if (num_moves == 1 && G.cells[id] == "0") {
        // switch!
        agent_is_blue = !agent_is_blue;
        info = "Switched!";
      } else {
        let cur_player = ctx.currentPlayer;
        if (!agent_is_blue) {
          cur_player = ctx.currentPlayer === "0" ? "1" : "0";
        }
        G.cells[id] = cur_player;
        const has_won = AddStone(G.connected_sets, G.connected_set_rows, id, cur_player);
        G.last_move = id;
        if (has_won) {
          G.winner = cur_player;
        }
      }
    },
  },

  endIf: (G, ctx) => {
    if (G.winner !== null) {
      return { winner: G.winner };
    }
  },

  turn: {
    moveLimit: 1,
  },

  ai: {
    enumerate: (G, ctx) => {
      let moves = [];
      for (let i = 0; i < board_size * board_size; i++) {
        if (G.cells[i] === null) {
          moves.push({ move: "clickCell", args: [i] });
        }
      }
      return moves;
    },
  },
};

function minimax(board, sets, rows, depth, maximizing_player) {
    const player = agent_is_blue ? "0" : "1";
    const agent = agent_is_blue ? "1" : "0";

    if (depth === 0) {
      return [0, null];
    }
    if (maximizing_player) {
      let value = -10
      let best = null;
      for (let i = 0; i < board_size * board_size; i++) {
        if (board[i] !== null) { continue; }
        let child_board = Array.from(board);
        let child_sets = JSON.parse(JSON.stringify(sets));
        let child_rows = JSON.parse(JSON.stringify(rows));
        child_board[i] = 'a';
        if (AddStone(child_sets, child_rows, i, agent)) {
          // agent wins :-)
          // earlier = better
          return [1, i];
        }
        const a = minimax(board, child_sets, child_rows, depth - 1, false);
        if (a[0] > value) {
          value = a[0];
          best = i;
        }
        if (value >= 1) {
          // this is good enough
          return [value, best];
        }
      }
      return [value, best];
    }
    else {
      let value = 10;
      let best = null;
      for (let i = 0; i < board_size * board_size; i++) {
        if (board[i] !== null) { continue; }
        let child_board = Array.from(board);
        let child_sets = JSON.parse(JSON.stringify(sets));
        let child_rows = JSON.parse(JSON.stringify(rows));
        child_board[i] = 'p';
        if (AddStone(child_sets, child_rows, i, player)) {
          // player wins.
          return [-1, i];
        }
        const a = minimax(child_board, child_sets, child_rows, depth - 1, true);
        if (a[0] < value) {
          value = a[0];
          best = i;
          if (value <= 0) {
            // this is bad enough
            return [value, best];
          }
        }
      }
      return [value, best];
    }
  }

class HexBoard extends React.Component {
  constructor(props) {
    super(props);
    this.setDisplayRatings = this.setDisplayRatings.bind(this);
    this.state = { display_ratings: false };
  }

  findSureWinMove(board, connected_sets, connected_set_rows) {
    const depth = 3;
    const ai_first = true;
    const a = minimax(board, connected_sets, connected_set_rows, depth, ai_first);
    if (a[0] > 0) {
      return a[1];
    } 
    return null;
  }

  onClick(id) {
    if (!this.isActive(id)) {
      return;
    }
    // build board for ai independently from
    // game mechanics to avoid race conditions.
    const current_player = agent_is_blue ? "0" : "1";
    const agent = agent_is_blue ? "1" : "0";
    let ai_board = Array.from(this.props.G.cells);
    ai_board[id] = current_player;

    // actually make the move
    this.props.moves.clickCell(id);

    let connected_sets = JSON.parse(JSON.stringify(this.props.G.connected_sets));
    let connected_set_rows = JSON.parse(JSON.stringify(this.props.G.connected_set_rows));
    if (AddStone(connected_sets, connected_set_rows, id, current_player)) {
      // player has already won.
      return; 
    }

    const sure_win_move = this.findSureWinMove(ai_board, connected_sets, connected_set_rows);
    if (sure_win_move !== null) {
      console.log("Found sure win move", sure_win_move);
      this.props.moves.clickCell(sure_win_move);
      for (let i = 0; i < board_size * board_size; i++) {
        this.props.G.model_display[i] = "";
      }
      return;
    }

    this.runModel(ai_board).then((result) => {
      // AI move selection
      const num_moves = this.props.G.cells.reduce((x, y) => x + (y ? 1 : 0), 0);
      let best = -1;
      for (let i = 0; i < board_size * board_size; i++) {
        if (num_moves <= 1 || this.props.G.cells[i] === null) {
          if (best === -1 || result[i] > max_rating) {
            best = i;
            max_rating = result[i];
          }
        }
      }
      let score_sum = 0;
      for (let i = 0; i < board_size * board_size; i++) {
        if (num_moves <= 1 || this.props.G.cells[i] === null) {
          const score = Math.pow(2, result[i] - max_rating);
          score_sum += score;
        }
      }

      for (let i = 0; i < board_size * board_size; i++) {
        if (num_moves <= 1 || this.props.G.cells[i] === null) {
          const score = Math.pow(2, result[i] - max_rating);
          this.props.G.model_display[i] = (100 * score / score_sum).toFixed(0);
        } else {
          this.props.G.model_display[i] = "";
        }
      }

      this.props.moves.clickCell(best);
    });
  }

  setDisplayRatings(display_ratings) {
    this.setState({ display_ratings: display_ratings });
  }

  async runModel(cells) {
    try {
      info = "waiting for agent to move...";
      if (!sess_init) {
        await load_model();
      }

      let input_values = [];
      if (agent_is_blue) {
        for (let x = 0; x < board_size; x++) {
          for (let y = 0; y < board_size; y++) {
            const id = PosToId(x, y);
            input_values.push(cells[id] === "1" ? 1 : 0);
          }
        }
        for (let x = 0; x < board_size; x++) {
          for (let y = 0; y < board_size; y++) {
            const id = PosToId(x, y);
            input_values.push(cells[id] === "0" ? 1 : 0);
          }
        }
      } else {
        for (let y = 0; y < board_size; y++) {
          for (let x = 0; x < board_size; x++) {
            const id = PosToId(x, y);
            input_values.push(cells[id] === "0" ? 1 : 0);
          }
        }
        for (let y = 0; y < board_size; y++) {
          for (let x = 0; x < board_size; x++) {
            const id = PosToId(x, y);
            input_values.push(cells[id] === "1" ? 1 : 0);
          }
        }
      }
      let input_values2 = [];
      for (let id = 0; id < board_size * board_size; id++) {
        input_values2.push(input_values[board_size*board_size - id - 1]);
      }
      for (let id = 0; id < board_size * board_size; id++) {
        input_values2.push(input_values[2*board_size*board_size - id - 1]);
      }
      const input = [
        new Tensor(new Float32Array(input_values), "float32", [1, 2, 11, 11]),
      ];
      const input2 = [
        new Tensor(new Float32Array(input_values2), "float32", [1, 2, 11, 11]),
      ];
      const output = await sess.run(input);
      const output2 = await sess.run(input2);
      const outputTensor = output.values().next().value;
      const outputTensor2 = output2.values().next().value;
      let output_transposed = [];
      if (agent_is_blue) {
        for (let x = 0; x < board_size; x++) {
          for (let y = 0; y < board_size; y++) {
            const id = PosToId(x, y);
            output_transposed.push(
              (outputTensor.data[id] +
               outputTensor2.data[board_size * board_size - id - 1]
              ) / 2);
          }
        }
      } else {
        output_transposed = outputTensor.data;
      }

      for (let i = 0; i < board_size * board_size; i++) {
        this.props.G.model_output[i] = output_transposed[i];
      }

      info = "";
      return output_transposed;
    } catch (e) {
      console.error(e);
    }
  }

  cellStyle(id) {
    if (this.props.G.cells[id] === null) {
      return "emptyStyle";
    }
    if (this.props.G.cells[id] === "0") {
      return "p1Style";
    }
    return "p2Style";
  }

  cellText(id, display_ratings) {
    if (display_ratings) {
      return this.props.G.model_display[id];
    }
    return "";
  }

  isActive(id) {
    if (!this.props.isActive) return false;
    if (this.props.G.cells[id] !== null) return false;
    return true;
  }

  render() {
    if (this.props.ctx.gameover) {
      let player = agent_is_blue ? "0" : "1";
      info =
        this.props.ctx.gameover.winner === player
          ? "Player has won!"
          : "Agent has won!";
    }

    const p1Style = {
      fill: "rgb(251, 41, 67)",
      stroke: "black",
      strokeWidth: 0.1,
    };
    const p2Style = {
      fill: "rgb(6, 154, 243)",
      stroke: "black",
      strokeWidth: 0.1,
    };
    const emptyStyle = {
      fill: "white",
      stroke: "black",
      strokeWidth: 0.1,
    };

    let hexagons = [];
    // field hexagons, initially empty
    for (let q = 0; q < board_size; q++) {
      for (let r = 0; r < board_size; r++) {
        const id = q + r * board_size;
        hexagons.push(
          <Hexagon
            id={id}
            key={id}
            onClick={() => this.onClick(id)}
            cellStyle={
              this.cellStyle(id) === "p2Style"
                ? p2Style
                : this.cellStyle(id) === "p1Style"
                ? p1Style
                : emptyStyle
            }
            q={q}
            r={r}
            s={-q - r}
          >
            <Text>{this.cellText(id, this.state.display_ratings)}</Text>
          </Hexagon>
        );
      }
    }

    // border hexagons
    let id = board_size * board_size;
    for (let a = 0; a < board_size; a++) {
      let b = -1;
      hexagons.push(
        <Hexagon key={id++} cellStyle={p1Style} q={a} r={b} s={-a - b} />
      );
      hexagons.push(
        <Hexagon key={id++} cellStyle={p2Style} q={b} r={a} s={-a - b} />
      );
      b = board_size;
      hexagons.push(
        <Hexagon key={id++} cellStyle={p1Style} q={a} r={b} s={-a - b} />
      );
      hexagons.push(
        <Hexagon key={id++} cellStyle={p2Style} q={b} r={a} s={-a - b} />
      );
    }

    return (
      <div>
        <div class="intro">
          HexHex - A reinforcement deep learning agent by Simon Buchholz, David
          Harbecke, and Pascal Van Cleeff.
          <br />
          <a href="https://github.com/harbecke/hex">github.com/harbecke/hex</a>
        </div>
        <div id="winner">{info}</div>
        <div id="controls">
          <Toggle initial={false} toggle={this.setDisplayRatings} />
        </div>
        <div className="App">
          <HexGrid width={1000} height={800} viewBox="0 -3 30 30">
            <Layout size={{ x: 1, y: 1 }} flat={false} spacing={1}>
              {hexagons}
            </Layout>
          </HexGrid>
        </div>
      </div>
    );
  }
}

const App = Client({ game: HexGame, board: HexBoard, debug: false });

export default App;
