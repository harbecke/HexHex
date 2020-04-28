// TODO
// * monte carlo move selection

import {Client} from "boardgame.io/react";
import {Hexagon, HexGrid, Layout, Text} from "react-hexgrid";
import React from "react";
import {InferenceSession, Tensor} from "onnxjs";
import _ from "lodash";

import "./App.css";

const board_size = 11;
const sess = new InferenceSession();
const url = "./11_2w4_1262.onnx";

let sess_init = false;
let info = "";
let agent_is_blue = true;
let max_rating;


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

function FullNeighbors(id) {
  if (id === 'red-bottom') {
    return _.range(board_size * (board_size - 1), board_size * board_size);
  }
  if (id === 'red-top') {
    return _.range(0, board_size);
  }
  if (id === 'blue-left') {
    return _.range(0, board_size * board_size, board_size);
  }
  if (id === 'blue-right') {
    return _.range(board_size - 1, board_size * board_size, board_size);
  }
  let neighbors = Neighbors(id);
  const [x, y] = IdToPos(id);
  if (x === 0) {
    neighbors.push('blue-left');
  }
  if (x === board_size - 1) {
    neighbors.push('blue-right');
  }
  if (y === 0) {
    neighbors.push('red-top');
  }
  if (y === board_size - 1) {
    neighbors.push('red-bottom');
  }
  return neighbors;
}

/**
 * @return {boolean}
 */
function HasWinner(board) {
  const starts = ['red-top', 'blue-left'];
  const targets = ['red-bottom', 'blue-right'];
  for (let p in _.range(2)) {
    const target = targets[p];
    let toVisit = [starts[p]];
    let visited = new Set();
    while (toVisit.length > 0) {
      const node = toVisit.pop();
      for (const neighbor of FullNeighbors(node)) {
        if (neighbor === target) {
          return true;
        }
        if (typeof neighbor === 'string') {
          continue;
        }
        if (board[neighbor] === p.toString() && !visited.has(neighbor)) {
          toVisit.push(neighbor);
          visited.add(neighbor);
        }
      }
    }
  }
  return false;
}

class Toggle extends React.Component {
  constructor(props) {
    super(props);
    this.state = {showRatings: props.initial};

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
    model_display: Array(board_size * board_size).fill(""),
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
        const has_won = HasWinner(G.cells);
        G.last_move = id;
        if (has_won) {
          G.winner = cur_player;
        }
      }
    },
  },

  endIf: (G, ctx) => {
    if (G.winner !== null) {
      return {winner: G.winner};
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
          moves.push({move: "clickCell", args: [i]});
        }
      }
      return moves;
    },
  },
};

// rating: +1 = red ("0") wins
//         -1 = blue ("1") wins
// might return 0 although the opponent could force a win.
function minimax(board, depth, current_player, first_player) {
  const other_player = current_player === "0" ? "1" : "0";
  if (HasWinner(board)) {
    // other player won.
    if (current_player === "0") {
      return [-1, null]; // blue won
    } else {
      return [1, null]; // red won
    }
  }

  if (depth === 0) {
    return [0, null]; // no one won
  }
  if (current_player === "0") { // red, maximizing
    let value = -10;
    let best = null;
    for (let i = 0; i < board_size * board_size; i++) {
      if (board[i] !== null) {
        continue;
      }
      board[i] = current_player;
      const a = minimax(board, depth - 1, other_player, first_player);
      board[i] = null;
      if (a[0] > value) {
        value = a[0];
        best = i;
      }
      if (value >= 1) {
        // this is good enough
        return [value, best];
      }
      if (current_player !== first_player && value === 0) {
          // this is good enough for the second player.
          return [value, best];
      }
    }
    return [value, best];
  } else { // blue, minimizing
    let value = 10;
    let best = null;
    for (let i = 0; i < board_size * board_size; i++) {
      if (board[i] !== null) {
        continue;
      }
      board[i] = current_player;
      const a = minimax(board, depth - 1, other_player, first_player);
      board[i] = null;
      if (a[0] < value) {
        value = a[0];
        best = i;
        if (value <= -1) {
          // this is bad enough
          return [value, best];
        }
        if (current_player !== first_player && value === 0) {
          // this is good enough for the second player.
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
    this.state = {display_ratings: false};
  }

  findSureWinMove(board, player) {
    for (let depth of [1, 3]) {
      const a = minimax(board, depth, player, player);
      if (player === '0') {
        if (a[0] > 0) {
          return a[1];
        }
      } else {
        if (a[0] < 0) {
          return a[1];
        }
      }
    }

    return null;
  }

  onClick(id) {
    if (!this.isActive(id)) {
      return;
    }
    // build board for ai independently from
    // game mechanics to avoid race conditions.
    const player = agent_is_blue ? "0" : "1";
    const agent = agent_is_blue ? "1" : "0";
    let ai_board = Array.from(this.props.G.cells);
    ai_board[id] = player;

    // actually make the move
    this.props.moves.clickCell(id);

    if (HasWinner(ai_board)) {
      // player has already won.
      return;
    }

    const sure_win_move = this.findSureWinMove(ai_board, agent);
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
          //this.props.G.model_display[i] = (100 * score / score_sum).toFixed(0);
          this.props.G.model_display[i] = result[i].toFixed(1);
        } else {
          this.props.G.model_display[i] = "";
        }
      }

      let test_board = Array.from(this.props.G.cells);
      test_board[id] = player;
      test_board[best] = agent;
      const sure_win = this.findSureWinMove(test_board, player);
      if (sure_win !== null) {
        console.log("Player can surely win with the suggested move", best, "Moving to the sure win move instead.");
        best = sure_win;
      }

      this.props.moves.clickCell(best);
    });
  }

  setDisplayRatings(display_ratings) {
    this.setState({display_ratings: display_ratings});
  }

  async evalModel(input_array)
  {
      const input = [
        new Tensor(new Float32Array(input_array), "float32", [1, 2, 11, 11]),
      ];
      const output = await sess.run(input);
      return output.values().next().value.data;
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
        input_values2.push(input_values[board_size * board_size - id - 1]);
      }
      for (let id = 0; id < board_size * board_size; id++) {
        input_values2.push(input_values[2 * board_size * board_size - id - 1]);
      }

      const outputTensor = await this.evalModel(input_values);
      const outputTensor2 = await this.evalModel(input_values2);
      let average_output = [];
      for (let id = 0; id < board_size * board_size; id++) {
        average_output.push((outputTensor[id] + outputTensor2[board_size * board_size - id - 1])/2);
      }

      let final_output = [];
      if (agent_is_blue) {
        // need to transpose
        for (let x = 0; x < board_size; x++) {
          for (let y = 0; y < board_size; y++) {
            const id = PosToId(x, y);
            final_output.push(average_output[id]);
          }
        }
      } else {
        final_output = average_output;
      }

      for (let i = 0; i < board_size * board_size; i++) {
        this.props.G.model_output[i] = final_output[i];
      }

      info = "";
      return final_output;
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
        <Hexagon key={id++} cellStyle={p1Style} q={a} r={b} s={-a - b}/>
      );
      hexagons.push(
        <Hexagon key={id++} cellStyle={p2Style} q={b} r={a} s={-a - b}/>
      );
      b = board_size;
      hexagons.push(
        <Hexagon key={id++} cellStyle={p1Style} q={a} r={b} s={-a - b}/>
      );
      hexagons.push(
        <Hexagon key={id++} cellStyle={p2Style} q={b} r={a} s={-a - b}/>
      );
    }

    return (
      <div>
        <div class="intro">
          HexHex - A reinforcement deep learning agent by Simon Buchholz, David
          Harbecke, and Pascal Van Cleeff.
          <br/>
          <a href="https://github.com/harbecke/hex">github.com/harbecke/hex</a>
        </div>
        <div id="winner">{info}</div>
        <div id="controls">
          <Toggle initial={false} toggle={this.setDisplayRatings}/>
        </div>
        <div className="App">
          <HexGrid width={1000} height={800} viewBox="0 -3 30 30">
            <Layout size={{x: 1, y: 1}} flat={false} spacing={1}>
              {hexagons}
            </Layout>
          </HexGrid>
        </div>
      </div>
    );
  }
}

const App = Client({game: HexGame, board: HexBoard, debug: false});

export default App;
