import { Client } from "boardgame.io/react";
import { HexGrid, Layout, Hexagon } from "react-hexgrid";
import React from "react";

const board_size = 5;

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

function AddStone(G, id, player_color) {
  const [x, y] = IdToPos(id);
  let new_cc = new Set([id]);
  let new_cc_rows = player_color == 0 ? new Set([y]) : new Set([x]);

  const neighbors = new Set(Neighbors(id));
  for (let idx = 0; idx < G.connected_sets[player_color].length; idx++) {
    let cc = new Set(G.connected_sets[player_color][idx]);
    let cc_rows = new Set(G.connected_set_rows[player_color][idx]);
    const intersection = new Set([...cc].filter((x) => neighbors.has(x)));
    if (intersection.size > 0) {
      new_cc = new Set([...new_cc, ...cc]);
      new_cc_rows = new Set([...new_cc_rows, ...cc_rows]);
    }
  }

  G.connected_sets[player_color].push(Array.from(new_cc));
  G.connected_set_rows[player_color].push(Array.from(new_cc_rows));
  return new_cc_rows.has(0) && new_cc_rows.has(board_size - 1);
}

const HexGame = {
  setup: () => ({
    cells: Array(board_size * board_size).fill(null),
    connected_sets: [[], []],
    connected_set_rows: [[], []],
    winner: null,
    // index 0 will always be red independent of swap
    // save a pair of sets for each player:
    //   * the first set is a connected component of stones
    //   * the second set is the indices of these stones in the direction
    //     of the player thus the winning condition is having
    //     0 and (size-1) in one of the second sets.
  }),

  moves: {
    clickCell: (G, ctx, id) => {
      if (G.cells[id] !== null) {
        return;
      }
      G.cells[id] = ctx.currentPlayer;
      const has_won = AddStone(
        G,
        id,
        ctx.currentPlayer
      );
      if (has_won) {
        G.winner = ctx.currentPlayer;
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
          moves.push({ move: 'clickCell', args: [i] });
        }
      }
      return moves;
    },
  },
};

class HexBoard extends React.Component {
  onClick(id) {
    if (this.isActive(id)) {
      this.props.moves.clickCell(id);
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

  isActive(id) {
    if (!this.props.isActive) return false;
    if (this.props.G.cells[id] !== null) return false;
    return true;
  }

  render() {
    let winner = "";
    if (this.props.ctx.gameover) {
      winner =
        this.props.ctx.gameover.winner !== undefined ? (
          <div id="winner">Winner: {this.props.ctx.gameover.winner}</div>
        ) : (
          <div id="winner">Draw!</div>
        );
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
          />
        );
      }
    }

    // border hexagons
    for (let a = 0; a < board_size; a++) {
      let b = -1;
      hexagons.push(<Hexagon cellStyle={p1Style} q={a} r={b} s={-a - b} />);
      hexagons.push(<Hexagon cellStyle={p2Style} q={b} r={a} s={-a - b} />);
      b = board_size;
      hexagons.push(<Hexagon cellStyle={p1Style} q={a} r={b} s={-a - b} />);
      hexagons.push(<Hexagon cellStyle={p2Style} q={b} r={a} s={-a - b} />);
    }

    return (
      <div>
        <div className="App">
          <HexGrid width={1000} height={800} viewBox="-2 -4 30 30">
            <Layout size={{ x: 1, y: 1 }} flat={false} spacing={1}>
              {hexagons}
            </Layout>
          </HexGrid>
        </div>
        {winner}
      </div>
    );
  }
}

const App = Client({ game: HexGame, board: HexBoard });

export default App;
