import { Client } from 'boardgame.io/react';
import { HexGrid, Layout, Hexagon, Path, Hex } from 'react-hexgrid';
import React from 'react';


// Return true if `cells` is in a winning configuration.
function IsVictory(cells) {
  return false;
}

// Return true if all `cells` are occupied.
function IsDraw(cells) {
  return cells.filter(c => c === null).length == 0;
}

const board_size = 11;

const HexGame = {
  setup: () => ({ cells: Array(board_size*board_size).fill(null) }),

  moves: {
    clickCell: (G, ctx, id) => {
      if (G.cells[id] === null) {
        G.cells[id] = ctx.currentPlayer;
      }
    },
  },

  endIf: (G, ctx) => {
    if (IsVictory(G.cells)) {
      return { winner: ctx.currentPlayer };
    }
    if (IsDraw(G.cells)) {
      return { draw: true };
    }
  },

  turn: {
    moveLimit: 1,
  },
};

class HexBoard extends React.Component {
  onClick(id) {
    if (this.isActive(id)) {
      this.props.moves.clickCell(id);
      this.props.events.endTurn();
    }
  }

  cellStyle(id) {
    if (this.props.G.cells[id] === null) {
      return "emptyStyle";
    }
    if (this.props.G.cells[id] === '0') {
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
    let winner = '';
    if (this.props.ctx.gameover) {
      winner =
        this.props.ctx.gameover.winner !== undefined ? (
          <div id="winner">Winner: {this.props.ctx.gameover.winner}</div>
        ) : (
          <div id="winner">Draw!</div>
        );
    }

    const p1Style = {
      fill: 'rgb(251, 41, 67)',
      stroke: 'black',
      "stroke-width": 0.1,
    };
    const p2Style = {
      fill: 'rgb(6, 154, 243)',
      stroke: 'black',
      "stroke-width": 0.1,
    };
    const emptyStyle = {
      fill: 'white',
      stroke: 'black',
      "stroke-width": 0.1,
    }

    let hexagons = []
    for (let q = 0; q < board_size; q++) {
      for (let r = 0; r < board_size; r++) {
        const id = q + r * board_size;
        hexagons.push(<Hexagon id={id} onClick={() => this.onClick(id)} cellStyle={
          this.cellStyle(id) === 'p2Style' ? p2Style : (this.cellStyle(id) === 'p1Style' ? p1Style : emptyStyle)
        } q={q} r={r} s={-q-r} />)
      }
    }
    for (let a = 0; a < board_size; a++) {
      let b = -1;
      hexagons.push(<Hexagon cellStyle={p1Style} q={a} r={b} s={-a-b} />)
      hexagons.push(<Hexagon cellStyle={p2Style} q={b} r={a} s={-a-b} />)
      b = board_size
      hexagons.push(<Hexagon cellStyle={p1Style} q={a} r={b} s={-a-b} />)
      hexagons.push(<Hexagon cellStyle={p2Style} q={b} r={a} s={-a-b} />)
    }


    const tbody = [<div className="App">
        <HexGrid width={1000} height={800} viewBox="-2 -4 30 30">
          <Layout size={{ x: 1, y: 1 }} flat={0} spacing={1.} >
            {hexagons}
          </Layout>
        </HexGrid>
      </div>]


    return (
      <div>
        <table id="board">
          <tbody>{tbody}</tbody>
        </table>
        {winner}
      </div>
    );
  }
}

const App = Client({ game: HexGame, board: HexBoard });

export default App;

