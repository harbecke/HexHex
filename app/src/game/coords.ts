import { BOARD_SIZE } from "./constants";

export type VirtualNode = number | "red-top" | "red-bottom" | "blue-left" | "blue-right";

export function posToId(x: number, y: number): number {
  return x + BOARD_SIZE * y;
}

export function idToPos(id: number): [number, number] {
  const x = id % BOARD_SIZE;
  const y = (id - x) / BOARD_SIZE;
  return [x, y];
}

/** The up-to-6 board cell neighbors of a cell (no virtual border nodes). */
export function neighbors(id: number): number[] {
  const [x, y] = idToPos(id);
  const result: number[] = [];

  if (x > 0) result.push(posToId(x - 1, y));
  if (x > 0 && y < BOARD_SIZE - 1) result.push(posToId(x - 1, y + 1));
  if (y > 0) result.push(posToId(x, y - 1));
  if (y < BOARD_SIZE - 1) result.push(posToId(x, y + 1));
  if (x < BOARD_SIZE - 1 && y > 0) result.push(posToId(x + 1, y - 1));
  if (x < BOARD_SIZE - 1) result.push(posToId(x + 1, y));

  return result;
}

/** Neighbors including virtual border nodes for win detection. */
export function fullNeighbors(node: VirtualNode): VirtualNode[] {
  if (node === "red-top") {
    return Array.from({ length: BOARD_SIZE }, (_, i) => posToId(i, 0));
  }
  if (node === "red-bottom") {
    return Array.from({ length: BOARD_SIZE }, (_, i) => posToId(i, BOARD_SIZE - 1));
  }
  if (node === "blue-left") {
    return Array.from({ length: BOARD_SIZE }, (_, i) => posToId(0, i));
  }
  if (node === "blue-right") {
    return Array.from({ length: BOARD_SIZE }, (_, i) => posToId(BOARD_SIZE - 1, i));
  }

  const [x, y] = idToPos(node);
  const result: VirtualNode[] = neighbors(node);

  if (x === 0) result.push("blue-left");
  if (x === BOARD_SIZE - 1) result.push("blue-right");
  if (y === 0) result.push("red-top");
  if (y === BOARD_SIZE - 1) result.push("red-bottom");

  return result;
}
