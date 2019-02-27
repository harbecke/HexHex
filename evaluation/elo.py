#!/usr/bin/env python3
import subprocess
from evaluation.match import MatchResults


def export_tournament_as_pgn(filename, tournament):
    with open(filename, 'w') as file:
        for game in tournament:
            file.write(game.to_pgn())

def create_ratings(tournament):
    """needs to be called from hex home directory."""
    export_tournament_as_pgn("game_history.pgn", tournament)
    subprocess.check_call(['evaluation/call_bayeselo.sh'])

def test():
    tournament = [
        MatchResults("first model", "second model", [[0, 20], [10, 5]]),
        MatchResults("first model", "third model", [[15, 5], [1, 29]]),
        MatchResults("second model", "third model", [[10, 2], [16, 4]])
    ]
    create_ratings(tournament)

