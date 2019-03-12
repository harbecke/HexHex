#!/usr/bin/env python3
import subprocess

from hex.elo import match


def export_tournament_as_pgn(filename, tournament):
    with open(filename, 'w') as file:
        for game in tournament:
            file.write(game.to_pgn())

def create_ratings(tournament):
    """needs to be called from hex home directory."""
    export_tournament_as_pgn("game_history.pgn", tournament)
    subprocess.check_call(['hex/elo/call_bayeselo.sh'])

def test():
    tournament = [
        match.MatchResults("first model", "second model", [[0, 20], [10, 5]]),
        match.MatchResults("first model", "third model", [[15, 5], [1, 29]]),
        match.MatchResults("second model", "third model", [[10, 2], [16, 4]])
    ]
    create_ratings(tournament)

