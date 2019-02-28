#!/usr/bin/env python3

class MatchResults:
    def __init__(self, first_model_name, second_model_name, results):
        self.first_model = first_model_name
        self.second_model = second_model_name
        self.results = results

    def to_pgn(self):
        output_lines = []
        for _ in range(self.results[0][0]):
            output_lines.append(self._result_block(black=self.first_model, white=self.second_model, white_won=False))

        for _ in range(self.results[0][1]):
            output_lines.append(self._result_block(black=self.first_model, white=self.second_model, white_won=True))

        for _ in range(self.results[1][0]):
            output_lines.append(self._result_block(black=self.second_model, white=self.first_model, white_won=True))

        for _ in range(self.results[1][1]):
            output_lines.append(self._result_block(black=self.second_model, white=self.first_model, white_won=False))
        return '\n'.join(output_lines)

    @staticmethod
    def _result_block(black, white, white_won):
        result_string = '1-0' if white_won else '0-1'  # Results always have to be written with white points - black points
        return '\n'.join([f'[White "{white}"]',
                          f'[Black "{black}"]',
                          f'[Result "{result_string}"]',
                          f'{result_string}',
                          ''])
