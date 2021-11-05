# StockReduction

A simple "chess engine" that works by evaluating a very shallow search tree. While this may seem that it would lead to poor performance, the objective function of the tree search is not a static function: it is a neural function that evaluates how the game might look in the future. The ground truth of this method are deep Stockfish evaluations: this model aims to predict how Stockfish (a chess engine that depends on deep tree searches) might evaluate a given state. Because of this, the search tree can be shallow and still informed - by using an approximation of an informed search.

If you'd like to read more about this project, how it works, and how it performs, a report is available here: https://connorlennox.com/other/stock-reduction.pdf
