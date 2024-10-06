let playerSymbol = '';
let gameOver = false;
let board = Array(9).fill('');

function setPlayer(symbol) {
    playerSymbol = symbol;
    document.getElementById('status').innerText = `You are playing as ${playerSymbol}. Your move!`;
}

function makeMove(index) {
    if (!playerSymbol || board[index] || gameOver) return;

    board[index] = playerSymbol;
    updateBoard();

    // Send player's move to the backend
    fetch('/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            board: board,
            player_symbol: playerSymbol
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'game_over') {
            gameOver = true;
            document.getElementById('status').innerText = data.message;
        } else {
            board = data.board;
            updateBoard();

            if (data.status === 'game_over') {
                gameOver = true;
                document.getElementById('status').innerText = data.message;
            }
        }
    });
}

function updateBoard() {
    for (let i = 0; i < 9; i++) {
        document.querySelectorAll('.cell')[i].innerText = board[i];
    }
}

function resetGame() {
    board = Array(9).fill('');
    gameOver = false;
    document.getElementById('status').innerText = 'New game! Make your move.';
    updateBoard();
}
