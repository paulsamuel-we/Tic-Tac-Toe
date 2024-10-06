document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.getElementById('board');
    const statusElement = document.getElementById('status');
    let board = Array(9).fill(0);

    function createBoard() {
        boardElement.innerHTML = '';
        board.forEach((cell, index) => {
            const cellElement = document.createElement('div');
            cellElement.classList.add('cell');
            cellElement.dataset.index = index;
            cellElement.innerText = cell === 1 ? 'X' : cell === -1 ? 'O' : '';
            cellElement.addEventListener('click', () => handleCellClick(index));
            boardElement.appendChild(cellElement);
        });
    }

    function handleCellClick(index) {
        if (board[index] !== 0) return;

        board[index] = -1;  // Human move (O)
        updateBoard();

        fetch('/move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ move: index })
        })
        .then(response => response.json())
        .then(data => {
            board = data.board;
            if (data.done) {
                statusElement.innerText = data.winner === 'human' ? "You win!" : "It's a draw!";
            } else {
                updateBoard();
            }
        });
    }

    function updateBoard() {
        createBoard();
    }

    // Initialize the game
    fetch('/init')
    .then(response => response.json())
    .then(data => {
        board = data.board;
        createBoard();
    });
});
