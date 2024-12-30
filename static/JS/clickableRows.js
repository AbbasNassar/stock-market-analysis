document.addEventListener('DOMContentLoaded', function() {
    const rows = document.querySelectorAll('table tr');
    rows.forEach(row => {
        row.addEventListener('click', () => {
            const stockName = row.cells[0]?.textContent;
            if (stockName) {
                fetch('/table', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ stock_name: stockName })
                }).then(response => response.json())

            }
        });
    });
});
