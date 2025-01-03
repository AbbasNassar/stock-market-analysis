document.addEventListener('DOMContentLoaded', function () {
    const rows = document.querySelectorAll('table tr');

    rows.forEach(row => {
        row.addEventListener('click', () => {
            const stockName = row.cells[0]?.textContent.trim();
            if (stockName && stockName !== "Stock") {
                fetch('/table', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ stock_name: stockName })
                })
                .then(response => {
                    if (response.redirected) {
                        window.location.href = response.url;
                    } else {
                        return response.json();
                    }
                })
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        });
    });
});
