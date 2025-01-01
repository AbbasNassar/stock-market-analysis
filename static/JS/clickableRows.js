document.addEventListener('DOMContentLoaded', function () {
    const rows = document.querySelectorAll('table tr');

    rows.forEach(row => {
        row.addEventListener('click', () => {
            const stockName = row.cells[0]?.textContent.trim(); // Get stock name from the first cell
            if (stockName && stockName !== "Stock") { // Ensure a valid stock name is clicked
                fetch('/table', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ stock_name: stockName })
                })
                .then(response => {
                    if (response.redirected) {
                        // If the response includes a redirection URL, navigate to it
                        window.location.href = response.url;
                    } else {
                        return response.json();
                    }
                })
                .then(data => {
                    if (data.error) {
                        alert(data.error); // Show an error message if the response includes one
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        });
    });
});
